# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig
from cosmos_predict2.config import InferenceArguments, SetupArguments, path_to_str

_REASON1_REPO_ID = "nvidia/Cosmos-Reason1-7B"
_REASON1_CHECKPOINT_ENV = "COSMOS_PREDICT2_REASON1_CKPT_PATH"
_EMBEDDINGS_DIR_ENV = "COSMOS_PREDICT2_PRECOMPUTED_EMBEDDINGS_DIR"
_NEGATIVE_EMBEDDING_PATH_ENV = "COSMOS_PREDICT2_NEGATIVE_EMBEDDING_PATH"
_AUTO_OFFLOAD_ENV = "COSMOS_PREDICT2_LORA_AUTO_OFFLOAD"
_DATASET_PARENT_DIRS = {"videos", "captions", "metas", "images"}
_AUTO_OFFLOAD_MAX_GPU_MEMORY_BYTES = 24 * 1024**3


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _get_hf_hub_dir() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return hf_home / "hub"


def _find_local_hf_snapshot(repo_id: str) -> Path | None:
    repo_dir = _get_hf_hub_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return snapshots[-1]


def _load_embedding_file(path: Path) -> torch.Tensor:
    embedding = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(embedding, dict):
        embedding = embedding.get("t5_text_embeddings", embedding.get("embedding"))
    if embedding is None:
        raise ValueError(f"No tensor found in {path}")
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.as_tensor(embedding)
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    elif embedding.ndim == 3 and embedding.shape[0] != 1:
        raise ValueError(f"Expected embedding batch dimension of 1 in {path}, got shape {tuple(embedding.shape)}")
    return embedding.to(dtype=torch.bfloat16).cpu()


def _get_total_cuda_memory_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    except Exception:
        return None


def _maybe_enable_auto_offload(args: SetupArguments) -> SetupArguments:
    if os.environ.get(_AUTO_OFFLOAD_ENV, "1") == "0":
        return args
    if args.offload_diffusion_model or args.offload_text_encoder or args.offload_tokenizer:
        return args

    total_memory = _get_total_cuda_memory_bytes()
    if total_memory is None or total_memory > _AUTO_OFFLOAD_MAX_GPU_MEMORY_BYTES:
        return args

    total_memory_gib = total_memory / (1024**3)
    log.warning(
        f"[Memory Optimization] Detected {total_memory_gib:.2f} GiB GPU memory. "
        "Auto-enabling staged offload for LoRA inference: text encoder -> tokenizer -> diffusion model. "
        f"Set {_AUTO_OFFLOAD_ENV}=0 to disable this behavior."
    )
    return args.model_copy(
        update={
            "offload_diffusion_model": True,
            "offload_text_encoder": True,
            "offload_tokenizer": True,
        }
    )


class LoraVideo2WorldInference(Video2WorldInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_embedding_override: torch.Tensor | None = None
        self._neg_text_embedding_override: torch.Tensor | None = None
        self._local_reason1_initialized = False

    def _expects_reason1_embeddings(self) -> bool:
        text_encoder_class = str(getattr(self.model.config, "text_encoder_class", ""))
        return text_encoder_class.startswith("reason1")

    def _resolve_local_reason1_checkpoint(self) -> str | None:
        override = os.environ.get(_REASON1_CHECKPOINT_ENV)
        if override:
            override_path = Path(override).expanduser()
            return str(override_path.resolve()) if override_path.exists() else override

        local_snapshot = _find_local_hf_snapshot(_REASON1_REPO_ID)
        if local_snapshot is not None:
            return str(local_snapshot)
        return None

    def _ensure_local_reason1_text_encoder(self) -> bool:
        if self.model.text_encoder is not None:
            return True
        if not self._expects_reason1_embeddings():
            return False
        if self._local_reason1_initialized:
            return self.model.text_encoder is not None

        self._local_reason1_initialized = True
        ckpt_path = self._resolve_local_reason1_checkpoint()
        if ckpt_path is None:
            log.warning(
                "Model expects Reason1 embeddings, but no local Cosmos-Reason1 checkpoint was found. "
                f"Set {_REASON1_CHECKPOINT_ENV} or populate the Hugging Face cache for {_REASON1_REPO_ID}."
            )
            return False

        init_device = os.environ.get("COSMOS_PREDICT2_TEXT_ENCODER_INIT_DEVICE")
        if init_device is None:
            os.environ["COSMOS_PREDICT2_TEXT_ENCODER_INIT_DEVICE"] = "cpu"

        log.info(f"Initializing local Reason1 text encoder from {ckpt_path}")
        self.model.text_encoder = TextEncoder(
            TextEncoderConfig(
                compute_online=True,
                embedding_concat_strategy="full_concat",
                ckpt_path=ckpt_path,
            ),
            device="cuda",
        )
        return True

    def _normalize_override(self, embedding: torch.Tensor | None) -> torch.Tensor | None:
        if embedding is None:
            return None
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(0)
        elif embedding.ndim != 3:
            raise ValueError(f"Expected text embedding with 2 or 3 dims, got shape {tuple(embedding.shape)}")
        if embedding.shape[0] != 1:
            raise ValueError(f"Expected text embedding batch size of 1, got shape {tuple(embedding.shape)}")
        return embedding.to(dtype=torch.bfloat16)

    @contextmanager
    def use_text_embedding_overrides(
        self,
        text_embedding: torch.Tensor | None = None,
        negative_text_embedding: torch.Tensor | None = None,
    ) -> Iterator[None]:
        prev_text = self._text_embedding_override
        prev_neg = self._neg_text_embedding_override
        self._text_embedding_override = self._normalize_override(text_embedding)
        self._neg_text_embedding_override = self._normalize_override(negative_text_embedding)
        try:
            yield
        finally:
            self._text_embedding_override = prev_text
            self._neg_text_embedding_override = prev_neg

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str,
        num_conditional_frames: int = 1,
        negative_prompt: str | None = None,
        use_neg_prompt: bool = True,
        camera=None,
        action: torch.Tensor | None = None,
    ):
        if self._text_embedding_override is None and self._neg_text_embedding_override is None:
            if self.model.text_encoder is None and self._expects_reason1_embeddings():
                if not self._ensure_local_reason1_text_encoder():
                    raise RuntimeError(
                        "This LoRA model expects Reason1 embeddings, but inference_lora.py could not find a local "
                        "Reason1 checkpoint or precomputed text embedding."
                    )
            return super()._get_data_batch_input(
                video=video,
                prompt=prompt,
                num_conditional_frames=num_conditional_frames,
                negative_prompt=negative_prompt,
                use_neg_prompt=use_neg_prompt,
                camera=camera,
                action=action,
            )

        if use_neg_prompt and negative_prompt is None:
            raise AssertionError("Negative prompt is required when use_neg_prompt is True")

        _, _, _, height, width = video.shape
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "action": action.unsqueeze(0) if action is not None else None,
            "fps": torch.randint(16, 32, (self.batch_size,)).float(),
            "padding_mask": torch.zeros(self.batch_size, 1, height, width),
            "num_conditional_frames": num_conditional_frames,
        }
        if camera is not None:
            image_size = camera.image_size
            if image_size is None:
                image_size = torch.tensor([height, width, height, width], device=video.device)
            data_batch.update(
                {
                    "intrinsics": camera.intrinsics,
                    "extrinsics": camera.extrinsics,
                    "image_size": image_size,
                }
            )

        if self._text_embedding_override is not None:
            data_batch["t5_text_embeddings"] = self._text_embedding_override
            data_batch["t5_text_mask"] = torch.ones(
                self._text_embedding_override.shape[0],
                self._text_embedding_override.shape[1],
                dtype=torch.int64,
            )
        elif self.model.text_encoder is not None or self._ensure_local_reason1_text_encoder():
            data_batch["ai_caption"] = [prompt]
            data_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [prompt], "images": None},
                input_caption_key="ai_caption",
            )
        else:
            raise RuntimeError(
                "A precomputed positive text embedding was not found, and inference_lora.py could not initialize "
                "the local Reason1 text encoder."
            )

        if use_neg_prompt:
            if self._neg_text_embedding_override is not None:
                data_batch["neg_t5_text_embeddings"] = self._neg_text_embedding_override
            elif self.model.text_encoder is not None or self._ensure_local_reason1_text_encoder():
                data_batch["neg_t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [negative_prompt], "images": None},
                    input_caption_key="ai_caption",
                )
            elif self._text_embedding_override is not None:
                log.warning(
                    "Positive prompt uses a precomputed embedding, but no Reason1 encoder or negative embedding was "
                    "available. Falling back to a zero negative embedding."
                )
                data_batch["neg_t5_text_embeddings"] = torch.zeros_like(self._text_embedding_override)

        for key, value in list(data_batch.items()):
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                data_batch[key] = value.cuda().to(dtype=torch.bfloat16)

        return data_batch


class Inference:
    def __init__(self, args: SetupArguments):
        args = _maybe_enable_auto_offload(args)
        log.debug(f"{args.__class__.__name__}({args})")

        torch.enable_grad(False)

        self.rank0 = distributed.is_rank0()
        self.setup_args = args
        experiment_opts = []
        if args.model_key.distilled:
            experiment_opts.append("model.config.init_student_with_teacher=False")
        self.pipe = LoraVideo2WorldInference(
            experiment_name=args.experiment,
            ckpt_path=args.checkpoint_path,
            s3_credential_path="",
            context_parallel_size=args.context_parallel_size,
            config_file=args.config_file,
            experiment_opts=experiment_opts,
            offload_diffusion_model=args.offload_diffusion_model,
            offload_text_encoder=args.offload_text_encoder,
            offload_tokenizer=args.offload_tokenizer,
        )
        if self.rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = args.output_dir / "config.yaml"
            LazyConfig.save_yaml(self.pipe.config, config_path)
            log.info(f"Saved config to {config_path}")

        self.guardrail_enabled = not args.disable_guardrails

        if self.rank0 and self.guardrail_enabled:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            self.text_guardrail_runner = None
            self.video_guardrail_runner = None

    def _iter_embedding_candidates(self, sample: InferenceArguments) -> Iterator[Path]:
        seen: set[Path] = set()
        stems = [sample.name]

        prompt_path = _resolve_optional_path(getattr(sample, "prompt_path", None))
        input_path = _resolve_optional_path(sample.input_path)
        for path in (prompt_path, input_path):
            if path is not None:
                stems.append(path.stem)

        embeddings_dir = os.environ.get(_EMBEDDINGS_DIR_ENV)
        if embeddings_dir:
            base_dir = Path(embeddings_dir).expanduser().resolve()
            for stem in stems:
                candidate = base_dir / f"{stem}.pt"
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate

        for source_path in (prompt_path, input_path):
            if source_path is None:
                continue
            for candidate in (
                source_path.with_suffix(".pt"),
                source_path.parent / "embeddings" / f"{source_path.stem}.pt",
            ):
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate
            if source_path.parent.name in _DATASET_PARENT_DIRS:
                candidate = source_path.parent.parent / "embeddings" / f"{source_path.stem}.pt"
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate

    def _maybe_load_precomputed_text_embedding(self, sample: InferenceArguments) -> torch.Tensor | None:
        for candidate in self._iter_embedding_candidates(sample):
            if candidate.exists():
                log.info(f"Using precomputed text embedding from {candidate}")
                return _load_embedding_file(candidate)
        return None

    def _maybe_load_negative_text_embedding(self) -> torch.Tensor | None:
        override = os.environ.get(_NEGATIVE_EMBEDDING_PATH_ENV)
        if not override:
            return None
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Negative embedding path from {_NEGATIVE_EMBEDDING_PATH_ENV} does not exist: {path}"
            )
        log.info(f"Using negative prompt embedding from {path}")
        return _load_embedding_file(path)

    def generate(self, samples: list[InferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        sample_names = [sample.name for sample in samples]
        log.info(f"Generating {len(samples)} samples: {sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir)
            if output_path is not None:
                output_paths.append(output_path)
        return output_paths

    def _generate_sample(self, sample: InferenceArguments, output_dir: Path) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        if self.rank0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

            if self.text_guardrail_runner is not None:
                if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):
                    message = f"Guardrail blocked text2world generation. Prompt: {sample.prompt}"
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    raise Exception(message)
                log.success("Passed guardrail on prompt")
            else:
                log.warning("Guardrail checks on prompt are disabled")

        text_embedding = self._maybe_load_precomputed_text_embedding(sample)
        negative_text_embedding = self._maybe_load_negative_text_embedding()

        video: torch.Tensor
        with self.pipe.use_text_embedding_overrides(text_embedding, negative_text_embedding):
            if sample.enable_autoregressive:
                log.info("Generating video with autoregressive mode...")
                video = self.pipe.generate_autoregressive_from_batch(
                    prompt=sample.prompt,
                    input_path=path_to_str(sample.input_path),
                    num_output_frames=sample.num_output_frames,
                    chunk_size=sample.chunk_size,
                    chunk_overlap=sample.chunk_overlap,
                    guidance=sample.guidance,
                    num_latent_conditional_frames=sample.num_input_frames,
                    resolution=sample.resolution,
                    seed=sample.seed,
                    negative_prompt=sample.negative_prompt,
                    num_steps=sample.num_steps,
                )
            else:
                log.info("Generating video with standard mode...")
                video = self.pipe.generate_vid2world(
                    prompt=sample.prompt,
                    input_path=path_to_str(sample.input_path),
                    guidance=sample.guidance,
                    num_video_frames=sample.num_output_frames,
                    num_latent_conditional_frames=sample.num_input_frames,
                    resolution=sample.resolution,
                    seed=sample.seed,
                    negative_prompt=sample.negative_prompt,
                    num_steps=sample.num_steps,
                )

        if self.rank0:
            video = (1.0 + video[0]) / 2

            if self.video_guardrail_runner is not None:
                log.info("Running guardrail check on video...")
                frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                if processed_frames is None:
                    message = "Guardrail blocked video2world generation."
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    raise Exception(message)
                log.success("Passed guardrail on generated video")
                processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                video = processed_video.to(video.device, dtype=video.dtype)
            else:
                log.warning("Guardrail checks on video are disabled")

            save_img_or_video(video, str(output_path), fps=16)
            log.success(f"Saved video to {output_path}.mp4")
        return f"{output_path}.mp4"
