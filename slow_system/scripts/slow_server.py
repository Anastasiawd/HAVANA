#!/usr/bin/env python3
"""Slow Reason1 plan RPC server (cosmos-predict2 venv).

Wraps the temporal waypoint model behind a ``multiprocessing.connection``
listener so the fast process can request plan bundles without importing
Cosmos. Legacy future-video generation remains available for compatibility.

Wire format:
    request:  ("generate_plan", recent_frames_list, instruction, seed)
    response: {"ok": True, "result": {
                  "reason_context": Tensor[L,1024] fp16 CPU,
                  "context_mask": Tensor[L] bool CPU,
                  "traj_latents": Tensor[3,3584] fp16 CPU,
                  "elapsed": float,
                  "schema_version": str,
              }}

Legacy video generation remains available:
    request:  ("generate", recent_frames_list, instruction, seed)
              recent_frames_list: list[np.ndarray]  (each (H,W,3) uint8)
              instruction: str
              seed: int
    response: {"ok": True, "result": {
                  "video":  np.ndarray  (T, H, W, 3) uint8,
                  "elapsed": float,
              }}
              {"ok": False, "error": str}

Other ops:
    ("ping",)                   -> "pong"
    ("info",)                   -> {"resolution": (H,W), "num_frames": int}
    ("set_resolution", "H,W")   -> None
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

# Use the preserved, checkpoint-producing Cosmos checkout as the canonical
# inference source. The slow_system directory contains only Havana wrappers.
_HAVANA_ROOT = Path(__file__).resolve().parents[2]
_SLOW_ROOT = _HAVANA_ROOT / "slow_system"
_COSMOS_ROOT = _HAVANA_ROOT / "cosmos-predict2.5"
_COSMOS_OSS_ROOT = _COSMOS_ROOT / "packages" / "cosmos-oss"
for p in reversed((_COSMOS_ROOT, _COSMOS_OSS_ROOT)):
    if p.is_dir():
        source_path = str(p)
        while source_path in sys.path:
            sys.path.remove(source_path)
        sys.path.insert(0, source_path)


LOG = logging.getLogger("havana.slow.server")
PLAN_SCHEMA_VERSION = "havana.reason1-plan.v2"


def _setup_logger(level: int = logging.INFO) -> None:
    if LOG.handlers:
        LOG.setLevel(level)
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    LOG.addHandler(h)
    LOG.setLevel(level)
    LOG.propagate = False


# =========================================================================== #
# Config
# =========================================================================== #
@dataclass
class SlowServerConfig:
    checkpoint_path: str = str(
        _COSMOS_ROOT
        / "runs/cosmos_predict_v2p5/video2world_vln/2b_vln_temporal_wp_17f_full_finetune"
        / "checkpoints/iter_000060000"
    )
    checkpoint_filename: str = "model.pt"
    experiment: str = "predict2_video2world_training_2b_vln_temporal_waypoint_17f_full_finetune"
    config_file: str = ""
    resolution: str = "272,480"
    # `num_conditional_frames` maps to cosmos's `num_latent_conditional_frames`
    # (must be 1 or 2). The H20 LoRA training pinned both
    # max/min_num_conditional_frames=2, so we default to 2 here to stay in
    # distribution. Set to 1 only if you really want image-conditioned mode.
    num_conditional_frames: int = 2
    num_steps: int = 5
    guidance: int = 7
    offload_diffusion_model: bool = False
    offload_text_encoder: bool = False
    offload_tokenizer: bool = False
    disable_guardrails: bool = True

    # When True, after each generate() call all big submodules are moved back
    # to CPU and torch.cuda.empty_cache() is called. The next generate() pays
    # ~0.5-2s to re-upload weights but VRAM peaks between calls drop to nearly
    # zero. Recommended on 24 GB cards or smaller.
    aggressive_unload: bool = True


def _load_yaml(path: Optional[Path]) -> SlowServerConfig:
    cfg = SlowServerConfig()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        LOG.warning("slow config %s not found — using defaults", path)
        return cfg
    data = yaml.safe_load(path.read_text()) or {}
    # Accept either a top-level dict or a `slow:` sub-dict (so the same
    # policy_server.yaml can be re-used).
    if "slow" in data and isinstance(data["slow"], dict):
        data = data["slow"]
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# =========================================================================== #
# Cosmos pipe wrapper
# =========================================================================== #
def _strip_to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class CosmosSlowEngine:
    """Owns the standard ``cosmos_predict2.inference.Inference`` pipe."""

    def __init__(self, cfg: SlowServerConfig):
        self.cfg = cfg
        self._h, self._w = self._parse_resolution(cfg.resolution)
        self._inference = None
        self._pipe = None
        self._negative_prompt = None
        self._init_pipe()

    @staticmethod
    def _parse_resolution(resolution: str) -> Tuple[int, int]:
        parts = [int(x) for x in resolution.split(",")]
        if len(parts) != 2:
            raise ValueError(f"resolution must be 'H,W'; got '{resolution}'")
        return parts[0], parts[1]

    def _init_pipe(self) -> None:
        t0 = time.time()
        import cosmos_predict2.inference as canonical_inference  # noqa: WPS433
        import cosmos_predict2._src.predict2.inference.video2world as video2world_inference  # noqa: WPS433
        from cosmos_oss.init import init_environment  # noqa: WPS433
        from cosmos_predict2.config import (  # noqa: WPS433
            DEFAULT_MODEL_KEY,
            DEFAULT_NEGATIVE_PROMPT,
            SetupArguments,
        )

        expected_inference = (_COSMOS_ROOT / "cosmos_predict2" / "inference.py").resolve()
        loaded_inference = Path(canonical_inference.__file__).resolve()
        expected_video2world = (
            _COSMOS_ROOT / "cosmos_predict2" / "_src" / "predict2" / "inference" / "video2world.py"
        ).resolve()
        loaded_video2world = Path(video2world_inference.__file__).resolve()
        if loaded_inference != expected_inference or loaded_video2world != expected_video2world:
            raise RuntimeError(
                "slow server imported Cosmos from the wrong source tree: "
                f"inference={loaded_inference}, video2world={loaded_video2world}; "
                f"expected inference={expected_inference}, video2world={expected_video2world}"
            )
        LOG.info("canonical inference source: %s", loaded_inference)
        LOG.info("video2world source      : %s", loaded_video2world)
        Inference = canonical_inference.Inference

        init_environment()

        ckpt_root = Path(self.cfg.checkpoint_path).expanduser().resolve()
        ckpt = ckpt_root / self.cfg.checkpoint_filename if ckpt_root.is_dir() else ckpt_root
        if not ckpt.exists():
            raise FileNotFoundError(f"slow checkpoint not found: {ckpt}")

        config_file = self.cfg.config_file or "cosmos_predict2/_src/predict2/configs/video2world/config.py"
        LOG.info("loading cosmos pipe: experiment=%s ckpt=%s", self.cfg.experiment, ckpt)

        # Construct the pipe through the exact setup path used by
        # cosmos_predict2/inference.py.  In particular, do not substitute a
        # LoRA-specific inference wrapper here: the checkpoint and experiment
        # are interpreted by the canonical Inference implementation.
        setup_args = SetupArguments.model_validate(
            {
                "output_dir": "/tmp/havana-slow-inference",
                # SetupArguments.validate_model requires the model name even
                # though the field itself has a default.  This is the same
                # default model selected by cosmos_predict2.inference.py.
                "model": DEFAULT_MODEL_KEY.name,
                "checkpoint_path": str(ckpt),
                "experiment": self.cfg.experiment,
                "config_file": config_file,
                "context_parallel_size": int(os.environ.get("WORLD_SIZE", "1")),
                "offload_diffusion_model": self.cfg.offload_diffusion_model,
                "offload_text_encoder": self.cfg.offload_text_encoder,
                "offload_tokenizer": self.cfg.offload_tokenizer,
                "disable_guardrails": self.cfg.disable_guardrails,
            }
        )
        self._inference = Inference(setup_args)
        self._pipe = self._inference.pipe
        self._pipe.model.eval()
        self._negative_prompt = DEFAULT_NEGATIVE_PROMPT
        try:
            self._req_frames = int(self._pipe.model.tokenizer.get_pixel_num_frames(self._pipe.model.config.state_t))
        except Exception:  # noqa: BLE001
            self._req_frames = 77
        if self._req_frames != 17:
            raise RuntimeError(f"Temporal Reason1 plan contract requires 17 frames, loaded {self._req_frames}.")
        if (self._h, self._w) != (272, 480):
            raise ValueError(f"Temporal Reason1 plan contract requires resolution 272x480, got {self._h}x{self._w}.")
        if int(self.cfg.num_conditional_frames) != 2:
            raise ValueError("Temporal Reason1 plan contract requires two latent conditional frames (RGB positions 0-4).")
        self._load_diagnostics = self._diagnose_loaded_plan_model(ckpt)
        LOG.info("pipe ready in %.1fs (model_required_frames=%d, resolution=%dx%d)",
                 time.time() - t0, self._req_frames, self._h, self._w)

        # Start the server in the same low-VRAM state we want between calls.
        if self.cfg.aggressive_unload:
            self._force_unload_to_cpu()

    @property
    def info(self) -> dict:
        return {
            "resolution": (self._h, self._w),
            "num_frames": self._req_frames,
            "condition_frame_positions": (0, 1, 2, 3, 4),
            "fps": 10.0,
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "load_diagnostics": self._load_diagnostics,
        }

    def _diagnose_loaded_plan_model(self, checkpoint: Path) -> dict:
        model = self._pipe.model
        expected_name = "VLNTemporalWaypointVideo2WorldModelRectifiedFlow"
        if type(model).__name__ != expected_name:
            raise TypeError(f"Expected {expected_name}, loaded {type(model).__name__}.")
        modules = {
            "reason1": getattr(model, "trajectory_reason_encoder", None),
            "reason2_to_dit_proj": getattr(model, "reason2_to_dit_proj", None),
        }
        missing = [name for name, module in modules.items() if module is None]
        if missing:
            raise RuntimeError(f"Plan checkpoint is missing required modules: {missing}")
        counts = {name: sum(value.numel() for value in module.parameters()) for name, module in modules.items()}
        empty = [name for name, count in counts.items() if count <= 0]
        if empty:
            raise RuntimeError(f"Plan modules have no parameters after checkpoint load: {empty}")
        traj_tokens = int(getattr(modules["reason1"], "num_traj_tokens", -1))
        if traj_tokens != 3:
            raise RuntimeError(f"Temporal plan requires three trajectory tokens, loaded {traj_tokens}.")
        diagnostics = {
            "checkpoint": str(checkpoint),
            "model_class": type(model).__name__,
            "parameter_counts": counts,
            "num_trajectory_tokens": traj_tokens,
            "checkpoint_keys": self._audit_checkpoint_keys(checkpoint),
        }
        LOG.info("plan load diagnostics: %s", diagnostics)
        return diagnostics

    @staticmethod
    def _audit_checkpoint_keys(checkpoint: Path) -> dict:
        """Confirm the exported file contains every plan-specific weight family."""
        try:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(str(checkpoint), map_location="cpu", mmap=True)

        candidates = []
        if isinstance(payload, dict):
            candidates.append(payload)
            candidates.extend(value for value in payload.values() if isinstance(value, dict))
        state = max(
            candidates,
            key=lambda value: sum(torch.is_tensor(tensor) for tensor in value.values()),
            default={},
        )
        keys = [str(key) for key, value in state.items() if torch.is_tensor(value)]
        required = {
            "reason1": "trajectory_reason_encoder.",
            "reason2_to_dit_proj": "reason2_to_dit_proj.",
        }
        matches = {name: sum(prefix in key for key in keys) for name, prefix in required.items()}
        missing = [name for name, count in matches.items() if count == 0]
        del payload, candidates, state
        if missing:
            raise RuntimeError(
                f"Exported checkpoint {checkpoint} lacks required plan weight families: {missing}."
            )
        return matches

    def _build_input_video(self, recent: List[np.ndarray]) -> torch.Tensor:
        H, W, T = self._h, self._w, self._req_frames
        cond_frames = []
        for f in recent:
            img = Image.fromarray(_strip_to_rgb_u8(f)).resize((W, H), Image.BILINEAR)
            cond_frames.append(torch.from_numpy(np.asarray(img, dtype=np.uint8))
                                .permute(2, 0, 1).contiguous())
        # cosmos expects 4*(N-1)+1 raw frames for `num_latent_conditional_frames=N`.
        cond_count = max(1, 4 * (self.cfg.num_conditional_frames - 1) + 1)
        if not cond_frames:
            cond_frames = [torch.zeros(3, H, W, dtype=torch.uint8)] * cond_count
        elif len(cond_frames) < cond_count:
            cond_frames = [cond_frames[0]] * (cond_count - len(cond_frames)) + cond_frames
        else:
            cond_frames = cond_frames[-cond_count:]

        full = torch.zeros(3, T, H, W, dtype=torch.uint8)
        for i, fr in enumerate(cond_frames):
            full[:, i] = fr
        if cond_count < T:
            full[:, cond_count:] = cond_frames[-1].unsqueeze(1)
        return full.unsqueeze(0)  # (1, 3, T, H, W)

    def _force_unload_to_cpu(self) -> None:
        """Move every heavy submodule of the cosmos pipe back to CPU and free cache.

        Cosmos's built-in offload path runs *between* sampling stages but
        re-uploads modules at the end of each generate() call (notably the
        text encoder). This explicitly puts everything back on CPU so the next
        generate() starts from a clean slate.
        """
        pipe = self._pipe
        if pipe is None:
            return
        model = pipe.model

        moved = []
        # 1. Diffusion DiT + conditioner.
        try:
            model.net = model.net.to("cpu")
            moved.append("net")
        except Exception:  # noqa: BLE001
            pass
        if getattr(model, "conditioner", None) is not None:
            try:
                model.conditioner = model.conditioner.to("cpu")
                moved.append("conditioner")
            except Exception:  # noqa: BLE001
                pass

        if self.cfg.offload_text_encoder and self._offload_reason_conditioner(model):
            moved.append("trajectory_reason_conditioner")

        # 2. Tokenizer encoder + decoder.
        tok = getattr(model, "tokenizer", None)
        if tok is not None:
            for attr in ("encoder", "decoder"):
                sub = getattr(tok, attr, None)
                if sub is not None:
                    try:
                        setattr(tok, attr, sub.to("cpu"))
                        moved.append(f"tokenizer.{attr}")
                    except Exception:  # noqa: BLE001
                        pass

        # 3. Text encoder. The cosmos wrapper stores the actual nn.Module under
        #    `.model`; that's the ~7B Reason1 chunk.
        te = getattr(model, "text_encoder", None)
        if te is not None:
            inner = getattr(te, "model", None)
            if inner is not None:
                try:
                    te.model = inner.to("cpu")
                    moved.append("text_encoder.model")
                except Exception:  # noqa: BLE001
                    pass

        # 4. Drop dangling references and ask CUDA to release cached blocks.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        free, total = (None, None)
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free = free_b / 1024**3
            total = total_b / 1024**3
        except Exception:  # noqa: BLE001
            pass
        moved_str = ", ".join(moved) if moved else "(nothing)"
        if free is not None:
            LOG.info("[Unload] moved %s -> CPU; cuda free %.1f / %.1f GiB",
                     moved_str, free, total)
        else:
            LOG.info("[Unload] moved %s -> CPU", moved_str)

    @torch.no_grad()
    def generate(self, recent: List[np.ndarray], instruction: str, seed: int) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("slow pipe not initialised")
        vid_input = self._build_input_video(recent)
        LOG.info("generate: prompt='%s' input=%s frames=%d steps=%d",
                 instruction[:120], tuple(vid_input.shape), self._req_frames, self.cfg.num_steps)
        try:
            sampled = self._pipe.generate_vid2world(
                prompt=instruction,
                input_path=vid_input,
                guidance=int(self.cfg.guidance),
                num_video_frames=self._req_frames,
                num_latent_conditional_frames=int(self.cfg.num_conditional_frames),
                resolution=self.cfg.resolution,
                seed=int(seed),
                negative_prompt=self._negative_prompt,
                num_steps=int(self.cfg.num_steps),
            )
            # sampled: (1, 3, T, H, W) in [-1, 1]
            video = ((sampled + 1.0) * 0.5).clamp(0.0, 1.0)
            if video.ndim == 5:
                video = video[0]
            # (3, T, H, W) -> (T, H, W, 3) uint8 on CPU before unload so we
            # don't keep references to GPU tensors during the cleanup.
            out = (video.permute(1, 2, 3, 0) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            del sampled, video, vid_input
        finally:
            if self.cfg.aggressive_unload:
                self._force_unload_to_cpu()
        return np.ascontiguousarray(out)

    @torch.inference_mode()
    def generate_plan(self, recent: List[np.ndarray], instruction: str, seed: int) -> dict:
        """Run only temporal Reason1 and return semantic context/trajectory tokens."""
        del seed  # Reason1 eval is deterministic; kept in the wire contract.
        if self._pipe is None:
            raise RuntimeError("slow pipe not initialised")
        vid_input = self._build_input_video(recent)
        model = self._pipe.model
        t0 = time.time()
        try:
            data_batch = self._pipe._get_data_batch_input(
                video=vid_input,
                prompt=instruction,
                num_conditional_frames=int(self.cfg.num_conditional_frames),
                negative_prompt=self._negative_prompt,
                use_neg_prompt=False,
            )
            model.trajectory_reason_encoder.to(device="cuda")
            model.trajectory_reason_encoder.model.eval()
            model.reason2_to_dit_proj.to(**model.tensor_kwargs)
            reason_output = model.trajectory_reason_encoder(data_batch, model.input_caption_key)
            reason_context = model.reason2_to_dit_proj(
                reason_output.context_tokens.to(**model.tensor_kwargs)
            )

            result = {
                "reason_context": reason_context[0].detach().to(device="cpu", dtype=torch.float16).contiguous(),
                "context_mask": reason_output.context_mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                "traj_latents": reason_output.traj_latents[0]
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous(),
                "elapsed": time.time() - t0,
                "schema_version": PLAN_SCHEMA_VERSION,
                "instruction": str(instruction),
            }
            expected = {
                "reason_context": (None, 1024),
                "traj_latents": (3, 3584),
            }
            for name, shape in expected.items():
                actual = tuple(result[name].shape)
                if shape[0] is None:
                    valid = len(actual) == 2 and actual[1] == shape[1]
                else:
                    valid = actual == shape
                if not valid:
                    raise ValueError(f"Unexpected {name} shape {actual}; expected {shape}.")
            if result["context_mask"].shape != result["reason_context"].shape[:1]:
                raise ValueError("Reason1 context mask length does not match context length.")
            if not result["context_mask"].any():
                raise ValueError("Reason1 returned a plan with no valid context tokens.")
            return result
        finally:
            del vid_input
            if self.cfg.offload_text_encoder:
                self._offload_reason_conditioner(model)

    @staticmethod
    def _offload_reason_conditioner(model) -> bool:
        offload = getattr(model, "offload_reason_conditioner_for_inference", None)
        if callable(offload):
            offload()
            return True
        moved = False
        for name in ("trajectory_reason_encoder", "reason2_to_dit_proj", "waypoint_head"):
            module = getattr(model, name, None)
            if module is not None:
                setattr(model, name, module.to(device="cpu"))
                moved = True
        if moved:
            torch.cuda.empty_cache()
        return moved


# =========================================================================== #
# RPC loop
# =========================================================================== #
class SlowRpcServer:
    _OPS = {"ping", "info", "set_resolution", "generate", "generate_plan"}

    def __init__(self, engine: CosmosSlowEngine, address: str, authkey: bytes):
        self.engine = engine
        self.address = self._parse_address(address)
        self.authkey = authkey
        if isinstance(self.address, str):
            try:
                Path(self.address).unlink()
            except FileNotFoundError:
                pass
        self._listener = Listener(self.address, authkey=authkey)
        LOG.info("listening on %s", self.address)

    @staticmethod
    def _parse_address(addr: str):
        if isinstance(addr, str) and ":" in addr and not addr.startswith("/"):
            host, port = addr.rsplit(":", 1)
            return (host, int(port))
        return addr

    def serve_forever(self) -> None:
        try:
            while True:
                LOG.info("waiting for client...")
                conn = self._listener.accept()
                LOG.info("client connected from %s", self._listener.last_accepted)
                try:
                    self._serve_one(conn)
                except (EOFError, ConnectionResetError):
                    LOG.info("client disconnected")
                finally:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
        except KeyboardInterrupt:
            LOG.info("interrupted by user")
        finally:
            try:
                self._listener.close()
            except Exception:  # noqa: BLE001
                pass

    def _serve_one(self, conn) -> None:
        while True:
            try:
                req = conn.recv()
            except EOFError:
                return
            if not isinstance(req, tuple) or not req:
                conn.send({"ok": False, "error": f"bad request: {req!r}"})
                continue
            op, *args = req
            if op not in self._OPS:
                conn.send({"ok": False, "error": f"unknown op {op!r}"})
                continue
            try:
                result = self._dispatch(op, args)
                conn.send({"ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001
                LOG.error("op=%s failed: %s\n%s", op, exc, traceback.format_exc())
                conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _dispatch(self, op: str, args: list):
        if op == "ping":
            return "pong"
        if op == "info":
            return self.engine.info
        if op == "set_resolution":
            resolution = self.engine._parse_resolution(args[0])
            if resolution != (272, 480):
                raise ValueError("Temporal plan server resolution is fixed at 272,480.")
            self.engine._h, self.engine._w = resolution
            self.engine.cfg.resolution = "272,480"
            return None
        if op == "generate":
            recent_list, instruction, seed = args[0], args[1], int(args[2])
            t0 = time.time()
            video = self.engine.generate(recent_list, instruction, seed)
            elapsed = time.time() - t0
            LOG.info("generated video shape=%s in %.2fs", video.shape, elapsed)
            return {"video": video, "elapsed": elapsed}
        if op == "generate_plan":
            recent_list, instruction, seed = args[0], args[1], int(args[2])
            plan = self.engine.generate_plan(recent_list, instruction, seed)
            LOG.info(
                "generated plan context=%d trajectory_tokens=%d in %.2fs",
                int(plan["context_mask"].sum()),
                int(plan["traj_latents"].shape[0]),
                plan["elapsed"],
            )
            return plan
        raise AssertionError(f"unhandled op {op!r}")


# =========================================================================== #
# CLI
# =========================================================================== #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Havana slow-system RPC server.")
    p.add_argument("--config", type=Path, default=None,
                   help="optional YAML config (top-level or under `slow:`)")
    p.add_argument("--serve", type=str, required=True,
                   help="UNIX socket path (/tmp/havana-slow.sock) or host:port")
    p.add_argument("--authkey", type=str, default="havana-slow")
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument("--resolution", type=str, default=None,
                   help="H,W override (default 192,320)")
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--guidance", type=int, default=None)
    p.add_argument("--disable-guardrails", action="store_true", default=None,
                   help="disable text and video guardrails")
    p.add_argument("--offload-text-encoder", action="store_true", default=None,
                   help="offload the text encoder between inference stages")
    p.add_argument("--offload-tokenizer", action="store_true", default=None,
                   help="offload tokenizer encoder/decoder between inference stages")
    p.add_argument("--offload-diffusion-model", action="store_true", default=None,
                   help="offload the diffusion model between inference stages")
    return p.parse_args()


def main() -> int:
    _setup_logger()
    args = _parse_args()
    cfg = _load_yaml(args.config)
    if args.checkpoint_path:
        cfg.checkpoint_path = args.checkpoint_path
    if args.experiment:
        cfg.experiment = args.experiment
    if args.resolution:
        cfg.resolution = args.resolution
    if args.num_steps is not None:
        cfg.num_steps = int(args.num_steps)
    if args.guidance is not None:
        cfg.guidance = int(args.guidance)
    if args.disable_guardrails is not None:
        cfg.disable_guardrails = bool(args.disable_guardrails)
    if args.offload_text_encoder is not None:
        cfg.offload_text_encoder = bool(args.offload_text_encoder)
    if args.offload_tokenizer is not None:
        cfg.offload_tokenizer = bool(args.offload_tokenizer)
    if args.offload_diffusion_model is not None:
        cfg.offload_diffusion_model = bool(args.offload_diffusion_model)

    LOG.info("=" * 72)
    LOG.info("Slow RPC server starting")
    LOG.info("  checkpoint     : %s", cfg.checkpoint_path)
    LOG.info("  experiment     : %s", cfg.experiment)
    LOG.info("  resolution     : %s", cfg.resolution)
    LOG.info("  num_steps      : %d", cfg.num_steps)
    LOG.info("  guardrails     : %s", "disabled" if cfg.disable_guardrails else "enabled")
    LOG.info("  offload        : text=%s tokenizer=%s diffusion=%s",
             cfg.offload_text_encoder, cfg.offload_tokenizer, cfg.offload_diffusion_model)
    LOG.info("  serve          : %s", args.serve)
    LOG.info("=" * 72)

    engine = CosmosSlowEngine(cfg)
    rpc = SlowRpcServer(engine, args.serve, authkey=args.authkey.encode())
    rpc.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
