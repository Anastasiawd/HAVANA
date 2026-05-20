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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]
OVERFIT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]
TARGET_VIDEO_FPS = 24.0


# Cosmos-NeMo-Assets dataset and dataloader for LoRA training
# Text format configuration
example_dataset_cosmos_nemo_assets_lora_txt = L(VideoDataset)(
    dataset_dir="datasets/vln",
    num_frames=93,
    video_size=(272, 480),
    target_fps=TARGET_VIDEO_FPS,
)

dataloader_train_cosmos_nemo_assets_lora_txt = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora_txt,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora_txt),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# JSON format configuration with long prompts
# Training dataset
example_dataset_cosmos_nemo_assets_lora_json_train = L(VideoDataset)(
    dataset_dir="datasets/vln_json/train",
    num_frames=93,
    video_size=(272, 480),
    caption_format="json",
    prompt_type="long",
    target_fps=TARGET_VIDEO_FPS,
)

dataloader_train_cosmos_nemo_assets_lora_json = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora_json_train,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora_json_train),
    batch_size=2,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# Validation dataset
example_dataset_cosmos_nemo_assets_lora_json_val = L(VideoDataset)(
    dataset_dir="datasets/vln_json/validation",
    num_frames=93,
    video_size=(272, 480),
    caption_format="json",
    prompt_type="long",
    random_clip=False,
    target_fps=TARGET_VIDEO_FPS,
)

dataloader_val_cosmos_nemo_assets_lora_json = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora_json_val,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora_json_val, shuffle=False),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# 1-sample overfit dataset with Cosmos-style descriptive prompts
example_dataset_cosmos_nemo_assets_lora_json_overfit_train = L(VideoDataset)(
    dataset_dir="datasets/vln_json_overfit_1sample/train",
    num_frames=93,
    video_size=(272, 480),
    caption_format="json",
    prompt_type="long",
    random_clip=False,
    target_fps=TARGET_VIDEO_FPS,
)

dataloader_train_cosmos_nemo_assets_lora_json_overfit = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora_json_overfit_train,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora_json_overfit_train),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

example_dataset_cosmos_nemo_assets_lora_json_overfit_val = L(VideoDataset)(
    dataset_dir="datasets/vln_json_overfit_1sample/validation",
    num_frames=93,
    video_size=(272, 480),
    caption_format="json",
    prompt_type="long",
    random_clip=False,
    target_fps=TARGET_VIDEO_FPS,
)

dataloader_val_cosmos_nemo_assets_lora_json_overfit = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_nemo_assets_lora_json_overfit_val,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_nemo_assets_lora_json_overfit_val, shuffle=False),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

# LoRA post-training configuration for all modes (text2world, image2world, video2world)
# Shared configuration parameters
_lora_defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    # LoRA training needs the DCP checkpointer to materialize the pretrained DiT
    # from `checkpoint.load_path` before adapter tuning begins.
    {"override /ckpt_type": "dcp"},
    "_self_",
]

_overfit_defaults = [
    f"/experiment/{OVERFIT_CHECKPOINT.experiment}",
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    {"override /ckpt_type": "dcp"},
    "_self_",
]

_lora_checkpoint_base = dict(
    load_path=DEFAULT_CHECKPOINT.s3.uri,
    load_from_object_store=dict(
        enabled=False,
    ),
    save_to_object_store=dict(
        enabled=False,
    ),
)

_overfit_checkpoint_base = dict(
    load_path=OVERFIT_CHECKPOINT.s3.uri,
    load_from_object_store=dict(
        enabled=False,
    ),
    save_to_object_store=dict(
        enabled=False,
    ),
)

_lora_optimizer = dict(
    lr=2 ** (-14.5),
    weight_decay=0.001,
)

_lora_scheduler = dict(
    f_start=[0.1],
    f_max=[0.5],
    f_min=[0.2],
    warm_up_steps=[100],
    cycle_lengths=[400000],
)

_lora_trainer = dict(
    run_validation=True,
    run_validation_on_start=True,
    validation_iter=2000,
    max_val_iter=10,
    logging_iter=100,
    max_iter=400000,
    callbacks=dict(
        heart_beat=dict(
            save_s3=False,
        ),
        iter_speed=dict(
            hit_thres=500,
            save_s3=False,
        ),
        device_monitor=dict(
            save_s3=False,
        ),
        every_n_sample_reg=dict(
            every_n=500,
            save_s3=False,
        ),
        every_n_sample_ema=dict(
            every_n=500,
            save_s3=False,
        ),
        wandb=dict(
            save_s3=False,
        ),
        wandb_10x=dict(
            save_s3=False,
        ),
        dataloader_speed=dict(
            save_s3=False,
        ),
        validation_draw_sample_reg=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=False,
            save_s3=False,
            do_x0_prediction=False,
            run_at_start=True,
        ),
        validation_draw_sample_ema=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=True,
            save_s3=False,
            do_x0_prediction=False,
            run_at_start=True,
        ),
    ),
)

_lora_model_config = dict(
    config=dict(
        # Enable LoRA training
        use_lora=True,
        # Expect precomputed Reason1 embeddings from the local dataset instead of
        # loading the 7B text encoder online during training.
        text_encoder_config=None,
        # LoRA configuration parameters
        lora_rank=32,
        lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,
        # Training configuration for all three modes
        # The model will randomly sample between 1, 2, and 3 conditional frames during training
        min_num_conditional_frames=2,  # Allow text2world (1 frame)
        max_num_conditional_frames=2,  # Allow up to video2world (2 frames)
        # Probability distribution for sampling number of conditional frames
        # This controls how often each mode is trained:
        # - 1 frame: text2world (33.3%)
        # - 2 frames: image2world (33.3%)
        # - 3 frames: video2world (33.3%)
        # conditional_frames_probs=None,
        conditional_frames_probs={2: 1.0},
        # conditional_frames_probs={1: 1.0},
        # Optional: set conditional_frame_timestep for better control
        conditional_frame_timestep=-1.0,  # Default -1 means not effective
        # Keep the default conditioning strategy
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
    ),
)

_lora_model_parallel = dict(
    context_parallel_size=1,
)

# Dedicated experiment settings for checkpoints trained on the external A800
# server. Keeping these values under a separate experiment name avoids future
# edits to the generic VLN LoRA config from accidentally breaking inference for
# that checkpoint family.
_a800_model_config = dict(
    config={
        **_lora_model_config["config"],
        "text_encoder_config": None,
        "lora_rank": 128,
        "lora_alpha": 128,
        "lora_target_modules": "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        "state_t": 8,
        "min_num_conditional_frames": 2,
        "max_num_conditional_frames": 2,
        "conditional_frames_probs": {2: 1.0},
        "conditional_frame_timestep": -1.0,
        "conditioning_strategy": "frame_replace",
        "denoise_replace_gt_frames": True,
    },
)

_overfit_scheduler = dict(
    f_start=[0.5],
    f_max=[1.0],
    f_min=[1.0],
    warm_up_steps=[20],
    cycle_lengths=[100000],
)

_overfit_trainer = dict(
    run_validation=True,
    validation_iter=2000,
    logging_iter=10,
    max_iter=2000,
    callbacks=dict(
        heart_beat=dict(
            save_s3=False,
        ),
        iter_speed=dict(
            hit_thres=200,
            save_s3=False,
        ),
        device_monitor=dict(
            save_s3=False,
        ),
        every_n_sample_reg=dict(
            every_n=100,
            save_s3=False,
        ),
        every_n_sample_ema=dict(
            every_n=100,
            save_s3=False,
        ),
        wandb=dict(
            save_s3=False,
        ),
        wandb_10x=dict(
            save_s3=False,
        ),
        dataloader_speed=dict(
            save_s3=False,
        ),
        validation_draw_sample_reg=L(ValidationDrawSample)(
            n_samples=1,
            is_ema=False,
            save_s3=False,
            do_x0_prediction=False,
        ),
        validation_draw_sample_ema=L(ValidationDrawSample)(
            n_samples=1,
            is_ema=True,
            save_s3=False,
            do_x0_prediction=False,
        ),
    ),
)

_overfit_model_config = dict(
    config=dict(
        **_lora_model_config["config"],
        conditioner=dict(
            text=dict(
                dropout_rate=0.0,
            )
        ),
    ),
)

# Text format experiment
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_lora_training_2b_cosmos_nemo_assets_txt
predict2_lora_training_2b_vln_txt = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_vln_lora",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora_txt,
    checkpoint=dict(
        **_lora_checkpoint_base,
        save_iter=500,
    ),
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_lora_model_config,
    model_parallel=_lora_model_parallel,
)

# JSON format experiment with long prompts
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_lora_training_2b_cosmos_nemo_assets_json
predict2_lora_training_2b_vln_json = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_vln_json_lora",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora_json,
    dataloader_val=dataloader_val_cosmos_nemo_assets_lora_json,
    checkpoint=dict(
        **_lora_checkpoint_base,
        save_iter=5000,
    ),
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_lora_model_config,
    model_parallel=_lora_model_parallel,
)

# JSON format experiment matching the externally trained A800 LoRA checkpoints.
# The CLI is still expected to override `checkpoint.load_path` with
# `--checkpoint-path` during inference.
predict2_lora_training_2b_vln_json_a800 = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_vln_json_lora_a800",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora_json,
    dataloader_val=dataloader_val_cosmos_nemo_assets_lora_json,
    checkpoint=dict(
        **_lora_checkpoint_base,
        save_iter=500,
    ),
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_a800_model_config,
    model_parallel=_lora_model_parallel,
)

# 1-sample overfit experiment with a post-trained 2B checkpoint and
# descriptive prompts closer to the shipped Cosmos examples.
predict2_lora_training_2b_vln_json_overfit_1sample = dict(
    defaults=_overfit_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_vln_json_overfit_1sample",
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora_json_overfit,
    dataloader_val=dataloader_val_cosmos_nemo_assets_lora_json_overfit,
    checkpoint=dict(
        **_overfit_checkpoint_base,
        save_iter=2000,
    ),
    optimizer=_lora_optimizer,
    scheduler=_overfit_scheduler,
    trainer=_overfit_trainer,
    model=_overfit_model_config,
    model_parallel=_lora_model_parallel,
)

cs = ConfigStore.instance()

# Register the configurations with Hydra ConfigStore
for _item in [
    predict2_lora_training_2b_vln_txt,
    predict2_lora_training_2b_vln_json,
    predict2_lora_training_2b_vln_json_a800,
    predict2_lora_training_2b_vln_json_overfit_1sample,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
