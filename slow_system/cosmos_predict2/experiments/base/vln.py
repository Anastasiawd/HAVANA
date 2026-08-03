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

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_vln_waypoint import (
    InternDataVLNWaypointDataset,
    get_vln_action_source_balanced_sampler,
    get_vln_waypoint_eval_sampler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_vln_waypoint import (
    get_generic_dataloader as get_vln_waypoint_generic_dataloader,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_vln_waypoint import (
    get_sampler as get_vln_waypoint_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT_2B = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


vln_cosmos_video_dataset_29f_272_480 = L(VideoDataset)(
    dataset_dir="datasets/vln_cosmos",
    num_frames=29,
    video_size=(272, 480),
    caption_format="text",
)

vln_cosmos_dataloader_train_29f_272_480 = L(get_generic_dataloader)(
    dataset=vln_cosmos_video_dataset_29f_272_480,
    sampler=L(get_sampler)(dataset=vln_cosmos_video_dataset_29f_272_480),
    batch_size=16,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
)


vln_intern_waypoint_dataset_29f_272_480 = L(InternDataVLNWaypointDataset)(
    dataset_dir="/home/csevolunt/dongzhih/dataset/InternData",
    num_frames=29,
    video_size=(272, 480),
    target_fps=10.0,
    waypoint_positions=(10, 16, 22, 28),
)

vln_intern_waypoint_dataloader_train_29f_272_480 = L(get_vln_waypoint_generic_dataloader)(
    dataset=vln_intern_waypoint_dataset_29f_272_480,
    sampler=L(get_vln_waypoint_sampler)(dataset=vln_intern_waypoint_dataset_29f_272_480),
    batch_size=16,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)


VLN_TEMPORAL_DATASET_ROOT = "/home/csevolunt/dongzhih/dataset/InternData"
VLN_TEMPORAL_AUDIT_MANIFEST = "outputs/vln_waypoint_source_audit_v2.json"
# Point these two validation constants at a held-out trusted split before
# using video turn-direction accuracy for checkpoint selection.
VLN_TEMPORAL_VAL_DATASET_ROOT = VLN_TEMPORAL_DATASET_ROOT
VLN_TEMPORAL_VAL_AUDIT_MANIFEST = VLN_TEMPORAL_AUDIT_MANIFEST

vln_intern_temporal_waypoint_dataset_17f_272_480 = L(InternDataVLNWaypointDataset)(
    dataset_dir=VLN_TEMPORAL_DATASET_ROOT,
    num_frames=17,
    video_size=(272, 480),
    target_fps=10.0,
    condition_frame_positions=(0, 1, 2, 3, 4),
    subtask_start_position=5,
    subtask_anchor_position=5,
    waypoint_reference_position=4,
    waypoint_positions=(8, 12, 16),
    post_subtask_tail_frames=4,
    straight_window_stride=8,
    source_audit_manifest=VLN_TEMPORAL_AUDIT_MANIFEST,
    source_audit_policy="report_only",
    temporal_mode=True,
    strict_target_fps=True,
    structured_prompt=True,
    horizontal_flip_prob=0.5,
    onset_jitter=1,
)

vln_intern_temporal_waypoint_dataloader_train_17f_272_480 = L(get_vln_waypoint_generic_dataloader)(
    dataset=vln_intern_temporal_waypoint_dataset_17f_272_480,
    sampler=L(get_vln_action_source_balanced_sampler)(dataset=vln_intern_temporal_waypoint_dataset_17f_272_480),
    batch_size=16,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

vln_intern_temporal_waypoint_dataset_val_17f_272_480 = copy.deepcopy(vln_intern_temporal_waypoint_dataset_17f_272_480)
vln_intern_temporal_waypoint_dataset_val_17f_272_480.dataset_dir = VLN_TEMPORAL_VAL_DATASET_ROOT
vln_intern_temporal_waypoint_dataset_val_17f_272_480.source_audit_manifest = VLN_TEMPORAL_VAL_AUDIT_MANIFEST
vln_intern_temporal_waypoint_dataset_val_17f_272_480.horizontal_flip_prob = 0.0
vln_intern_temporal_waypoint_dataset_val_17f_272_480.onset_jitter = 0

vln_intern_temporal_waypoint_dataloader_val_17f_272_480 = L(get_vln_waypoint_generic_dataloader)(
    dataset=vln_intern_temporal_waypoint_dataset_val_17f_272_480,
    sampler=L(get_vln_waypoint_eval_sampler)(dataset=vln_intern_temporal_waypoint_dataset_val_17f_272_480),
    batch_size=4,
    drop_last=False,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

predict2_video2world_training_2b_vln_intern_reason1_traj_waypoint_29f_10fps_272_480 = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /model": "fsdp_rectified_flow_vln_waypoint"},
        {"override /net": "cosmos_v1_2B_waypoint"},
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=vln_intern_waypoint_dataloader_train_29f_272_480,
    checkpoint=dict(
        save_iter=2000,
        load_path=DEFAULT_CHECKPOINT_2B.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_vln_intern_reason1_traj_wp_29f_10fps_272x480_b2",
        wandb_mode="disabled",
    ),
    optimizer=dict(
        lr=2e-5,
        weight_decay=0.0001,
    ),
    scheduler=dict(
        f_max=[1.0],
        f_min=[0.01],
        warm_up_steps=[500],
        cycle_lengths=[20_000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=20_000,
        straggler_detection=dict(
            enabled=False,
            max_diff=1.5,
        ),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    model=dict(
        config=dict(
            state_t=8,
            fsdp_shard_size=4,
            min_num_conditional_frames=2,
            max_num_conditional_frames=2,
            conditional_frames_probs=None,
            text_encoder_class="reason1p1_7B",
            text_encoder_config=None,
            trajectory_reason_encoder_config=dict(
                compute_online=True,
                model_config=dict(
                    model_config=dict(
                        use_fsdp2=False,
                    ),
                ),
            ),
            reason_hidden_dim=3584,
            dit_context_dim=1024,
            num_waypoints=4,
            waypoint_dim=3,
            condition_frame_positions=(0, 1, 2, 3, 4),
            waypoint_loss_weight=1.0,
            waypoint_swap_loss_weight=0.05,
            waypoint_swap_margin=0.02,
            waypoint_swap_min_distance=0.5,
            waypoint_only_dropout_prob=0.1,
            net=dict(
                use_crossattn_projection=True,
                crossattn_proj_in_channels=100352,
                crossattn_emb_channels=1024,
                num_waypoints=4,
                waypoint_dim=3,
            ),
            conditioner=dict(
                text=dict(
                    dropout_rate=0.0,
                    use_empty_string=False,
                ),
                use_video_condition=dict(
                    dropout_rate=0.0,
                ),
            ),
        ),
    ),
)


_temporal_waypoint_model_config = dict(
    state_t=5,
    fsdp_shard_size=4,
    min_num_conditional_frames=2,
    max_num_conditional_frames=2,
    conditional_frames_probs=None,
    text_encoder_class="reason1p1_7B",
    text_encoder_config=None,
    trajectory_reason_encoder_config=dict(
        compute_online=True,
        model_config=dict(
            model_config=dict(
                use_fsdp2=False,
            ),
        ),
    ),
    reason_hidden_dim=3584,
    dit_context_dim=1024,
    num_waypoints=3,
    waypoint_dim=3,
    condition_frame_positions=(0, 1, 2, 3, 4),
    waypoint_loss_weight=1.0,
    waypoint_swap_loss_weight=0.0,
    waypoint_only_dropout_prob=0.0,
    dit_context_mode="full",
    waypoint_condition_dropout_prob=0.1,
    mirrored_waypoint_loss_weight=0.05,
    mirrored_waypoint_margin=0.02,
    language_instruction_loss_weight=0.05,
    language_instruction_margin=0.05,
    language_instruction_t_min=0.2,
    language_instruction_t_max=0.8,
    backbone_lr=3e-6,
    conditioning_lr=3e-5,
    net=dict(
        use_crossattn_projection=True,
        crossattn_proj_in_channels=100352,
        crossattn_emb_channels=1024,
        num_waypoints=3,
        waypoint_dim=3,
        num_condition_latents=2,
    ),
    conditioner=dict(
        text=dict(
            dropout_rate=0.0,
            use_empty_string=False,
        ),
        use_video_condition=dict(
            dropout_rate=0.0,
        ),
    ),
)

predict2_video2world_training_2b_vln_temporal_waypoint_17f_warmup = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /model": "fsdp_rectified_flow_vln_temporal_waypoint"},
        {"override /net": "cosmos_v1_2B_temporal_waypoint"},
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=vln_intern_temporal_waypoint_dataloader_train_17f_272_480,
    dataloader_val=vln_intern_temporal_waypoint_dataloader_val_17f_272_480,
    checkpoint=dict(
        save_iter=500,
        load_path=DEFAULT_CHECKPOINT_2B.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_vln_temporal_wp_17f_fullctx_warmup",
        wandb_mode="disabled",
    ),
    optimizer=dict(
        lr=3e-5,
        weight_decay=0.0001,
    ),
    scheduler=dict(
        # Scheduler lists describe sequential cycles, not optimizer groups.
        # LambdaLR applies this one multiplier to every parameter group while
        # preserving each group's explicit base learning rate.
        f_start=[1.0e-6],
        f_max=[1.0],
        f_min=[0.01],
        warm_up_steps=[200],
        cycle_lengths=[2_000],
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=2_000,
        run_validation=True,
        validation_iter=500,
        max_val_iter=32,
        straggler_detection=dict(enabled=False, max_diff=1.5),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(
                every_n=500,
                guidance=[0.0, 1.0, 2.0],
                fps=10,
                save_s3=False,
            ),
            every_n_sample_ema=dict(every_n=0, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
    model=dict(
        config={
            **copy.deepcopy(_temporal_waypoint_model_config),
            "freeze_dit_backbone": True,
            "teacher_forcing_gt_steps": 500,
            "teacher_forcing_decay_steps": 1_500,
            "detach_predicted_waypoints_for_denoise": True,
        },
    ),
)

TEMPORAL_WAYPOINT_WARMUP_CHECKPOINT = (
    "runs/cosmos_predict_v2p5/video2world/2b_vln_temporal_wp_17f_fullctx_warmup/checkpoints/iter_000002000"
)

predict2_video2world_training_2b_vln_temporal_waypoint_17f_full_finetune = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /model": "fsdp_rectified_flow_vln_temporal_waypoint"},
        {"override /net": "cosmos_v1_2B_temporal_waypoint"},
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=vln_intern_temporal_waypoint_dataloader_train_17f_272_480,
    dataloader_val=vln_intern_temporal_waypoint_dataloader_val_17f_272_480,
    checkpoint=dict(
        save_iter=1_000,
        load_path=TEMPORAL_WAYPOINT_WARMUP_CHECKPOINT,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_vln_temporal_wp_17f_full_finetune",
        wandb_mode="disabled",
    ),
    optimizer=dict(
        lr=3e-6,
        weight_decay=0.0001,
    ),
    scheduler=dict(
        # One schedule is shared by all FSDP parameter groups; the groups keep
        # their distinct 3e-6 backbone and 3e-5 conditioning base LRs.
        # Four-rank training has about 12.1k steps per effective epoch, so
        # eight epochs require about 96.8k valid scheduler steps.
        f_start=[1.0e-6],
        f_max=[1.0],
        f_min=[0.1],
        warm_up_steps=[200],
        cycle_lengths=[100_000],
    ),
    trainer=dict(
        max_iter=999_999_999,
        max_epochs=5.0,
        logging_iter=100,
        run_validation=True,
        validation_epoch_interval=0.5,
        max_val_iter=32,
        straggler_detection=dict(enabled=False, max_diff=1.5),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(
                every_n=1_000,
                guidance=[0.0, 1.0, 2.0],
                fps=10,
                save_s3=False,
            ),
            every_n_sample_ema=dict(every_n=0, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
    model=dict(
        config={
            **copy.deepcopy(_temporal_waypoint_model_config),
            "freeze_dit_backbone": False,
            "teacher_forcing_gt_steps": 0,
            "teacher_forcing_decay_steps": 0,
            "detach_predicted_waypoints_for_denoise": False,
        },
    ),
)

predict2_video2world_training_2b_vln_temporal_waypoint_17f_vision_only_ablation = copy.deepcopy(
    predict2_video2world_training_2b_vln_temporal_waypoint_17f_full_finetune
)
predict2_video2world_training_2b_vln_temporal_waypoint_17f_vision_only_ablation["job"]["name"] = (
    "2b_vln_temporal_wp_17f_vision_only_ablation"
)
predict2_video2world_training_2b_vln_temporal_waypoint_17f_vision_only_ablation["model"]["config"][
    "dit_context_mode"
] = "vision_only"


predict2_video2world_training_2b_vln_cosmos_29f_10fps_272_480 = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=vln_cosmos_dataloader_train_29f_272_480,
    checkpoint=dict(
        save_iter=2000,
        load_path="",
        # load_path="/home/csevolunt/dongzhih/cosmos-predict2.5/runs/cosmos_predict_v2p5/video2world/2b_vln_cosmos_vam_49f_10fps_272x480_h200x4_b24/checkpoints/iter_000020000",
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_vln_cosmos_vam_29f_10fps_272x480_h200x4_gb112",
        wandb_mode="disabled",
    ),
    optimizer=dict(
        lr=1.5e-4,
        weight_decay=0.0001,
    ),
    scheduler=dict(
        f_max=[1.0],
        f_min=[0.01],
        warm_up_steps=[1000],
        cycle_lengths=[50_000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=50_000,
        straggler_detection=dict(
            enabled=False,
            max_diff=1.5,
        ),
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=100,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=0,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=0,
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
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    model=dict(
        config=dict(
            state_t=8,
            fsdp_shard_size=4,
            min_num_conditional_frames=2,
            max_num_conditional_frames=2,
            conditional_frames_probs=None,
        ),
    ),
)


cs = ConfigStore.instance()

for _item in [
    predict2_video2world_training_2b_vln_cosmos_29f_10fps_272_480,
    predict2_video2world_training_2b_vln_intern_reason1_traj_waypoint_29f_10fps_272_480,
    predict2_video2world_training_2b_vln_temporal_waypoint_17f_warmup,
    predict2_video2world_training_2b_vln_temporal_waypoint_17f_full_finetune,
    predict2_video2world_training_2b_vln_temporal_waypoint_17f_vision_only_ablation,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
