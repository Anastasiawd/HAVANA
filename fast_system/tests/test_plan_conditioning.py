from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fast_system.dataset.intern_plan_dataset as intern_plan_dataset
import fast_system.trainer.train_fast_nav as train_fast_nav
from fast_system.dataset.intern_plan_dataset import (
    InternDataOnlinePlanDataset,
    _local_xy,
    _local_xyt,
    build_native_intern_dataset,
    collate_online_plan_batch,
    collate_plan_batch,
    stratified_scene_split,
)
from fast_system.model.adapter import RouteQFormer
from fast_system.model.online_plan import FrozenReason1Planner
from fast_system.model.plan_encoder import PLAN_SCHEMA_VERSION, PlanEncoder
from fast_system.model.nav_model import (
    actions_to_deltas,
    normalize_deltas,
    normalized_deltas_to_actions,
)
from fast_system.scripts.run_pipeline import (
    ActionAdapterConfig,
    TrajectoryActionAdapter,
    _coerce_depth_m,
    validate_plan_bundle,
)
from fast_system.trainer.train_fast_nav import (
    resolve_stage2_init_checkpoint,
    select_stage2_pretrained_state,
    wandb_batch_payload,
)
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


def _plan_batch(batch: int = 2, length: int = 9):
    context = torch.randn(batch, length, 1024)
    mask = torch.ones(batch, length, dtype=torch.bool)
    mask[-1, -2:] = False
    trajectories = torch.randn(batch, 3, 3584, requires_grad=True)
    return context, mask, trajectories


def test_plan_encoder_shape_mask_and_gradients():
    model = PlanEncoder(dropout=0.0)
    context, mask, trajectories = _plan_batch()
    offsets = torch.arange(1, 13).repeat(2, 1)
    output = model(context, mask, trajectories, offsets)
    assert output.shape == (2, 12, 19, 384)
    output.square().mean().backward()
    assert trajectories.grad is not None and trajectories.grad.abs().sum() > 0
    assert not torch.equal(output[:, 0, -3:], output[:, -1, -3:])


def test_plan_encoder_rejects_empty_context():
    model = PlanEncoder(dropout=0.0)
    context, mask, trajectories = _plan_batch(batch=1)
    with pytest.raises(ValueError, match="at least one valid token"):
        model(context, torch.zeros_like(mask), trajectories, torch.ones(1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 attention is validated on CUDA")
def test_plan_encoder_fp16():
    model = PlanEncoder(dropout=0.0).cuda().half()
    context, mask, trajectories = _plan_batch(batch=1)
    output = model(context.cuda().half(), mask.cuda(), trajectories.cuda().half(), torch.ones(1, device="cuda"))
    assert output.shape == (1, 19, 384)
    assert output.dtype == torch.float16


def test_route_qformer_legacy_and_external_queries():
    model = RouteQFormer(input_dim=8, hidden_dim=12, num_queries=4, num_layers=1, num_heads=3, dropout=0)
    tokens = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert model(tokens, mask).shape == (2, 4, 12)
    assert model(tokens, mask, query_tokens=torch.randn(2, 7, 12)).shape == (2, 7, 12)


def test_variable_context_collate():
    def item(length: int):
        return {
            "reason_context": torch.randn(length, 1024).half(),
            "context_mask": torch.ones(length, dtype=torch.bool),
            "traj_latents": torch.randn(3, 3584).half(),
            "obs": torch.randn(12, 4, 3, 16, 16),
            "depth": torch.rand(12, 16, 16),
            "action": torch.randn(12, 6, 3),
            "action_mask": torch.ones(12, 6),
            "distance": torch.randn(12),
            "async_offsets": torch.arange(1, 13),
            "metadata": {"schema": PLAN_SCHEMA_VERSION},
        }

    batch = collate_plan_batch([item(5), item(8)])
    assert batch["reason_context"].shape == (2, 8, 1024)
    assert not batch["context_mask"][0, 5:].any()
    assert batch["obs"].shape[:2] == (2, 12)


def test_pose_reroot_uses_x_forward_y_left():
    reference = np.eye(4, dtype=np.float32)
    target = np.eye(4, dtype=np.float32)
    target[0, 3] = 2.0
    target[1, 3] = 1.0
    np.testing.assert_allclose(_local_xy([target], reference), [[2.0, 1.0]], atol=1e-6)
    target[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    np.testing.assert_allclose(_local_xyt([target], reference), [[2.0, 1.0, np.pi / 2]], atol=1e-6)


def test_stage2_pretrained_selection_excludes_legacy_goal_and_heads():
    current = {
        "vision_encoder.obs_encoder.block.weight": torch.zeros(2, 2),
        "vision_encoder.sa_encoder.layer.weight": torch.zeros(2, 2),
        "vision_encoder.z_plan_projector.1.weight": torch.zeros(2, 3),
        "noise_pred_net.block.weight": torch.zeros(2, 2),
    }
    legacy = {
        "module.vision_encoder.obs_encoder.block.weight": torch.ones(2, 2),
        "module.vision_encoder.sa_encoder.layer.weight": torch.ones(2, 2),
        "module.vision_encoder.z_goal_projector.1.weight": torch.ones(2, 2),
        "module.noise_pred_net.block.weight": torch.ones(2, 2),
    }
    selected, counts = select_stage2_pretrained_state(current, legacy, mode="backbone")
    assert set(selected) == {
        "vision_encoder.obs_encoder.block.weight",
        "vision_encoder.sa_encoder.layer.weight",
    }
    assert counts["loaded"] == 2
    assert counts["excluded"] == 1
    assert counts["not_allowed"] == 1

    selected_full, _ = select_stage2_pretrained_state(current, legacy, mode="full")
    assert "noise_pred_net.block.weight" in selected_full
    assert not any("goal_projector" in key for key in selected_full)


def test_stage2_checkpoint_directory_prefers_best(tmp_path):
    torch.save({"model": {}}, tmp_path / "last.pt")
    torch.save({"model": {}}, tmp_path / "best.pt")
    assert resolve_stage2_init_checkpoint(tmp_path).name == "best.pt"


def test_online_plan_collate_keeps_reason_inputs_without_latents():
    def item(instruction: str):
        return {
            "reason_video": torch.zeros(3, 17, 272, 480, dtype=torch.uint8),
            "instruction": instruction,
            "obs": torch.randn(12, 4, 3, 16, 16),
            "depth": torch.rand(12, 16, 16),
            "action": torch.randn(12, 6, 3),
            "action_mask": torch.ones(12, 6),
            "distance": torch.randn(12),
            "async_offsets": torch.arange(1, 13),
            "metadata": {"instruction": instruction},
        }

    batch = collate_online_plan_batch([item("left"), item("right")])
    assert batch["reason_video"].shape == (2, 3, 17, 272, 480)
    assert batch["instruction"] == ["left", "right"]
    assert "reason_context" not in batch
    assert batch["obs"].shape[:2] == (2, 12)


def test_online_dataset_initializes_observation_context_stride(monkeypatch):
    class EmptyNativeDataset:
        episodes = ()
        samples = []
        sample_anchor_frames = []

        def __len__(self):
            return 0

    monkeypatch.setattr(
        "fast_system.dataset.intern_plan_dataset.build_native_intern_dataset",
        lambda _cfg, _split: EmptyNativeDataset(),
    )

    dataset = InternDataOnlinePlanDataset({"obs_context_dt_frames": 3}, "train")

    assert dataset.obs_context_dt_frames == 3


def test_online_dataset_keeps_episode_path_when_native_ids_collide(tmp_path):
    roots = [
        tmp_path / "matterport3d_zed" / "same-scene",
        tmp_path / "matterport_d435i" / "same-scene",
    ]
    episodes = [
        SimpleNamespace(
            root=root,
            video_path=root / "videos" / "episode_000013.mp4",
            data_path=root / "data" / "episode_000013.parquet",
            source_fps=30.0,
            episode_index=13,
        )
        for root in roots
    ]

    class CollidingNativeDataset:
        pose_key = "action"
        samples = [(episode, object()) for episode in episodes]

        def _load_sample(self, episode, _subtask, _index):
            return {
                "video": torch.zeros(3, 17, 4, 4, dtype=torch.uint8),
                "ai_caption": "raw instruction",
                "frame_indices": torch.arange(17),
                # These identifiers intentionally collide across both sources.
                "source_id": "same-scene",
                "scene_id": "same-scene",
                "episode_index": episode.episode_index,
                "subtask_index": 0,
                "subtask_anchor_frame": 5,
            }

    dataset = object.__new__(InternDataOnlinePlanDataset)
    dataset.native_dataset = CollidingNativeDataset()
    dataset.source_by_scene = {
        str(root.resolve()): root.parent.name
        for root in roots
    }
    dataset._build_fast_fields = lambda record: {"selected_video_path": record["video_path"]}

    first = dataset[0]
    second = dataset[1]

    assert first["selected_video_path"] == str(episodes[0].video_path)
    assert second["selected_video_path"] == str(episodes[1].video_path)
    assert first["metadata"]["source_id"] == "matterport3d_zed"
    assert second["metadata"]["source_id"] == "matterport_d435i"


def test_fastnav_uses_every_native_frame_at_30_fps(monkeypatch, tmp_path):
    class FakeBatch:
        def __init__(self, frames):
            self.frames = frames

        def asnumpy(self):
            return self.frames

    class FakeReader:
        def __init__(self):
            self.requested = None

        def get_batch(self, indices):
            self.requested = list(indices)
            frames = np.zeros((len(indices), 4, 4, 3), dtype=np.uint8)
            return FakeBatch(frames)

    poses = []
    for frame in range(100):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(frame)
        poses.append(pose)

    class FakeIloc:
        def __getitem__(self, frame):
            return {"action": poses[int(frame)]}

    class FakeDataFrame:
        iloc = FakeIloc()

    reader = FakeReader()
    dataset = object.__new__(intern_plan_dataset.InternDataFastDataset)
    dataset.size = (4, 4)
    dataset.context_size = 1
    dataset.action_horizon = 3
    dataset.async_offsets = (1, 2)
    dataset.obs_context_dt_frames = 1
    dataset.min_depth_m = 0.1
    dataset.max_depth_m = 5.0
    dataset._load_video_and_poses = lambda _record: (
        reader,
        FakeDataFrame(),
        "action",
        100,
    )
    depth_requests = []

    def fake_depth(path, _minimum, _maximum):
        depth_requests.append(path.name)
        return np.ones((4, 4), dtype=np.float32)

    monkeypatch.setattr(intern_plan_dataset, "_load_depth_m", fake_depth)
    result = dataset._build_fast_fields(
        {
            "video_path": str(tmp_path / "episode_000001.mp4"),
            "reference_frame": 10,
            # This must not create a stride of three for FastNav.
            "source_fps": 30.0,
        }
    )

    assert reader.requested == [10, 11, 12]
    assert depth_requests == ["episode_000001_011.png", "episode_000001_012.png"]
    np.testing.assert_allclose(
        result["action"][..., 0],
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
    )
    assert result["async_offsets"].tolist() == [1, 2]


def test_fastnav_full_horizon_filter_is_in_native_frames(monkeypatch):
    dataset = object.__new__(intern_plan_dataset.InternDataFastDataset)
    dataset.async_offsets = tuple(range(1, 13))
    dataset.action_horizon = 16
    monkeypatch.setattr(intern_plan_dataset, "_parquet_num_rows", lambda _path: 39)
    assert dataset._record_has_full_horizon(
        {
            "reference_frame": 10,
            "source_fps": 30.0,
            "data_path": "unused.parquet",
        }
    )


def test_online_native_dataset_uses_original_instruction_without_structured_prefix(monkeypatch, tmp_path):
    captured = {}

    class CapturingNativeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    scene = tmp_path / "scene"
    audit_manifest = tmp_path / "audit.json"
    audit_manifest.write_text("{}")
    monkeypatch.setattr(
        "fast_system.dataset.intern_plan_dataset.discover_scene_roots",
        lambda _roots: [("source", scene)],
    )
    monkeypatch.setattr(
        "fast_system.dataset.intern_plan_dataset.stratified_scene_split",
        lambda scenes, _seed, _fraction: ([scenes[0][1]], [scenes[0][1]]),
    )
    monkeypatch.setattr(
        "fast_system.dataset.intern_plan_dataset._native_dataset_class",
        lambda: CapturingNativeDataset,
    )

    build_native_intern_dataset(
        {
            "dataset_roots": [tmp_path],
            "source_audit_manifest": audit_manifest,
            "prompt_fields": ["sub_instruction"],
            "structured_prompt": False,
        },
        "train",
    )

    assert captured["prompt_fields"] == ("sub_instruction",)
    assert captured["structured_prompt"] is False


def test_online_plan_validation_requires_detached_complete_batch():
    plan = {
        "reason_context": torch.randn(2, 7, 1024),
        "context_mask": torch.ones(2, 7, dtype=torch.bool),
        "traj_latents": torch.randn(2, 3, 3584),
    }
    FrozenReason1Planner._validate(plan, batch_size=2)
    plan["traj_latents"] = plan["traj_latents"].requires_grad_()
    with pytest.raises(RuntimeError, match="detached"):
        FrozenReason1Planner._validate(plan, batch_size=2)
    with torch.inference_mode():
        plan["traj_latents"] = torch.randn(2, 3, 3584)
    with pytest.raises(RuntimeError, match="inference tensors"):
        FrozenReason1Planner._validate(plan, batch_size=2)


def test_online_scene_split_is_stratified_and_disjoint(tmp_path):
    scenes = [
        (source, tmp_path / f"{source}_{index}")
        for source in ("a", "b")
        for index in range(10)
    ]
    train, val = stratified_scene_split(scenes, seed=5, val_fraction=0.1)
    assert len(train) == 18
    assert len(val) == 2
    assert set(train).isdisjoint(val)
    assert {path.name.split("_", 1)[0] for path in val} == {"a", "b"}


def test_wandb_payload_contains_batch_losses_and_learning_rate():
    payload = wandb_batch_payload(
        "train",
        epoch=3,
        batch_index=7,
        metrics={
            "loss": 1.25,
            "flow_loss": 1.0,
            "distance_loss": 0.25,
            "reason1_latency": 0.8,
        },
        lr=1e-4,
    )
    assert payload["epoch"] == 3
    assert payload["train/batch"] == 7
    assert payload["train/loss"] == 1.25
    assert payload["train/flow_loss"] == 1.0
    assert payload["train/distance_loss"] == 0.25
    assert payload["train/reason1_latency"] == 0.8
    assert payload["train/learning_rate"] == 1e-4


def test_constant_scheduler_warms_up_then_stays_constant():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = train_fast_nav.build_constant_with_warmup_scheduler(optimizer, warmup_steps=5)
    update_lrs = []
    for _ in range(7):
        update_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    np.testing.assert_allclose(update_lrs, [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])


def test_trainable_ema_checkpoint_exposes_average_and_raw_resume_weights(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    plan_encoder = torch.nn.Linear(1, 1, bias=False)
    frozen = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)
    model.register_parameter("frozen", frozen)
    with torch.no_grad():
        model.weight.fill_(1.0)
        plan_encoder.weight.fill_(1.0)
    ema = train_fast_nav.TrainableEMA(
        {"model": model, "plan_encoder": plan_encoder},
        inv_gamma=1.0,
        power=1.0,
        max_value=0.5,
    )
    for value in (2.0, 4.0, 6.0):
        with torch.no_grad():
            model.weight.fill_(value)
            plan_encoder.weight.fill_(value)
        ema.update()

    # The third update uses decay=0.5, so EMA is 5 while raw is 6.
    assert torch.equal(ema.averaged_state_dict("model")["weight"], torch.tensor([[5.0]]))
    assert set(ema.shadow) == {"model.weight", "plan_encoder.weight"}

    optimizer = torch.optim.AdamW((*model.parameters(), *plan_encoder.parameters()), lr=1e-4)
    scheduler = train_fast_nav.build_constant_with_warmup_scheduler(optimizer, warmup_steps=5)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "step.pt"
    train_fast_nav.save_checkpoint(
        path,
        model,
        plan_encoder,
        optimizer,
        scaler,
        epoch=2,
        config={},
        scheduler=scheduler,
        ema=ema,
        global_step=3,
        batch_index=9,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(checkpoint["model"]["weight"], torch.tensor([[5.0]]))
    assert torch.equal(checkpoint["raw_model"]["weight"], torch.tensor([[6.0]]))
    assert torch.equal(checkpoint["model"]["frozen"], torch.tensor([7.0]))
    assert checkpoint["global_step"] == 3
    assert checkpoint["ema"]["optimization_step"] == 3
    assert checkpoint["scheduler"] is not None

    raw_before = model.weight.detach().clone()
    with ema.average_parameters():
        assert torch.equal(model.weight, torch.tensor([[5.0]]))
    assert torch.equal(model.weight, raw_before)


def test_step_checkpoint_updates_last_without_duplicate_storage(tmp_path):
    source = tmp_path / "step_000005000.pt"
    source.write_bytes(b"checkpoint")
    last = tmp_path / "last.pt"
    train_fast_nav.update_last_checkpoint(source, last)
    assert last.read_bytes() == b"checkpoint"
    assert source.stat().st_ino == last.stat().st_ino


def test_run_epoch_updates_progress_and_logs_wandb(monkeypatch, capsys):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.vision_encoder = torch.nn.Identity()

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, payload):
            self.logged.append(payload)

    def fake_compute_loss(model, plan_encoder, *_args, **_kwargs):
        loss = model.weight.square() + plan_encoder.weight.square().mean()
        value = float(loss.detach())
        return loss, {"loss": value, "flow_loss": value - 0.1, "distance_loss": 0.1}

    monkeypatch.setattr(train_fast_nav, "compute_loss", fake_compute_loss)
    model = DummyModel()
    plan_encoder = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD((*model.parameters(), *plan_encoder.parameters()), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    training_state = {"global_step": 0, "last_saved_step": 0}
    fake_run = FakeRun()
    config = {
        "training": {"plan_dropout_prob": 0.5, "distance_loss_weight": 1e-4, "grad_clip": 1.0},
        "model": {"action_stats": {"min": [-0.15, -0.025, -0.25], "max": [0.15, 0.075, 0.25]}},
        "wandb": {"log_every": 1},
    }
    metrics = train_fast_nav.run_epoch(
        model,
        plan_encoder,
        [{"obs": torch.zeros(1, 4, 1)}],
        optimizer,
        scaler,
        torch.device("cpu"),
        False,
        config,
        True,
        epoch=1,
        total_epochs=1,
        wandb_run=fake_run,
        training_state=training_state,
    )
    assert metrics["loss"] > 0
    assert fake_run.logged[0]["train/loss"] > 0
    assert fake_run.logged[0]["train/learning_rate"] == 1e-3
    assert training_state["global_step"] == 1
    assert "train epoch 1/1" in capsys.readouterr().err


def test_xyt_delta_normalization_round_trip_wraps_yaw():
    actions = torch.tensor(
        [[[0.1, 0.2, 3.10], [0.2, 0.4, -3.10], [0.4, 0.5, -2.90]]],
        dtype=torch.float32,
    )
    stats = {"min": [-1.0, -1.0, -np.pi], "max": [1.0, 1.0, np.pi], "clip": False}
    normalized = normalize_deltas(actions_to_deltas(actions), stats)
    restored = normalized_deltas_to_actions(normalized, stats)
    np.testing.assert_allclose(restored[..., :2], actions[..., :2], atol=1e-6)
    yaw_error = torch.remainder(restored[..., 2] - actions[..., 2] + torch.pi, 2 * torch.pi) - torch.pi
    assert yaw_error.abs().max() < 1e-5


def test_depth_contract_scales_uint16_and_masks_invalid_values():
    raw = np.array([[0, 1000, 10000, 65535]], dtype=np.uint16)
    depth = _coerce_depth_m(raw, min_depth_m=0.1, max_depth_m=5.0)
    np.testing.assert_allclose(depth, [[0.0, 0.1, 1.0, 0.0]], atol=1e-6)
    with pytest.raises(ValueError, match="Expected depth"):
        _coerce_depth_m(np.zeros((2, 2, 2), dtype=np.float32))


def test_plan_bundle_v2_has_no_explicit_waypoint_field():
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "reason_context": torch.randn(7, 1024),
        "context_mask": torch.ones(7, dtype=torch.bool),
        "traj_latents": torch.randn(3, 3584),
    }
    tensors = validate_plan_bundle(plan)
    assert set(tensors) == {"reason_context", "context_mask", "traj_latents"}
    assert tensors["reason_context"].dtype == torch.float16


def test_action_adapter_uses_predicted_yaw_instead_of_xy_bearing():
    adapter = TrajectoryActionAdapter(
        ActionAdapterConfig(turn_threshold_rad=0.1, lookahead_index=0),
        ["stop", "move_forward", "turn_left", "turn_right"],
    )
    trajectory = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
    action, _ = adapter.convert(trajectory, step_id=0)
    assert action == "turn_left"


def test_unet_cross_attention_runs_at_all_action_resolutions():
    model = ConditionalUnet1D(
        input_dim=3,
        global_cond_dim=16,
        down_dims=(16, 32, 64),
        n_groups=8,
        context_dim=24,
        cross_attn_heads=4,
        cross_attn_on_all_resolutions=True,
    )
    output = model(
        torch.randn(2, 16, 3),
        torch.rand(2),
        global_cond=torch.randn(2, 16),
        context=torch.randn(2, 9, 24),
        context_mask=torch.tensor([[False] * 9, [True] * 9]),
    )
    assert output.shape == (2, 16, 3)
    assert torch.isfinite(output).all()
    assert len(model.down_cross_attn_layers) == 3
    assert len(model.up_cross_attn_layers) == 2
