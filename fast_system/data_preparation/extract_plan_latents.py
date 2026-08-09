#!/usr/bin/env python3
"""Deterministically extract Reason1 plan bundles from native InternData samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


HAVANA_ROOT = Path(__file__).resolve().parents[2]
SLOW_ROOT = HAVANA_ROOT / "slow_system"
for path in (HAVANA_ROOT, SLOW_ROOT, SLOW_ROOT / "packages" / "cosmos-oss"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_system.model.plan_encoder import PLAN_SCHEMA_VERSION  # noqa: E402
from slow_system.scripts.slow_server import CosmosSlowEngine, SlowServerConfig  # noqa: E402
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_vln_waypoint import (  # noqa: E402
    InternDataVLNWaypointDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Existing iter_000060000/model.pt; it is never modified.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="predict2_video2world_training_2b_vln_temporal_waypoint_17f_full_finetune")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def discover_scene_roots(roots: Sequence[Path]) -> list[tuple[str, Path]]:
    """Discover LeRobot scene roots before constructing the Cosmos dataset."""
    found: list[tuple[str, Path]] = []
    for dataset_root in roots:
        if (dataset_root / "meta" / "episodes.jsonl").exists():
            found.append((dataset_root.name, dataset_root))
            continue
        info_paths = []
        for pattern in ("*/meta/info.json", "*/*/meta/info.json", "*/*/*/meta/info.json"):
            info_paths.extend(sorted(dataset_root.glob(pattern)))
        for info in info_paths:
            scene = info.parent.parent
            if (scene / "meta" / "episodes.jsonl").exists():
                relative = scene.relative_to(dataset_root)
                source = relative.parts[0] if relative.parts else dataset_root.name
                found.append((source, scene))
    unique = {(source, str(path.resolve())): (source, path.resolve()) for source, path in found}
    return list(unique.values())


def stratified_scene_split(
    scenes: Sequence[tuple[str, Path]], seed: int, val_fraction: float
) -> tuple[list[Path], list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for source, path in scenes:
        groups[source].append(path)
    rng = random.Random(seed)
    train: list[Path] = []
    val: list[Path] = []
    for source in sorted(groups):
        values = sorted(groups[source])
        rng.shuffle(values)
        count = int(round(len(values) * val_fraction))
        if len(values) > 1:
            count = min(max(count, 1), len(values) - 1)
        val.extend(values[:count])
        train.extend(values[count:])
    if set(train) & set(val):
        raise AssertionError("Scene split overlap detected.")
    return sorted(train), sorted(val)


def build_dataset(args: argparse.Namespace) -> InternDataVLNWaypointDataset:
    roots = [Path(value).expanduser().resolve() for value in args.dataset_root]
    scenes = discover_scene_roots(roots)
    train_scenes, val_scenes = stratified_scene_split(scenes, args.seed, args.val_fraction)
    selected = roots if args.split == "all" else (train_scenes if args.split == "train" else val_scenes)
    if args.scene:
        filters = set(args.scene)
        split_members = None if args.split == "all" else set(selected)
        selected = [
            path
            for _, path in scenes
            if (split_members is None or path in split_members)
            and (path.name in filters or str(path) in filters)
        ]
    if not selected:
        raise ValueError(f"No scene roots selected for split={args.split!r} filters={args.scene}.")
    return InternDataVLNWaypointDataset(
        dataset_dir=[str(path) for path in selected],
        num_frames=17,
        video_size=(272, 480),
        target_fps=10.0,
        waypoint_positions=(8, 12, 16),
        subtask_anchor_position=5,
        temporal_mode=True,
        condition_frame_positions=(0, 1, 2, 3, 4),
        subtask_start_position=5,
        waypoint_reference_position=4,
        post_subtask_tail_frames=4,
        straight_window_stride=8,
        source_audit_manifest=str(args.audit_manifest.expanduser().resolve()),
        source_audit_policy="report_only",
        prompt_fields=("sub_instruction",),
        structured_prompt=False,
        horizontal_flip_prob=0.0,
        strict_target_fps=True,
        onset_jitter=0,
        seed=args.seed,
    )


def existing_indices(manifest: Path) -> set[int]:
    if not manifest.exists():
        return set()
    with manifest.open("r", encoding="utf-8") as stream:
        return {int(json.loads(line)["dataset_index"]) for line in stream if line.strip()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = build_dataset(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_dir = args.output_dir / args.split / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"{args.split}_manifest.jsonl"
    completed = existing_indices(manifest) if args.resume else set()

    checkpoint = args.checkpoint.expanduser().resolve()
    cfg = SlowServerConfig(
        checkpoint_path=str(checkpoint.parent),
        checkpoint_filename=checkpoint.name,
        experiment=args.experiment,
        resolution="272,480",
        num_conditional_frames=2,
        num_steps=5,
        offload_diffusion_model=False,
        offload_text_encoder=False,
        offload_tokenizer=False,
        aggressive_unload=False,
    )
    engine = CosmosSlowEngine(cfg)
    engine._pipe.model.eval()
    limit = len(dataset) if args.sample_limit is None else min(len(dataset), args.sample_limit)
    mode = "a" if args.resume else "w"
    with manifest.open(mode, encoding="utf-8") as output, torch.inference_mode():
        for index in range(limit):
            if index in completed:
                continue
            plan_path = plan_dir / f"sample_{index:08d}.pt"
            item = dataset[index]
            if plan_path.exists() and args.skip_existing:
                try:
                    stored = torch.load(str(plan_path), map_location="cpu", weights_only=False)
                except TypeError:
                    stored = torch.load(str(plan_path), map_location="cpu")
                result = {**stored, "elapsed": 0.0}
            else:
                condition_video = item["video"][:, :5].permute(1, 2, 3, 0).cpu().numpy()
                result = engine.generate_plan(
                    [frame for frame in condition_video], str(item["ai_caption"]), seed=args.seed
                )
            episode, _ = dataset.samples[index]
            frame_indices = item["frame_indices"].tolist()
            bundle = {
                key: result[key]
                for key in ("reason_context", "context_mask", "traj_latents")
            }
            bundle.update({"schema_version": PLAN_SCHEMA_VERSION, "dataset_index": index})
            if not (plan_path.exists() and args.skip_existing):
                torch.save(bundle, plan_path)
            try:
                reloaded = torch.load(str(plan_path), map_location="cpu", weights_only=False)
            except TypeError:
                reloaded = torch.load(str(plan_path), map_location="cpu")
            for key in ("reason_context", "context_mask", "traj_latents"):
                if not torch.equal(torch.as_tensor(bundle[key]).cpu(), torch.as_tensor(reloaded[key]).cpu()):
                    raise RuntimeError(f"Plan reload consistency check failed for {plan_path}:{key}.")
            record: dict[str, Any] = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "dataset_index": index,
                "plan_path": str(plan_path.relative_to(manifest.parent)),
                "instruction": str(item["ai_caption"]),
                "source_id": item["source_id"],
                "scene_id": item["scene_id"],
                "episode_index": int(item["episode_index"]),
                "subtask_index": int(item["subtask_index"]),
                "canonical_action": item["canonical_action"],
                "frame_indices": frame_indices,
                "reference_frame": int(frame_indices[4]),
                "source_step": int(item["subtask_anchor_frame"]),
                "source_fps": float(episode.source_fps or 30.0),
                "data_path": str(episode.data_path),
                "video_path": str(episode.video_path),
                "pose_key": dataset.pose_key,
                "elapsed": float(result["elapsed"]),
                "plan_bytes": int(plan_path.stat().st_size),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index + 1}/{limit}] {plan_path.name} context={int(result['context_mask'].sum())}", flush=True)


if __name__ == "__main__":
    main()
