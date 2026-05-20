#!/usr/bin/env python3
"""Extract SCAND ROS1 bags into image folders plus aligned trajectories.

Output layout:
  <out-root>/
    <scene-name>/
      images/
        <image_timestamp_ns>.jpg
        ...
      trajectory.npy   # shape [T, 4], columns: x, y, z, yaw
      timestamps.npy   # shape [T], int64 image timestamps in ns
      metadata.json

The script uses the standalone ``rosbags`` Python package and samples image
frames at 4 FPS by default.  Each saved trajectory row is the odometry pose
interpolated at the timestamp of the corresponding saved image.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
import shutil
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag1.reader import Header, Reader, RecordType, decompressors, read_bytes, read_uint32
except ImportError:  # pragma: no cover - handled with a clear runtime message.
    AnyReader = None
    Reader = None
    Header = None
    RecordType = None
    decompressors = None
    read_bytes = None
    read_uint32 = None


IMAGE_TOPICS = (
    "/image_raw/compressed",
    "/camera/rgb/image_raw/compressed",
    "/left/image_color/compressed",
)
ODOM_TOPICS = ("/odom", "/jackal_velocity_controller/odom")


@dataclass(frozen=True)
class BagScan:
    image_topic: str
    odom_topic: str
    image_ts: np.ndarray
    odom_ts: np.ndarray
    odom_xyz_yaw: np.ndarray | None = None


@dataclass
class FastBag:
    reader: Any
    image_topic: str
    odom_topic: str
    image_connections: list[Any]
    odom_connections: list[Any]
    image_ts: np.ndarray
    odom_ts: np.ndarray

    def close(self) -> None:
        self.reader.close()

    def scan(self) -> BagScan:
        return BagScan(
            image_topic=self.image_topic,
            odom_topic=self.odom_topic,
            image_ts=self.image_ts,
            odom_ts=self.odom_ts,
            odom_xyz_yaw=None,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SCAND ROS1 bags to extracted image/trajectory scenes.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing ROS1 .bag files.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("scand_extracted"),
        help="Output dataset root. Defaults to ./scand_extracted.",
    )
    parser.add_argument("--fps", type=float, default=4.0, help="Image sampling rate in Hz.")
    parser.add_argument(
        "--max-frame-dt",
        type=float,
        default=0.25,
        help="Maximum allowed nearest-frame timestamp error in seconds.",
    )
    parser.add_argument(
        "--image-topics",
        nargs="+",
        default=list(IMAGE_TOPICS),
        help="Candidate image topics, checked in order.",
    )
    parser.add_argument(
        "--odom-topics",
        nargs="+",
        default=list(ODOM_TOPICS),
        help="Candidate odometry topics, checked in order.",
    )
    parser.add_argument("--image-ext", choices=("jpg", "png"), default="jpg", help="Saved image format.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when --image-ext=jpg.")
    parser.add_argument("--max-bags", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument(
        "--mode",
        choices=("stream", "index"),
        default="stream",
        help="stream reads each bag once; index uses rosbag indexes for random access.",
    )
    parser.add_argument(
        "--sort",
        choices=("size", "name"),
        default="size",
        help="Bag processing order. Size order keeps huge bags from blocking early progress.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing scene directory.")
    parser.add_argument("--dry-run", action="store_true", help="Scan bags and print what would be extracted.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "scene"


def scene_names_for_bags(bag_paths: Sequence[Path], dataset_root: Path) -> dict[Path, str]:
    stem_counts: dict[str, int] = {}
    for path in bag_paths:
        stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1

    names: dict[Path, str] = {}
    used: set[str] = set()
    for path in bag_paths:
        if stem_counts[path.stem] == 1:
            candidate = sanitize_name(path.stem)
        else:
            rel = path.relative_to(dataset_root).with_suffix("")
            candidate = sanitize_name("__".join(rel.parts))

        name = candidate
        suffix = 1
        while name in used:
            suffix += 1
            name = f"{candidate}_{suffix}"
        used.add(name)
        names[path] = name
    return names


def require_rosbags() -> None:
    if Reader is None:
        raise ImportError(
            "The standalone `rosbags` Python package is required. "
            "Install it with `pip install rosbags`; no external ROS installation is needed."
        )


def choose_topic(available_topics: set[str], candidates: Sequence[str]) -> str | None:
    for topic in candidates:
        if topic in available_topics:
            return topic
    return None


def unique_sorted(values: Sequence[int]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.asarray(values, dtype=np.int64))


def open_fast_bag(bag_path: Path, image_topics: Sequence[str], odom_topics: Sequence[str]) -> FastBag | None:
    reader = Reader(bag_path)
    reader.bio = bag_path.open("rb")
    try:
        magic = reader.bio.readline().decode()
        if not magic.startswith("#ROSBAG V2.0"):
            raise RuntimeError(f"{bag_path} is not a ROS1 V2.0 bag.")

        header = Header.read(reader.bio, RecordType.BAGHEADER)
        index_pos = header.get_uint64("index_pos")
        conn_count = header.get_uint32("conn_count")
        chunk_count = header.get_uint32("chunk_count")
        if index_pos == 0:
            raise RuntimeError(f"{bag_path} is not indexed. Reindex it before extraction.")

        reader.bio.seek(index_pos)
        connections = [reader.read_connection() for _ in range(conn_count)]
        available_topics = {conn.topic for conn in connections}
        image_topic = choose_topic(available_topics, image_topics)
        odom_topic = choose_topic(available_topics, odom_topics)
        if image_topic is None or odom_topic is None:
            logging.warning("%s skipped: missing image or odom topic.", bag_path.name)
            reader.close()
            return None

        image_connections = [conn for conn in connections if conn.topic == image_topic]
        odom_connections = [conn for conn in connections if conn.topic == odom_topic]
        wanted_ids = {conn.id for conn in [*image_connections, *odom_connections]}

        chunk_infos = [reader.read_chunk_info() for _ in range(chunk_count)]
        reader.connections = connections
        reader.chunk_infos = chunk_infos
        indexes: dict[int, list[Any]] = defaultdict(list)
        for chunk_info in chunk_infos:
            if wanted_ids.isdisjoint(chunk_info.connection_counts):
                continue
            reader.bio.seek(chunk_info.pos)
            reader.chunks[chunk_info.pos] = reader.read_chunk()
            chunk_indexes: dict[int, list[Any]] = defaultdict(list)
            for _ in range(len(chunk_info.connection_counts)):
                reader.read_index_data(chunk_info.pos, chunk_indexes)
            for conn_id in wanted_ids:
                indexes[conn_id].extend(chunk_indexes.get(conn_id, []))

        reader.indexes = {conn_id: sorted(entries) for conn_id, entries in indexes.items()}
        image_ts = unique_sorted([entry.time for conn in image_connections for entry in reader.indexes.get(conn.id, [])])
        odom_ts = unique_sorted([entry.time for conn in odom_connections for entry in reader.indexes.get(conn.id, [])])
        if len(image_ts) < 2 or len(odom_ts) < 2:
            logging.warning("%s skipped: too few indexed image/odom messages.", bag_path.name)
            reader.close()
            return None

        return FastBag(
            reader=reader,
            image_topic=image_topic,
            odom_topic=odom_topic,
            image_connections=image_connections,
            odom_connections=odom_connections,
            image_ts=image_ts,
            odom_ts=odom_ts,
        )
    except Exception:
        reader.close()
        raise


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def odom_to_xyz_yaw(msg: Any) -> tuple[float, float, float, float]:
    pose = msg.pose.pose
    pos = pose.position
    ori = pose.orientation
    return (
        float(pos.x),
        float(pos.y),
        float(getattr(pos, "z", 0.0)),
        quat_to_yaw(float(ori.x), float(ori.y), float(ori.z), float(ori.w)),
    )


def connection_indexes(connection: Any) -> Sequence[Any] | None:
    owner = getattr(connection, "owner", None)
    indexes = getattr(owner, "indexes", None)
    if indexes is None or connection.id not in indexes:
        return None
    return indexes[connection.id]


def connection_index_times(connection: Any) -> list[int] | None:
    indexes = connection_indexes(connection)
    if indexes is None:
        return None
    return [int(entry.time) for entry in indexes]


def read_rosbag1_raw(connection: Any, entry: Any) -> bytes:
    owner = connection.owner
    if owner.current_chunk[0] != entry.chunk_pos:
        owner.current_chunk[1].close()
        chunk_header = owner.chunks[entry.chunk_pos]
        owner.bio.seek(chunk_header.datapos)
        rawbytes = chunk_header.decompressor(read_bytes(owner.bio, chunk_header.datasize))
        owner.current_chunk = (entry.chunk_pos, io.BytesIO(rawbytes))

    chunk = owner.current_chunk[1]
    chunk.seek(entry.offset)
    while True:
        header = Header.read(chunk)
        have = header.get_uint8("op")
        if have != RecordType.CONNECTION:
            break
        chunk.seek(read_uint32(chunk), 1)
    if have != RecordType.MSGDATA:
        raise RuntimeError("Expected MSGDATA while reading rosbag1 index entry.")
    return read_bytes(chunk, read_uint32(chunk))


def read_ros_string(raw: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<I", raw, offset)
    offset += 4
    value = raw[offset : offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def read_ros_bytes(raw: bytes, offset: int) -> tuple[bytes, int]:
    (length,) = struct.unpack_from("<I", raw, offset)
    offset += 4
    return raw[offset : offset + length], offset + length


def skip_header(raw: bytes, offset: int = 0) -> int:
    offset += 12
    _, offset = read_ros_string(raw, offset)
    return offset


def parse_compressed_image(raw: bytes) -> tuple[str, bytes]:
    offset = skip_header(raw)
    image_format, offset = read_ros_string(raw, offset)
    data, _ = read_ros_bytes(raw, offset)
    return image_format, data


def parse_raw_image(raw: bytes) -> Any:
    offset = skip_header(raw)
    height, width = struct.unpack_from("<II", raw, offset)
    offset += 8
    encoding, offset = read_ros_string(raw, offset)
    (is_bigendian,) = struct.unpack_from("<B", raw, offset)
    offset += 1
    (step,) = struct.unpack_from("<I", raw, offset)
    offset += 4
    data, _ = read_ros_bytes(raw, offset)
    return SimpleNamespace(
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=is_bigendian,
        step=step,
        data=data,
    )


def parse_odom_xyz_yaw(raw: bytes) -> tuple[float, float, float, float]:
    offset = skip_header(raw)
    _, offset = read_ros_string(raw, offset)
    x, y, z, qx, qy, qz, qw = struct.unpack_from("<7d", raw, offset)
    return float(x), float(y), float(z), quat_to_yaw(float(qx), float(qy), float(qz), float(qw))


def is_compressed_image_type(msgtype: str) -> bool:
    return msgtype in {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"} or msgtype.endswith(
        "/CompressedImage"
    )


def is_raw_image_type(msgtype: str) -> bool:
    return msgtype in {"sensor_msgs/msg/Image", "sensor_msgs/Image"} or msgtype.endswith("/Image")


def is_jpeg_payload(image_format: str, payload: bytes) -> bool:
    fmt = image_format.lower()
    return payload.startswith(b"\xff\xd8") or "jpeg" in fmt or "jpg" in fmt


def scan_bag(bag_path: Path, image_topics: Sequence[str], odom_topics: Sequence[str]) -> BagScan | None:
    with AnyReader([bag_path]) as reader:
        available_topics = {conn.topic for conn in reader.connections}
        image_topic = choose_topic(available_topics, image_topics)
        odom_topic = choose_topic(available_topics, odom_topics)
        if image_topic is None or odom_topic is None:
            logging.warning("%s skipped: missing image or odom topic.", bag_path.name)
            return None

        image_connections = [conn for conn in reader.connections if conn.topic == image_topic]
        odom_connections = [conn for conn in reader.connections if conn.topic == odom_topic]
        image_ts: list[int] = []
        odom_ts: list[int] = []
        odom_xyz_yaw: list[tuple[float, float, float, float]] = []

        all_image_times_indexed = True
        for conn in image_connections:
            indexed_times = connection_index_times(conn)
            if indexed_times is None:
                all_image_times_indexed = False
                image_ts.clear()
                break
            image_ts.extend(indexed_times)
        if not all_image_times_indexed:
            for conn, timestamp_ns, _raw in reader.messages(connections=image_connections):
                image_ts.append(int(timestamp_ns))

        all_odom_indexed = True
        for conn in odom_connections:
            indexes = connection_indexes(conn)
            if indexes is None:
                all_odom_indexed = False
                break
            for entry in indexes:
                odom_ts.append(int(entry.time))
        if not all_odom_indexed:
            odom_ts.clear()
            odom_xyz_yaw.clear()
            for conn, timestamp_ns, raw in reader.messages(connections=odom_connections):
                msg = reader.deserialize(raw, conn.msgtype)
                odom_ts.append(int(timestamp_ns))
                odom_xyz_yaw.append(odom_to_xyz_yaw(msg))

    if len(image_ts) < 2 or len(odom_ts) < 2:
        logging.warning("%s skipped: too few image/odom messages.", bag_path.name)
        return None

    image_array = np.asarray(image_ts, dtype=np.int64)
    image_order = np.argsort(image_array)
    image_sorted = image_array[image_order]
    image_sorted = np.unique(image_sorted)

    odom_array = np.asarray(odom_ts, dtype=np.int64)
    odom_order = np.argsort(odom_array)
    odom_sorted = odom_array[odom_order]
    odom_pose_array = None
    if all_odom_indexed:
        odom_sorted = np.unique(odom_sorted)
    else:
        odom_pose_array = np.asarray(odom_xyz_yaw, dtype=np.float64)[odom_order]
        odom_sorted, unique_odom_idx = np.unique(odom_sorted, return_index=True)
        odom_pose_array = odom_pose_array[unique_odom_idx]

    if len(image_sorted) < 2 or len(odom_sorted) < 2:
        logging.warning("%s skipped: too few unique image/odom timestamps.", bag_path.name)
        return None

    return BagScan(
        image_topic=image_topic,
        odom_topic=odom_topic,
        image_ts=image_sorted,
        odom_ts=odom_sorted,
        odom_xyz_yaw=odom_pose_array,
    )


def nearest_sample_timestamps(scan: BagScan, fps: float, max_frame_dt: float) -> np.ndarray:
    start_ns = int(max(scan.image_ts[0], scan.odom_ts[0]))
    end_ns = int(min(scan.image_ts[-1], scan.odom_ts[-1]))
    if end_ns <= start_ns:
        return np.asarray([], dtype=np.int64)

    period_ns = int(round(1_000_000_000 / fps))
    if period_ns <= 0:
        raise ValueError("--fps must be > 0.")
    target_ts = np.arange(start_ns, end_ns + 1, period_ns, dtype=np.int64)
    max_dt_ns = int(round(max_frame_dt * 1_000_000_000))

    chosen_indices: list[int] = []
    previous_idx: int | None = None
    for target in target_ts:
        idx = int(np.searchsorted(scan.image_ts, target, side="left"))
        candidates = []
        if idx < len(scan.image_ts):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue

        best_idx = min(candidates, key=lambda item: abs(int(scan.image_ts[item]) - int(target)))
        if abs(int(scan.image_ts[best_idx]) - int(target)) > max_dt_ns:
            continue
        if previous_idx is not None and best_idx == previous_idx:
            continue
        chosen_indices.append(best_idx)
        previous_idx = best_idx

    return scan.image_ts[np.asarray(chosen_indices, dtype=np.int64)]


def interpolate_trajectory(scan: BagScan, query_ts: np.ndarray) -> np.ndarray:
    if scan.odom_xyz_yaw is None:
        raise ValueError("BagScan does not contain odometry poses.")
    base_ns = int(scan.odom_ts[0])
    odom_seconds = (scan.odom_ts.astype(np.float64) - base_ns) / 1e9
    query_seconds = (query_ts.astype(np.float64) - base_ns) / 1e9
    poses = scan.odom_xyz_yaw.astype(np.float64).copy()
    poses[:, 3] = np.unwrap(poses[:, 3])

    trajectory = np.empty((len(query_ts), 4), dtype=np.float32)
    for col in range(3):
        trajectory[:, col] = np.interp(query_seconds, odom_seconds, poses[:, col]).astype(np.float32)
    yaw = np.interp(query_seconds, odom_seconds, poses[:, 3])
    trajectory[:, 3] = normalize_angle(yaw).astype(np.float32)
    return trajectory


def indexed_entries_for_connections(connections: Sequence[Any]) -> tuple[np.ndarray, list[Any], list[Any]] | None:
    times: list[int] = []
    entries: list[Any] = []
    entry_connections: list[Any] = []
    for conn in connections:
        indexes = connection_indexes(conn)
        if indexes is None:
            return None
        for entry in indexes:
            times.append(int(entry.time))
            entries.append(entry)
            entry_connections.append(conn)
    if not times:
        return None

    order = np.argsort(np.asarray(times, dtype=np.int64))
    sorted_times: list[int] = []
    sorted_entries: list[Any] = []
    sorted_connections: list[Any] = []
    seen: set[int] = set()
    for idx in order:
        idx = int(idx)
        timestamp_ns = int(times[idx])
        if timestamp_ns in seen:
            continue
        seen.add(timestamp_ns)
        sorted_times.append(timestamp_ns)
        sorted_entries.append(entries[idx])
        sorted_connections.append(entry_connections[idx])
    return np.asarray(sorted_times, dtype=np.int64), sorted_entries, sorted_connections


def interpolate_indexed_trajectory(odom_connections: Sequence[Any], query_ts: np.ndarray) -> np.ndarray:
    indexed = indexed_entries_for_connections(odom_connections)
    if indexed is None:
        raise RuntimeError("Odometry topic is not indexed.")
    odom_ts, entries, connections = indexed

    needed: set[int] = set()
    right_indices = np.searchsorted(odom_ts, query_ts, side="left")
    for idx in right_indices:
        idx = int(idx)
        if idx < len(odom_ts):
            needed.add(idx)
        if idx > 0:
            needed.add(idx - 1)
    if not needed:
        raise RuntimeError("No odometry entries overlap sampled image timestamps.")

    selected_indices = sorted(needed)
    selected_ts = odom_ts[np.asarray(selected_indices, dtype=np.int64)]
    selected_poses: list[tuple[float, float, float, float]] = []
    for idx in selected_indices:
        conn = connections[idx]
        raw = read_rosbag1_raw(conn, entries[idx])
        selected_poses.append(parse_odom_xyz_yaw(raw))

    sparse_scan = BagScan(
        image_topic="",
        odom_topic=odom_connections[0].topic if odom_connections else "",
        image_ts=np.asarray([], dtype=np.int64),
        odom_ts=selected_ts,
        odom_xyz_yaw=np.asarray(selected_poses, dtype=np.float64),
    )
    return interpolate_trajectory(sparse_scan, query_ts)


def compressed_image_to_pil(msg: Any) -> Image.Image:
    data = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(msg.data)
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def raw_image_to_pil(msg: Any) -> Image.Image:
    encoding = str(getattr(msg, "encoding", "rgb8") or "rgb8").lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(getattr(msg, "step", 0) or 0)
    raw_data = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(msg.data)
    data = np.frombuffer(raw_data, dtype=np.uint8)

    def reshape_rows(channels: int) -> np.ndarray:
        row_width = width * channels
        row_step = step or row_width
        if row_step == row_width:
            return data.reshape(height, width, channels)
        rows = data.reshape(height, row_step)[:, :row_width]
        return rows.reshape(height, width, channels)

    if encoding in {"rgb8", "bgr8"}:
        array = reshape_rows(3)
        if encoding == "bgr8":
            array = array[:, :, ::-1]
    elif encoding in {"rgba8", "bgra8"}:
        array = reshape_rows(4)[:, :, :3]
        if encoding == "bgra8":
            array = array[:, :, ::-1]
    elif encoding in {"mono8", "8uc1"}:
        gray = reshape_rows(1)[:, :, 0]
        array = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        raise ValueError(f"Unsupported raw image encoding: {encoding!r}")

    return Image.fromarray(array.copy(), mode="RGB")


def image_msg_to_pil(msg: Any) -> Image.Image:
    if hasattr(msg, "format") or "compressed" in type(msg).__name__.lower():
        return compressed_image_to_pil(msg)
    return raw_image_to_pil(msg)


def save_image(msg: Any, out_path: Path, image_ext: str, jpeg_quality: int) -> None:
    image = image_msg_to_pil(msg)
    if image_ext == "jpg":
        image.save(out_path, format="JPEG", quality=int(jpeg_quality))
    else:
        image.save(out_path, format="PNG")


def compressed_payload(msg: Any) -> bytes | None:
    if not (hasattr(msg, "format") or "compressed" in type(msg).__name__.lower()):
        return None
    data = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(msg.data)
    return data


def save_indexed_images(
    image_connections: Sequence[Any],
    sample_ts: np.ndarray,
    images_dir: Path,
    image_ext: str,
    jpeg_quality: int,
) -> list[int] | None:
    wanted = {int(ts) for ts in sample_ts}
    saved_ts: list[int] = []
    saved_set: set[int] = set()
    suffix = f".{image_ext}"

    for conn in image_connections:
        indexes = connection_indexes(conn)
        if indexes is None:
            return None
        msgtype = str(conn.msgtype)
        for entry in indexes:
            timestamp_ns = int(entry.time)
            if timestamp_ns not in wanted or timestamp_ns in saved_set:
                continue
            raw = read_rosbag1_raw(conn, entry)
            out_path = images_dir / f"{timestamp_ns}{suffix}"
            if is_compressed_image_type(msgtype):
                image_format, payload = parse_compressed_image(raw)
                if image_ext == "jpg" and is_jpeg_payload(image_format, payload):
                    out_path.write_bytes(payload)
                else:
                    with Image.open(io.BytesIO(payload)) as image:
                        image = image.convert("RGB")
                        if image_ext == "jpg":
                            image.save(out_path, format="JPEG", quality=int(jpeg_quality))
                        else:
                            image.save(out_path, format="PNG")
            elif is_raw_image_type(msgtype):
                save_image(parse_raw_image(raw), out_path, image_ext, jpeg_quality)
            else:
                raise ValueError(f"Unsupported image msgtype: {conn.msgtype!r}")
            saved_ts.append(timestamp_ns)
            saved_set.add(timestamp_ns)
            if len(saved_ts) == len(sample_ts):
                return saved_ts
    return saved_ts


def save_streamed_images(
    reader: AnyReader,
    image_connections: Sequence[Any],
    sample_ts: np.ndarray,
    images_dir: Path,
    image_ext: str,
    jpeg_quality: int,
) -> list[int]:
    wanted = {int(ts) for ts in sample_ts}
    saved_ts: list[int] = []
    saved_set: set[int] = set()
    suffix = f".{image_ext}"
    start = int(sample_ts[0])
    stop = int(sample_ts[-1]) + 1
    for conn, timestamp_ns, raw in reader.messages(connections=image_connections, start=start, stop=stop):
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns not in wanted or timestamp_ns in saved_set:
            continue
        msg = reader.deserialize(raw, conn.msgtype)
        out_path = images_dir / f"{timestamp_ns}{suffix}"
        payload = compressed_payload(msg)
        if image_ext == "jpg" and payload is not None:
            out_path.write_bytes(payload)
        else:
            save_image(msg, out_path, image_ext, jpeg_quality)
        saved_ts.append(timestamp_ns)
        saved_set.add(timestamp_ns)
        if len(saved_ts) == len(sample_ts):
            break
    return saved_ts


def read_bag_header(reader: Any, bag_path: Path) -> tuple[int, int, int]:
    reader.bio = bag_path.open("rb")
    magic = reader.bio.readline().decode()
    if not magic.startswith("#ROSBAG V2.0"):
        raise RuntimeError(f"{bag_path} is not a ROS1 V2.0 bag.")
    header = Header.read(reader.bio, RecordType.BAGHEADER)
    index_pos = header.get_uint64("index_pos")
    conn_count = header.get_uint32("conn_count")
    data_len = read_uint32(reader.bio)
    reader.bio.seek(data_len, 1)
    data_start = reader.bio.tell()
    if index_pos == 0:
        raise RuntimeError(f"{bag_path} is not indexed. Reindex it before extraction.")
    return index_pos, conn_count, data_start


def read_index_connections(reader: Any, index_pos: int, conn_count: int) -> list[Any]:
    reader.bio.seek(index_pos)
    connections = [reader.read_connection() for _ in range(conn_count)]
    reader.connections = connections
    return connections


def save_raw_image_record(conn: Any, raw: bytes, out_path: Path, image_ext: str, jpeg_quality: int) -> None:
    msgtype = str(conn.msgtype)
    if is_compressed_image_type(msgtype):
        image_format, payload = parse_compressed_image(raw)
        if image_ext == "jpg" and is_jpeg_payload(image_format, payload):
            out_path.write_bytes(payload)
            return
        with Image.open(io.BytesIO(payload)) as image:
            image = image.convert("RGB")
            if image_ext == "jpg":
                image.save(out_path, format="JPEG", quality=int(jpeg_quality))
            else:
                image.save(out_path, format="PNG")
        return
    if is_raw_image_type(msgtype):
        save_image(parse_raw_image(raw), out_path, image_ext, jpeg_quality)
        return
    raise ValueError(f"Unsupported image msgtype: {conn.msgtype!r}")


def stream_write_scene(
    bag_path: Path,
    scene_dir: Path,
    scene_name: str,
    args: argparse.Namespace,
) -> int:
    reader = Reader(bag_path)
    try:
        index_pos, conn_count, data_start = read_bag_header(reader, bag_path)
        connections = read_index_connections(reader, index_pos, conn_count)
        available_topics = {conn.topic for conn in connections}
        image_topic = choose_topic(available_topics, args.image_topics)
        odom_topic = choose_topic(available_topics, args.odom_topics)
        if image_topic is None or odom_topic is None:
            logging.warning("%s skipped: missing image or odom topic.", bag_path.name)
            return 0

        if args.dry_run:
            logging.info(
                "%s -> %s: stream mode, image_topic=%s, odom_topic=%s",
                bag_path.name,
                scene_dir,
                image_topic,
                odom_topic,
            )
            return 0

        if scene_dir.exists():
            if not args.overwrite:
                logging.info("%s skipped: %s already exists. Use --overwrite to replace it.", bag_path.name, scene_dir)
                return 0
            shutil.rmtree(scene_dir)
        images_dir = scene_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        conn_by_id = {conn.id: conn for conn in connections}
        image_ids = {conn.id for conn in connections if conn.topic == image_topic}
        odom_ids = {conn.id for conn in connections if conn.topic == odom_topic}
        period_ns = int(round(1_000_000_000 / float(args.fps)))
        max_dt_ns = int(round(float(args.max_frame_dt) * 1_000_000_000))
        suffix = f".{args.image_ext}"

        odom_ts: list[int] = []
        odom_xyz_yaw: list[tuple[float, float, float, float]] = []
        saved_ts: list[int] = []
        saved_set: set[int] = set()
        first_odom_ts: int | None = None
        target_ts: int | None = None
        prev_image: tuple[int, Any, bytes] | None = None

        def maybe_init_target(current_image_ts: int) -> None:
            nonlocal target_ts
            if target_ts is None and first_odom_ts is not None:
                target_ts = max(int(first_odom_ts), int(current_image_ts))

        def save_candidate(candidate: tuple[int, Any, bytes], target: int) -> None:
            timestamp_ns, conn, raw = candidate
            if timestamp_ns in saved_set or abs(timestamp_ns - target) > max_dt_ns:
                return
            save_raw_image_record(
                conn,
                raw,
                images_dir / f"{timestamp_ns}{suffix}",
                args.image_ext,
                args.jpeg_quality,
            )
            saved_ts.append(timestamp_ns)
            saved_set.add(timestamp_ns)

        reader.bio.seek(data_start)
        while reader.bio.tell() < index_pos:
            try:
                header = Header.read(reader.bio)
            except Exception:
                break
            op = header.get_uint8("op")
            if op != RecordType.CHUNK:
                reader.bio.seek(read_uint32(reader.bio), 1)
                continue

            compression = header.get_string("compression")
            datasize = read_uint32(reader.bio)
            chunk_raw = decompressors[compression](read_bytes(reader.bio, datasize))
            chunk = io.BytesIO(chunk_raw)
            chunk_size = len(chunk_raw)
            while chunk.tell() + 4 <= chunk_size:
                try:
                    record_header = Header.read(chunk)
                except Exception:
                    break
                record_op = record_header.get_uint8("op")
                if chunk.tell() + 4 > chunk_size:
                    break
                data_size = read_uint32(chunk)
                if record_op != RecordType.MSGDATA:
                    chunk.seek(data_size, 1)
                    continue

                conn_id = record_header.get_uint32("conn")
                timestamp_ns = int(record_header.get_time("time"))
                if conn_id in odom_ids:
                    raw = read_bytes(chunk, data_size)
                    odom_ts.append(timestamp_ns)
                    odom_xyz_yaw.append(parse_odom_xyz_yaw(raw))
                    if first_odom_ts is None:
                        first_odom_ts = timestamp_ns
                        if prev_image is not None:
                            maybe_init_target(prev_image[0])
                    continue

                if conn_id in image_ids:
                    raw = read_bytes(chunk, data_size)
                    conn = conn_by_id[conn_id]
                    current = (timestamp_ns, conn, raw)
                    maybe_init_target(timestamp_ns)
                    if target_ts is not None:
                        while target_ts <= timestamp_ns:
                            if prev_image is None:
                                candidate = current
                            else:
                                prev_dt = abs(prev_image[0] - target_ts)
                                curr_dt = abs(timestamp_ns - target_ts)
                                candidate = prev_image if prev_dt <= curr_dt else current
                            save_candidate(candidate, target_ts)
                            target_ts += period_ns
                    prev_image = current
                    continue

                chunk.seek(data_size, 1)

        if len(saved_ts) == 0 or len(odom_ts) < 2:
            logging.warning("%s skipped: no synced image/odom samples found.", bag_path.name)
            shutil.rmtree(scene_dir)
            return 0

        order = np.argsort(np.asarray(odom_ts, dtype=np.int64))
        odom_sorted = np.asarray(odom_ts, dtype=np.int64)[order]
        pose_sorted = np.asarray(odom_xyz_yaw, dtype=np.float64)[order]
        odom_sorted, unique_odom_idx = np.unique(odom_sorted, return_index=True)
        pose_sorted = pose_sorted[unique_odom_idx]

        sample_ts = np.asarray(saved_ts, dtype=np.int64)
        keep = (sample_ts >= odom_sorted[0]) & (sample_ts <= odom_sorted[-1])
        if not np.all(keep):
            for timestamp_ns in sample_ts[~keep]:
                (images_dir / f"{int(timestamp_ns)}{suffix}").unlink(missing_ok=True)
            sample_ts = sample_ts[keep]
        if len(sample_ts) == 0:
            shutil.rmtree(scene_dir)
            return 0

        scan = BagScan(
            image_topic=image_topic,
            odom_topic=odom_topic,
            image_ts=sample_ts,
            odom_ts=odom_sorted,
            odom_xyz_yaw=pose_sorted,
        )
        trajectory = interpolate_trajectory(scan, sample_ts)
        np.save(scene_dir / "trajectory.npy", trajectory)
        np.save(scene_dir / "timestamps.npy", sample_ts.astype(np.int64))
        metadata = {
            "schema_version": 1,
            "source_bag": str(bag_path),
            "scene_name": scene_name,
            "image_topic": image_topic,
            "odom_topic": odom_topic,
            "fps": float(args.fps),
            "num_frames": int(len(sample_ts)),
            "trajectory_columns": ["x", "y", "z", "yaw"],
            "image_timestamp_unit": "ns",
            "timestamp_source": "bag_record_time",
            "mode": "stream",
        }
        with (scene_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return int(len(sample_ts))
    finally:
        reader.close()


def write_scene(
    fast_bag: FastBag,
    bag_path: Path,
    scene_dir: Path,
    scan: BagScan,
    sample_ts: np.ndarray,
    args: argparse.Namespace,
) -> int:
    if sample_ts.size == 0:
        logging.warning("%s skipped: no frames matched the %.3f FPS grid.", bag_path.name, args.fps)
        return 0

    if scene_dir.exists():
        if not args.overwrite:
            logging.info("%s skipped: %s already exists. Use --overwrite to replace it.", bag_path.name, scene_dir)
            return 0
        shutil.rmtree(scene_dir)

    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if scan.odom_xyz_yaw is None:
        trajectory = interpolate_indexed_trajectory(fast_bag.odom_connections, sample_ts)
    else:
        trajectory = interpolate_trajectory(scan, sample_ts)

    saved_ts = save_indexed_images(
        fast_bag.image_connections,
        sample_ts,
        images_dir,
        args.image_ext,
        args.jpeg_quality,
    )
    if saved_ts is None:
        raise RuntimeError("Image topic is not indexed.")

    if len(saved_ts) != len(sample_ts):
        saved = set(saved_ts)
        keep = np.asarray([int(ts) in saved for ts in sample_ts], dtype=bool)
        missing = int(len(sample_ts) - keep.sum())
        logging.warning("%s: dropped %d selected frame(s) that could not be saved.", bag_path.name, missing)
        sample_ts = sample_ts[keep]
        trajectory = trajectory[keep]
        if len(sample_ts) == 0:
            shutil.rmtree(scene_dir)
            return 0

    order = {int(ts): idx for idx, ts in enumerate(sample_ts)}
    saved_ts_sorted = sorted(saved_ts, key=lambda ts: order.get(int(ts), len(order)))
    if saved_ts_sorted != [int(ts) for ts in sample_ts]:
        # Keep files as timestamp names, but make trajectory/timestamp arrays match saved file order.
        reorder = np.asarray([order[int(ts)] for ts in saved_ts_sorted], dtype=np.int64)
        sample_ts = sample_ts[reorder]
        trajectory = trajectory[reorder]

    np.save(scene_dir / "trajectory.npy", trajectory)
    np.save(scene_dir / "timestamps.npy", sample_ts.astype(np.int64))
    metadata = {
        "schema_version": 1,
        "source_bag": str(bag_path),
        "scene_name": scene_dir.name,
        "image_topic": scan.image_topic,
        "odom_topic": scan.odom_topic,
        "fps": float(args.fps),
        "num_frames": int(len(sample_ts)),
        "trajectory_columns": ["x", "y", "z", "yaw"],
        "image_timestamp_unit": "ns",
        "timestamp_source": "bag_record_time",
    }
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return int(len(sample_ts))


def main() -> None:
    setup_logging()
    args = parse_args()
    require_rosbags()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root does not exist: {args.dataset_root}")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0.")
    if args.max_frame_dt < 0:
        raise ValueError("--max-frame-dt must be >= 0.")

    if args.sort == "size":
        bag_paths = sorted(args.dataset_root.rglob("*.bag"), key=lambda path: (path.stat().st_size, natural_key(path)))
    else:
        bag_paths = sorted(args.dataset_root.rglob("*.bag"), key=natural_key)
    if args.max_bags is not None:
        bag_paths = bag_paths[: int(args.max_bags)]
    if not bag_paths:
        raise FileNotFoundError(f"No .bag files found under {args.dataset_root}")

    scene_names = scene_names_for_bags(bag_paths, args.dataset_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    written_scenes = 0
    for bag_path in tqdm(bag_paths, desc="SCAND bags", unit="bag"):
        fast_bag = None
        try:
            scene_dir = args.out_root / scene_names[bag_path]
            if args.mode == "stream":
                frames = stream_write_scene(bag_path, scene_dir, scene_names[bag_path], args)
                if args.dry_run:
                    continue
                if frames > 0:
                    written_scenes += 1
                    total_frames += frames
                continue

            fast_bag = open_fast_bag(bag_path, args.image_topics, args.odom_topics)
            if fast_bag is None:
                continue
            scan = fast_bag.scan()
            sample_ts = nearest_sample_timestamps(scan, float(args.fps), float(args.max_frame_dt))
            if args.dry_run:
                logging.info(
                    "%s -> %s: %d frames, image_topic=%s, odom_topic=%s",
                    bag_path.name,
                    scene_dir,
                    len(sample_ts),
                    scan.image_topic,
                    scan.odom_topic,
                )
                continue
            frames = write_scene(fast_bag, bag_path, scene_dir, scan, sample_ts, args)
        except Exception as exc:
            logging.warning("%s skipped: %s", bag_path.name, exc)
            continue
        finally:
            if fast_bag is not None:
                fast_bag.close()
        if frames > 0:
            written_scenes += 1
            total_frames += frames

    if args.dry_run:
        logging.info("Dry run complete for %d bag(s).", len(bag_paths))
    else:
        logging.info("Wrote %d scene(s), %d sampled frame(s) to %s.", written_scenes, total_frames, args.out_root)


if __name__ == "__main__":
    main()
