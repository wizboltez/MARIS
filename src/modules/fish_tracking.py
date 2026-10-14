"""Multi-frame fish tracking (ByteTrack) and behaviour metrics on TrashCan video sequences.

TrashCan images are frames from ROV videos (vid_<id>_frame<n>.jpg) and the test split
keeps whole videos together, so consecutive frames can be tracked. For each video:

1. YOLO11-seg + ByteTrack (Ultralytics `model.track(persist=True)`) assigns track IDs.
2. Camera (ROV) motion between frames is estimated with phase correlation on the
   grayscale frames and subtracted, so fish speed is relative to the scene, not the camera.
3. Per track: speed (px/frame), speed variance, turning angle between successive
   displacement vectors, and distance to the nearest plastic detection in the same frame.

Limitations: frames are sampled from the original videos (not every native frame), and
phase correlation only removes global translation (not zoom/rotation).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, plastic_conf_threshold

FISH_IDS = [i for i, n in enumerate(CLASS_NAMES) if n.startswith("animal_")]
# Animals that actually swim/move; behaviour metrics use only these.
MOBILE_IDS = [CLASS_NAMES.index(n) for n in ("animal_fish", "animal_eel", "animal_crab")]
# (Near-)sessile animals: their apparent motion after camera compensation is a measure
# of the remaining camera-motion error, i.e. the noise floor for speed.
STATIC_IDS = [CLASS_NAMES.index(n) for n in ("animal_starfish", "animal_shells")]
TRACKER_CFG = str(Path(__file__).with_name("bytetrack_trashcan.yaml"))
MIN_STEP_PX = 2.0  # displacement below this is within camera-compensation noise
PLASTIC_ID = CLASS_NAMES.index("trash_plastic")
FRAME_RE = re.compile(r"vid_(\d+)_frame(\d+)")


def group_frames_by_video(image_paths: list[Path]) -> dict[str, list[Path]]:
    """{video_id: [frame paths sorted by frame number]}"""
    videos: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in image_paths:
        m = FRAME_RE.match(p.stem)
        if m:
            videos[m.group(1)].append((int(m.group(2)), p))
    return {v: [p for _, p in sorted(fs)] for v, fs in videos.items()}


def estimate_camera_shift(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    """Global (dx, dy) translation of the scene between two frames, in pixels."""
    if prev_gray.shape != gray.shape:
        return 0.0, 0.0
    win = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(prev_gray.astype(np.float32), gray.astype(np.float32), win)
    if response < 0.05:  # unreliable estimate (e.g. featureless water)
        return 0.0, 0.0
    return float(dx), float(dy)


def load_tracking_model(weights_path) -> "YOLO":
    """Separate YOLO instance for tracking: track() registers tracker callbacks on the
    model it runs on, which would otherwise alter later plain predict() calls."""
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def track_video(yolo_model, frame_paths: list[Path], conf: float = 0.05, plastic_conf: float | None = None) -> pd.DataFrame:
    """Run ByteTrack over one video's frames (pass a model from load_tracking_model()).

    Returns one row per (frame, detection) for fish and plastic, with track_id for fish
    and the camera shift of that frame relative to the previous one.
    """
    if plastic_conf is None:
        plastic_conf = plastic_conf_threshold()
    rows = []
    prev_gray = None

    for idx, path in enumerate(frame_paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cam_dx, cam_dy = estimate_camera_shift(prev_gray, gray) if prev_gray is not None else (0.0, 0.0)
        prev_gray = gray

        # persist=False on the first frame resets the tracker for this video
        res = yolo_model.track(img, conf=conf, persist=idx > 0, tracker=TRACKER_CFG, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            rows.append({"frame_idx": idx, "image_id": path.stem, "cls_id": -1, "cam_dx": cam_dx, "cam_dy": cam_dy,
                         "img_h": img.shape[0], "img_w": img.shape[1]})
            continue
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(res.boxes)
        for box, cid, tid, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), ids,
                                    res.boxes.conf.cpu().numpy()):
            if cid not in FISH_IDS and not (cid == PLASTIC_ID and c >= plastic_conf):
                continue
            rows.append({
                "frame_idx": idx, "image_id": path.stem, "cls_id": int(cid), "cls_name": CLASS_NAMES[cid],
                "track_id": int(tid), "conf": float(c),
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "cx": (box[0] + box[2]) / 2, "cy": (box[1] + box[3]) / 2,
                "cam_dx": cam_dx, "cam_dy": cam_dy, "img_h": img.shape[0], "img_w": img.shape[1],
            })
    return pd.DataFrame(rows)


def compute_track_metrics(det: pd.DataFrame) -> pd.DataFrame:
    """Per-track behaviour metrics from a track_video() result."""
    if det.empty or "track_id" not in det:
        return pd.DataFrame()
    fish = det[det["cls_id"].isin(FISH_IDS) & (det["track_id"] >= 0)].sort_values("frame_idx")
    plastic = det[det["cls_id"] == PLASTIC_ID]
    plastic_by_frame = {f: g[["cx", "cy"]].to_numpy() for f, g in plastic.groupby("frame_idx")}
    cam_by_frame = det.groupby("frame_idx")[["cam_dx", "cam_dy"]].first()

    out = []
    for tid, tr in fish.groupby("track_id"):
        tr = tr.sort_values("frame_idx")
        pts = tr[["cx", "cy"]].to_numpy()
        frames = tr["frame_idx"].to_numpy()
        diag = float(np.hypot(tr["img_h"].iloc[0], tr["img_w"].iloc[0]))

        # Camera-compensated displacement per frame step
        disp = []
        for k in range(1, len(pts)):
            steps = frames[k] - frames[k - 1]
            cam = cam_by_frame.loc[frames[k - 1] + 1: frames[k]].sum().to_numpy()
            disp.append((pts[k] - pts[k - 1] - cam) / max(1, steps))
        disp = np.array(disp) if disp else np.zeros((0, 2))
        speeds = np.linalg.norm(disp, axis=1) if len(disp) else np.array([])

        turns = []
        for a, b in zip(disp[:-1], disp[1:]):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > MIN_STEP_PX and nb > MIN_STEP_PX:  # angle of near-stationary steps is noise
                turns.append(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))

        dists = []
        for f, (cx, cy) in zip(frames, pts):
            if f in plastic_by_frame:
                dists.append(float(np.min(np.hypot(*(plastic_by_frame[f] - [cx, cy]).T))) / diag)

        out.append({
            "track_id": int(tid),
            "species": tr["cls_name"].mode().iloc[0].replace("animal_", ""),
            "mobile": bool(tr["cls_id"].isin(MOBILE_IDS).mean() >= 0.5),
            "n_frames": len(tr),
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "mean_speed_px": float(speeds.mean()) if len(speeds) else np.nan,
            "speed_var": float(speeds.var()) if len(speeds) > 1 else np.nan,
            "mean_turn_deg": float(np.mean(turns)) if turns else np.nan,
            "min_plastic_dist_norm": float(min(dists)) if dists else np.nan,
            "frames_with_plastic": len(dists),
            "image_ids": list(tr["image_id"]),
            "path_x": list(pts[:, 0]),
            "path_y": list(pts[:, 1]),
        })
    return pd.DataFrame(out)


def fish_stats_for_frame(tracks: pd.DataFrame, image_id: str, plastic_boxes, image_shape) -> dict | None:
    """Behaviour stats for the fish tracks visible in one frame (None if no tracks there).

    Feeds measured speed / turning into the existing Behavioral Anomaly Index formula
    (src.modules.fish_behavior.analyze_fish_behavior) via its 'speed' / 'turn_angle' fields.
    """
    from src.modules.fish_behavior import analyze_fish_behavior

    if tracks is None or tracks.empty:
        return None
    visible = tracks[tracks["mobile"] & tracks["image_ids"].apply(lambda ids: image_id in ids)]
    moving = visible[visible["n_frames"] >= 2]
    if moving.empty:
        return None

    detections = []
    for _, t in moving.iterrows():
        k = t["image_ids"].index(image_id)
        cx, cy = t["path_x"][k], t["path_y"][k]
        detections.append({
            "bbox": [cx - 1, cy - 1, cx + 1, cy + 1],
            "cls_name": t["species"],
            "conf": 1.0,
            "speed": 0.0 if np.isnan(t["mean_speed_px"]) else t["mean_speed_px"],
            "turn_angle": 0.0 if np.isnan(t["mean_turn_deg"]) else t["mean_turn_deg"],
        })
    stats = analyze_fish_behavior(detections, plastic_boxes, image_shape=image_shape)
    stats["n_tracked_fish"] = len(moving)
    stats["mean_track_length_frames"] = float(moving["n_frames"].mean())
    stats["measurement"] = "multi-frame ByteTrack"
    return stats


def draw_trajectories(image_bgr: np.ndarray, tracks: pd.DataFrame, plastic_boxes=None) -> np.ndarray:
    """Draw every fish track's path (centres over time) on one frame."""
    out = image_bgr.copy()
    for box in plastic_boxes or []:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, "plastic", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    rng = np.random.default_rng(0)
    for _, t in tracks.iterrows():
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        pts = np.stack([t["path_x"], t["path_y"]], axis=1).astype(int)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[-1]), 4, color, -1)
        label = f"#{t['track_id']} {t['species']}"
        if not np.isnan(t["mean_speed_px"]):
            label += f" {t['mean_speed_px']:.1f}px/f"
        cv2.putText(out, label, (pts[-1][0] + 5, pts[-1][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out
