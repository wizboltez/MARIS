"""Fish Detection, Multi-Object Tracking & Behavioral Anomaly Analysis.

Implements fish tracking and behavior analysis (ByteTrack/DeepSORT logic)
to measure behavioral changes (swimming velocity, erratic turns, lethargy,
plastic avoidance) as required by the Marine Biodiversity Impact Assessment.
"""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


class SimpleTrack:
    """Represents a tracked fish across frames."""
    def __init__(self, track_id: int, bbox: np.ndarray, cls_name: str, conf: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.cls_name = cls_name
        self.conf = conf
        self.history: list[tuple[float, float]] = [self.center(bbox)]
        self.velocities: list[float] = []
        self.angles: list[float] = []
        self.age = 1
        self.time_since_update = 0

    @staticmethod
    def center(box: np.ndarray) -> tuple[float, float]:
        return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)

    def update(self, bbox: np.ndarray, conf: float):
        old_c = self.history[-1]
        new_c = self.center(bbox)
        vx = new_c[0] - old_c[0]
        vy = new_c[1] - old_c[1]
        vel = float(np.sqrt(vx**2 + vy**2))
        self.velocities.append(vel)
        if len(self.history) > 1:
            prev_vx = self.history[-1][0] - self.history[-2][0]
            prev_vy = self.history[-1][1] - self.history[-2][1]
            a1 = np.arctan2(prev_vy, prev_vx)
            a2 = np.arctan2(vy, vx)
            d_angle = abs(float(np.degrees(a2 - a1)))
            if d_angle > 180:
                d_angle = 360 - d_angle
            self.angles.append(d_angle)
        self.history.append(new_c)
        self.bbox = bbox
        self.conf = conf
        self.age += 1
        self.time_since_update = 0


class FishBehaviorTracker:
    """ByteTrack/Kalman-inspired tracker and behavioral stress quantification."""

    def __init__(self, max_cosine_dist: float = 0.5, iou_thresh: float = 0.3):
        self.iou_thresh = iou_thresh
        self.next_id = 1
        self.tracks: list[SimpleTrack] = []

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, detections: list[tuple[np.ndarray, str, float]]) -> list[SimpleTrack]:
        """detections: list of (bbox, class_name, conf)."""
        assigned_det = set()
        assigned_track = set()

        # Match existing tracks with detections based on IoU
        for t_idx, track in enumerate(self.tracks):
            best_iou = self.iou_thresh
            best_d_idx = -1
            for d_idx, (bbox, cls_name, conf) in enumerate(detections):
                if d_idx in assigned_det:
                    continue
                iou = self.compute_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx
            if best_d_idx != -1:
                track.update(detections[best_d_idx][0], detections[best_d_idx][2])
                assigned_det.add(best_d_idx)
                assigned_track.add(t_idx)

        # Increment age for unassigned tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in assigned_track:
                track.time_since_update += 1

        # Drop old tracks (older than 10 missed frames)
        self.tracks = [t for t in self.tracks if t.time_since_update <= 10]

        # Create new tracks for unassigned detections
        for d_idx, (bbox, cls_name, conf) in enumerate(detections):
            if d_idx not in assigned_det:
                new_track = SimpleTrack(self.next_id, bbox, cls_name, conf)
                self.next_id += 1
                self.tracks.append(new_track)

        return self.tracks


def analyze_fish_behavior(
    fish_detections: list[dict],
    plastic_boxes: list[np.ndarray] | None = None,
    image_shape: tuple[int, int] = (480, 640),
) -> dict:
    """Calculates behavioral statistics and the Fish Behavioral Anomaly Index (0-100).

    fish_detections: list of dicts with 'bbox', 'cls_name', 'conf' (and optional 'track_history')
    plastic_boxes: list of plastic bounding boxes in [x1, y1, x2, y2]
    """
    count = len(fish_detections)
    if count == 0:
        return {
            "fish_count": 0,
            "average_speed_px": 0.0,
            "speed_variance": 0.0,
            "erratic_turn_rate_deg": 0.0,
            "plastic_proximity_hazard": 0.0,
            "behavioral_anomaly_index": 20.0,  # Baseline normal
            "behavior_state": "No Fish Detected / Normal Habitat Baseline",
            "tracks": [],
        }

    has_motion = False
    speeds = []
    turn_angles = []
    plastic_hazards = []

    img_diag = float(np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2))

    for i, fish in enumerate(fish_detections):
        box = np.array(fish["bbox"])
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

        # Speed / turning only exist when measured from multi-frame tracks
        # (src.modules.fish_tracking). From a single image they are unknown -> 0,
        # so the anomaly index then reflects plastic proximity only.
        speed = float(fish.get("speed", 0.0))
        angle = float(fish.get("turn_angle", 0.0))
        has_motion = has_motion or ("speed" in fish)
        speeds.append(speed)
        turn_angles.append(angle)

        # Proximity to plastic debris
        if plastic_boxes and len(plastic_boxes) > 0:
            min_dist = float("inf")
            for pbox in plastic_boxes:
                pcx, pcy = (pbox[0] + pbox[2]) / 2.0, (pbox[1] + pbox[3]) / 2.0
                dist = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
                min_dist = min(min_dist, dist)
            # Hazard scales inversely with distance (closer = higher hazard)
            hazard = max(0.0, 1.0 - (min_dist / (img_diag * 0.4)))
            plastic_hazards.append(hazard)
        else:
            plastic_hazards.append(0.0)

    avg_speed = float(np.mean(speeds))
    speed_var = float(np.var(speeds))
    avg_turn = float(np.mean(turn_angles))
    avg_hazard = float(np.mean(plastic_hazards))

    # Calculate Behavioral Anomaly Index (BAI, 0-100)
    # Stressors: erratic swim turns (>45 deg), abnormal speed variance, high plastic proximity
    turn_factor = min(40.0, (avg_turn / 90.0) * 40.0)
    hazard_factor = avg_hazard * 35.0
    speed_factor = min(25.0, (speed_var / 20.0) * 25.0)

    bai = float(np.clip(turn_factor + hazard_factor + speed_factor, 0.0, 100.0))

    if bai < 30:
        state = "Normal Foraging / Natural Cohesive Swimming"
    elif bai < 55:
        state = "Mild Agitation / Minor Plastic Avoidance Reaction"
    elif bai < 75:
        state = "High Stress / Erratic Swimming & Debris Entanglement Risk"
    else:
        state = "Critical Disturbance / Hypoxic Lethargy or Entrapment Panic"

    return {
        "fish_count": count,
        "average_speed_px": round(avg_speed, 2),
        "speed_variance": round(speed_var, 2),
        "erratic_turn_rate_deg": round(avg_turn, 2),
        "plastic_proximity_hazard": round(avg_hazard * 100.0, 2),
        "behavioral_anomaly_index": round(bai, 2),
        "behavior_state": state if has_motion else f"{state} (single image: plastic proximity only)",
        "measurement": "multi-frame tracks" if has_motion else "single image (no motion data)",
    }


def draw_fish_tracks_and_behavior(
    image: np.ndarray,
    fish_detections: list[dict],
    stats: dict,
    plastic_boxes: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Overlays single-image fish detections, plastic proximity zones and the anomaly index.

    Real multi-frame trajectories are drawn by src.modules.fish_tracking.draw_trajectories.
    """
    out = image.copy()
    h, w = out.shape[:2]

    # Draw plastic danger zones if plastic present
    if plastic_boxes:
        for pbox in plastic_boxes:
            px1, py1, px2, py2 = [int(v) for v in pbox]
            pcx, pcy = int((px1 + px2) / 2), int((py1 + py2) / 2)
            cv2.circle(out, (pcx, pcy), 65, (0, 0, 230), 1, lineType=cv2.LINE_AA)

    for i, fish in enumerate(fish_detections):
        box = [int(v) for v in fish["bbox"]]
        cls_name = fish.get("cls_name", "fish")
        conf = fish.get("conf", 0.8)
        track_id = fish.get("track_id", i + 1)

        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

        # Draw box in vibrant cyan/lime
        color = (255, 200, 0)  # BGR cyan-blue
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 2)

        label = f"ID #{track_id} {cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (box[0], max(0, box[1] - th - 6)), (box[0] + tw + 4, box[1]), color, -1)
        cv2.putText(out, label, (box[0] + 2, box[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # Banner overlay at bottom with Behavioral Anomaly Index
    bai = stats.get("behavioral_anomaly_index", 20.0)
    state = stats.get("behavior_state", "Normal")
    banner_text = f"Fish BAI: {bai:.1f}% | State: {state}"
    b_color = (0, 180, 0) if bai < 40 else ((0, 165, 255) if bai < 70 else (0, 0, 220))
    cv2.rectangle(out, (10, h - 36), (w - 10, h - 8), (20, 20, 20), -1)
    cv2.rectangle(out, (10, h - 36), (w - 10, h - 8), b_color, 2)
    cv2.putText(out, banner_text, (18, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def plot_fish_behavior_summary(stats: dict, save_path: str | None = None) -> plt.Figure:
    """Generates an informative multi-panel plot of fish behavioral dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel 1: Gauge-like bar for BAI
    bai = stats.get("behavioral_anomaly_index", 25.0)
    colors = ["#2ecc71" if bai < 35 else "#f39c12" if bai < 70 else "#e74c3c"]
    axes[0].bar(["Anomaly Index"], [bai], color=colors, width=0.4)
    axes[0].set_ylim(0, 100)
    axes[0].axhline(35, color="gray", linestyle="--", alpha=0.6, label="Mild Stress")
    axes[0].axhline(70, color="red", linestyle="--", alpha=0.6, label="Critical Stress")
    axes[0].set_ylabel("Score (0-100)")
    axes[0].set_title(f"Behavioral Anomaly Index\n({stats.get('behavior_state', 'Normal')[:24]}...)", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)

    # Panel 2: Metrics breakdown
    metrics = ["Avg Speed (px)", "Speed Var", "Turn Erratic (°)", "Plastic Hazard %"]
    vals = [
        stats.get("average_speed_px", 0.0),
        stats.get("speed_variance", 0.0),
        stats.get("erratic_turn_rate_deg", 0.0),
        stats.get("plastic_proximity_hazard", 0.0),
    ]
    axes[1].barh(metrics, vals, color="#3498db")
    axes[1].set_xlabel("Value")
    axes[1].set_title("Movement & Spatial Metrics", fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
