"""
DASE7140 Object Tracking — Single Pedestrian Tracker
Approach: YOLOv8n per-frame detection + motion prediction + color histogram matching
No reliance on tracker IDs; each frame selects the best-matching person independently.
"""

import os
import sys

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


def get_box_center_area(box):
    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0], (x2 - x1) * (y2 - y1), [x1, y1, x2, y2]


def score_initial_box(center, area, img_w, img_h):
    """Score for selecting target on frame 0."""
    expected_cx = img_w * 0.5
    expected_cy = img_h * 0.75
    dist = ((center[0] - expected_cx) ** 2 + (center[1] - expected_cy) ** 2) ** 0.5
    return area * 3.0 - dist * 0.5 + center[1] * 1.0


def extract_hsv_hist(frame, box):
    """Extract 2D H-S histogram from a bounding box region."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def hist_similarity(hist_a, hist_b):
    """Bhattacharyya distance between two histograms (lower is more similar)."""
    if hist_a is None or hist_b is None:
        return 1.0
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)


def predict_center(center_history, lost_frames):
    """Linear extrapolation from recent centers."""
    if len(center_history) < 2:
        return center_history[-1] if center_history else [0, 0]
    n = min(5, len(center_history) - 1)
    vx = (center_history[-1][0] - center_history[-n - 1][0]) / (n + 1)
    vy = (center_history[-1][1] - center_history[-n - 1][1]) / (n + 1)
    return [
        center_history[-1][0] + vx * lost_frames,
        center_history[-1][1] + vy * lost_frames,
    ]


def main():
    video_path = "sample.mp4"
    output_path = "result_nocc.mp4"
    model_path = "yolov8n.pt"

    print("Loading YOLOv8n model...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    target_hist = None
    target_center = None
    target_area = None
    expected_area = None
    center_history = deque(maxlen=10)
    lost_count = 0

    print(f"Processing {total_frames} frames...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes
        persons = []
        if boxes is not None:
            for box in boxes:
                if int(box.cls[0]) == 0:  # person
                    center, area, xyxy = get_box_center_area(box)
                    persons.append({"center": center, "area": area, "box": xyxy})

        # Frame 0: select target
        if frame_idx == 0:
            best_score = -float("inf")
            best = None
            for p in persons:
                score = score_initial_box(p["center"], p["area"], width, height)
                if score > best_score:
                    best_score = score
                    best = p
            if best:
                target_center = best["center"]
                target_area = best["area"]
                expected_area = target_area
                target_box = best["box"]
                target_hist = extract_hsv_hist(frame, target_box)
                center_history.append(target_center)
                print(f"Frame 0: target selected at {target_center}, area={target_area:.0f}")
            else:
                print("Warning: no person found in frame 0")

        found = False
        best_match = None
        best_match_score = -float("inf")

        if target_center is not None:
            predicted = predict_center(list(center_history), lost_count + 1)

            for p in persons:
                # Position score: distance from prediction
                dist = ((p["center"][0] - predicted[0]) ** 2 + (p["center"][1] - predicted[1]) ** 2) ** 0.5
                pos_score = -dist * 0.3

                # Area score: penalize large deviation from expected area
                # Target moving away -> area shrinks roughly as 1/distance^2
                # We just penalize deviations
                area_ratio = p["area"] / (target_area + 1e-6)
                area_score = -abs(np.log(area_ratio + 1e-6)) * 50.0

                # Color score
                person_hist = extract_hsv_hist(frame, p["box"])
                color_dist = hist_similarity(target_hist, person_hist)
                color_score = -color_dist * 300.0

                # Combined score
                score = pos_score + area_score + color_score

                if score > best_match_score:
                    best_match_score = score
                    best_match = p

            # Accept match if it is reasonably close to prediction
            if best_match:
                dist_to_pred = ((best_match["center"][0] - predicted[0]) ** 2 +
                                (best_match["center"][1] - predicted[1]) ** 2) ** 0.5
                # Allow larger distance if target is far away (smaller area)
                distance_threshold = max(200, 400 * (best_match["area"] / (target_area + 1e-6)))
                if dist_to_pred < distance_threshold:
                    target_center = best_match["center"]
                    target_area = best_match["area"]
                    target_box = best_match["box"]
                    center_history.append(target_center)
                    # Update histogram occasionally (exponential moving average)
                    new_hist = extract_hsv_hist(frame, target_box)
                    if new_hist is not None and target_hist is not None:
                        target_hist = 0.9 * target_hist + 0.1 * new_hist
                        cv2.normalize(target_hist, target_hist, 0, 1, cv2.NORM_MINMAX)
                    found = True
                    lost_count = 0
                else:
                    lost_count += 1
            else:
                lost_count += 1

        # Draw
        annotated = frame.copy()
        if found:
            x1, y1, x2, y2 = map(int, target_box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = "Target"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        elif target_center is not None:
            # Draw dashed box at predicted position (same size as last known)
            x1, y1, x2, y2 = map(int, target_box)
            color = (0, 165, 255)
            thickness = 2
            dash_len = 15
            for i in range(x1, x2, dash_len * 2):
                cv2.line(annotated, (i, y1), (min(i + dash_len, x2), y1), color, thickness)
                cv2.line(annotated, (i, y2), (min(i + dash_len, x2), y2), color, thickness)
            for i in range(y1, y2, dash_len * 2):
                cv2.line(annotated, (x1, i), (x1, min(i + dash_len, y2)), color, thickness)
                cv2.line(annotated, (x2, i), (x2, min(i + dash_len, y2)), color, thickness)
            cv2.putText(annotated, "Lost", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out.write(annotated)

        if (frame_idx + 1) % 50 == 0 or frame_idx == total_frames - 1:
            status = "OK" if found else f"Lost({lost_count})"
            print(f"  Processed {frame_idx + 1}/{total_frames} frames — {status}")

    cap.release()
    out.release()
    print(f"Done. Output saved to {output_path}")


if __name__ == "__main__":
    main()
