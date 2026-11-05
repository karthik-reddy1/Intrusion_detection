# Agentic Loitering and Restricted Area Intrusion Detection

A real-time intelligent video analytics system for security monitoring.  
The project detects people, tracks movement patterns, and generates alerts for:

- Restricted area intrusion
- Loitering behavior (optional, configurable)

It is implemented as a modular multi-agent pipeline with `YOLOv8`, `OpenCV`, `Shapely`, and rule-based decision logic.

---

## 1) Project Overview

### Why this project exists

Conventional CCTV systems record footage but usually require continuous manual monitoring.  
This project automates basic behavior analytics so operators can receive timely, actionable alerts instead of watching all camera feeds continuously.

### Core objectives

- Detect people robustly in live or recorded video streams.
- Identify when a person enters a defined restricted polygon.
- Detect low-movement prolonged presence (loitering) when enabled.
- Label event severity through threat levels.
- Notify stakeholders through configurable channels.
- Keep architecture modular and easy to extend.

### Primary use cases

- Warehouse restricted zone monitoring
- Campus/lab perimeter watch
- Office after-hours intrusion detection
- Industrial safety and unauthorized access surveillance

---

## 2) Key Features

- Real-time person detection and track ID assignment
- Configurable restricted zones using polygon coordinates
- Optional loitering logic using dwell time + movement threshold
- Threat-level classification (`low`, `medium`, `high`)
- Multi-channel notifications:
  - Console
  - Email (SMTP)
  - SMS (Twilio)
  - WhatsApp (Twilio)
- Alert cooldown to reduce repeated spam
- On-frame visualization (zones, boxes, labels, trajectories, info panel)
- Optional headless mode for server execution
- Optional annotated output video saving
- Interactive polygon annotation tool

---

## 3) System Architecture

The orchestrator class `LoiteringDetectionSystem` in `main.py` coordinates all agents per frame.

### High-level flow

1. Read and normalize frame from input source.
2. Detect people and track IDs.
3. For each tracked person:
   - Update trajectory features
   - Update memory state
   - Run decision logic
   - Trigger notification if required
   - Draw overlays
4. Cleanup stale tracks/states.
5. Show frame and/or write output video.

### Architecture Flow Diagram

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Video Input (`main.py`)                                                      │
│ - Reads `INPUT_SOURCE` (webcam/video/RTSP)                                   │
│ - Captures frames via OpenCV and normalizes frame size/FPS                   │
└──────────────────────────────────────────────────────────────────────────────┘
                               │
                               │ per-frame data
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PoseAgent (`agents/pose_agent.py`)                                           │
│ - Runs YOLO model                                                            │
│ - Detects persons and maintains tracking IDs                                 │
│ - Outputs bbox, centroid, confidence, keypoints                              │
└──────────────────────────────────────────────────────────────────────────────┘
                               │
                               │ detection list
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Per-Person Processing Loop (`LoiteringDetectionSystem.process_frame`)        │
│ - Iterates through each detected person                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                │                                     │
                │ trajectory update                   │ memory/state update
                ▼                                     ▼
┌───────────────────────────────────────┐   ┌──────────────────────────────────┐
│ TrajectoryAgent                       │   │ MemoryAgent                      │
│ (`agents/trajectory_agent.py`)        │   │ (`agents/memory_agent.py`)       │
│ - Updates path points                 │   │ - Stores entry/last seen frames  │
│ - Calculates movement distance/speed  │   │ - Computes dwell duration        │
└───────────────────────────────────────┘   └──────────────────────────────────┘
                │                                     │
                └──────────────────┬──────────────────┘
                                   │ merged person features
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ DecisionAgent (`agents/decision_agent.py`)                                   │
│ - Checks centroid inside restricted polygons (intrusion rule)                │
│ - Applies loitering rule (duration + low movement)                           │
│ - Returns `Normal` / `Loitering` / `Intrusion` + threat level                │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
                │ alert path                          │ visualization path
                ▼                                     ▼
┌───────────────────────────────────────┐   ┌──────────────────────────────────┐
│ NotificationAgent                     │   │ Visualizer                       │
│ (`agents/notification_agent.py`)      │   │ (`utils/visualizer.py`)          │
│ - Applies cooldown per person/zone    │   │ - Draws zones, boxes, labels     │
│ - Sends Console/Email/SMS/WhatsApp    │   │ - Draws trajectory + info panel  │
│ - Uses async threads for I/O channels │   │ - Renders annotated frame        │
└───────────────────────────────────────┘   └──────────────────────────────────┘
                │                                     │
                │ alerts                              │ annotated frames
                ▼                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Output Layer                                                                 │
│ - Live display window (`enable_display: true`)                               │
│ - Optional output video writer (`OUTPUT_PATH`)                               │
│ - Alert history in memory (`notification_agent.alert_history`)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4) Detailed Module Documentation

## `main.py`

Entrypoint and orchestrator.

Main responsibilities:
- Loads `.env` variables.
- Loads YAML configuration.
- Initializes all agents and utilities.
- Opens the selected video source.
- Limits processing to approximately 15 FPS for stability.
- Runs per-frame inference and decision pipeline.
- Handles display controls (`q` quit, `p` pause/resume).
- Writes output video when configured.

Important runtime behavior:
- Camera-like sources are coerced to configured dimensions.
- RTSP stream handling attempts lower buffering for lower latency.
- Processing uses `cap.grab()`/`cap.retrieve()` pattern to keep fresher frames.

## `agents/pose_agent.py` (`PoseAgent`)

Purpose:
- Detect people and maintain tracking IDs across frames.

Implementation notes:
- Uses `ultralytics.YOLO(...).track(...)`.
- Restricts class detection to person (`classes=[0]`).
- Tracks a lower-center point near feet instead of geometric bbox center.
  - This improves zone-inclusion logic for floor-level intrusion checks.

Output fields per detection:
- `person_id`
- `bbox` (`[x, y, w, h]`)
- `centroid` (lower-center point)
- `keypoints` (when pose model is active)
- `confidence`
- `timestamp` (frame number)

## `agents/trajectory_agent.py` (`TrajectoryAgent`)

Purpose:
- Maintain trajectory paths and motion features for each person.

Computes:
- Path history
- Estimated distance traveled (meters from pixel approximation)
- Current speed
- Movement direction
- `is_moving` boolean

Notes:
- Uses an approximate pixel-to-meter conversion factor.
- Should be calibrated per camera for physically accurate metrics.

## `agents/memory_agent.py` (`MemoryAgent`)

Purpose:
- Maintain short-term state per tracked person.

Stores:
- Entry frame/time
- Last seen frame/time
- First/last positions
- Dwell duration
- Active/inactive status

Also handles cleanup of stale inactive entries.

## `agents/decision_agent.py` (`DecisionAgent`)

Purpose:
- Convert motion/context features into semantic status labels.

Rules:
1. **Intrusion rule**
   - If centroid lies inside any restricted polygon -> `Intrusion`.
2. **Loitering rule** (if enabled)
   - Requires both:
     - `dwell_duration > loitering_duration_threshold`
     - `distance_traveled < min_movement_meters`
3. **Normal**
   - If no intrusion and loitering conditions are not met.

Threat assignment:
- Intrusion: from zone configuration.
- Loitering: `high` if near restricted boundary, else `medium`.
- Normal: `low`.

## `agents/notification_agent.py` (`NotificationAgent`)

Purpose:
- Build and dispatch human-readable alert messages.

Features:
- Maintains alert history.
- Supports console/email/SMS/WhatsApp channels.
- Uses thread pool for network-bound notification dispatch.
- Optionally uploads frame snapshots for WhatsApp media links (imgbb).

Environment-driven behavior:
- Missing required channel credentials gracefully skips that channel.

## `utils/visualizer.py` (`Visualizer`)

Purpose:
- Render visual context directly on processed frames.

Draws:
- Restricted polygons
- Bounding boxes and IDs
- Status-dependent colors
- Trajectory lines and points
- Optional pose keypoints
- Top info panel with active persons and recent alert

## `utils/bbox_annotator.py` (`BBoxAnnotator`)

Purpose:
- Utility to create restricted-zone polygons interactively.

Workflow:
- Open first frame from target video.
- Click vertices.
- Press Enter to save nested coordinate list.
- Paste list into `config/config.yaml`.

Good practice:
- Match annotation resolution with runtime frame resolution.

---

## 5) Project Structure

```text
Restricted/
├── agents/
│   ├── pose_agent.py
│   ├── trajectory_agent.py
│   ├── memory_agent.py
│   ├── decision_agent.py
│   ├── notification_agent.py
│   └── __init__.py
├── config/
│   └── config.yaml
├── utils/
│   ├── visualizer.py
│   ├── bbox_annotator.py
│   └── __init__.py
├── main.py
├── polygon_coordinates.txt
├── requirements.txt
├── .env
└── README.md
```

---

## 6) Environment and Requirements

- Python 3.8+
- Camera, RTSP stream, or video file source
- OS: Windows/Linux/macOS (OpenCV-compatible)
- Optional GPU acceleration:
  - NVIDIA CUDA
  - Apple MPS

Dependencies are listed in `requirements.txt`.

Install:

```bash
cd Restricted
pip install -r requirements.txt
```

---

## 7) Configuration Reference (`config/config.yaml`)

The config file controls all runtime behavior.

## `thresholds`

- `enable_loitering_detection`: enable/disable loitering logic
- `loitering_duration_seconds`: dwell threshold in seconds
- `min_movement_meters`: movement threshold for loitering rule
- `intrusion_alert_immediate`: semantic flag for immediate intrusion alerts

## `pose_detection`

- `loitering_on_model`: model when loitering enabled (pose-capable)
- `loitering_off_model`: model when loitering disabled
- `confidence_threshold`: detector confidence floor
- `device`: `cpu`, `cuda`, or `mps`

## `tracking`

- `method`: tracking mode label
- `max_age`: stale frame retention before cleanup
- `min_hits`: minimum track consistency threshold

## `restricted_zones`

Define one or more polygonal regions:

```yaml
restricted_zones:
  - name: "Area 1"
    polygon: [[502, 289], [632, 327], [613, 454], [463, 384]]
    threat_level: "high"
```

Each point is `[x, y]` in frame pixel coordinates.

## `notifications`

Controls all alert delivery channels:

- `enable_console`
- `enable_email`
- `enable_sms`
- `enable_whatsapp`
- `alert_cooldown_frames`
- recipient lists for each enabled channel

## `visualization`

- `enable_display`
- `show_pose_keypoints`
- `show_trajectory`
- `trajectory_length`
- `colors` (`normal`, `loitering`, `intrusion`, `restricted_zone`)

## `video`

- `fps`
- `frame_width`
- `frame_height`
- optional input/output defaults

Important:
- Runtime processing in `main.py` is capped near 15 FPS.

---

## 8) Runtime Environment Variables

Read by `main.py`:

- `INPUT_SOURCE` (default: `0`)
- `OUTPUT_PATH` (optional)
- `CONFIG_PATH` (default: `config/config.yaml`)

Example:

```env
INPUT_SOURCE=0
OUTPUT_PATH=output/output_video.mp4
CONFIG_PATH=config/config.yaml
```

### Input source examples

- Webcam: `INPUT_SOURCE=0`
- Video file: `INPUT_SOURCE=videos/test.mp4`
- RTSP stream: `INPUT_SOURCE=rtsp://username:password@ip:port/stream`

---

## 9) Notification Credentials

## Email (SMTP)

Required:
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`

Optional:
- `SMTP_STARTTLS=true`

## SMS (Twilio)

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_SMS_FROM`

## WhatsApp (Twilio)

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_WHATSAPP_FROM` (example: `whatsapp:+14155238886`)

Optional media upload:
- `IMGBB_API_KEY`

---

## 10) Running the Project

Basic run:

```bash
python main.py
```

Interactive controls (when display enabled):
- `q` -> quit
- `p` -> pause/resume

Headless mode:
- Set `visualization.enable_display: false`
- Stop process with `Ctrl+C`

---

## 11) Detection Logic in Detail

### Intrusion logic

Input:
- person centroid
- list of configured polygons

Rule:
- If point-in-polygon is true for any restricted zone:
  - status -> `Intrusion`
  - threat -> zone-defined threat

### Loitering logic

Enabled only when `thresholds.enable_loitering_detection` is true.

Person is loitering if:
- Dwell duration exceeds configured time threshold, and
- Total movement remains below configured movement threshold.

This combines temporal persistence + low mobility, reducing false positives.

### Cooldown logic

To prevent repeated duplicate alerts:
- System keeps per-person alert state.
- Re-alert requires cooldown frame gap and zone re-entry/new zone context.

---

## 12) Polygon Annotation Guide

Generate zone coordinates from a sample video:

```bash
python utils/bbox_annotator.py path/to/video.mp4 --output polygon_coordinates.txt --config config/config.yaml
```

Steps:
1. Click multiple vertices in clockwise/anticlockwise order.
2. Press Enter to save polygon.
3. Copy coordinate list into `restricted_zones` in config.
4. Re-run main pipeline and validate zone overlay.

Tips:
- Mark only floor-plane area where feet/centroid should trigger event.
- Avoid including walkways unless intentionally restricted.
- Recalibrate after camera angle or zoom changes.

---

## 13) Performance and Tuning

### Throughput tuning

- Use smaller YOLO models for higher FPS.
- Prefer GPU (`cuda`/`mps`) where available.
- Reduce frame resolution for heavy deployments.
- Keep `trajectory_length` moderate for lower render overhead.

### Accuracy tuning

- Adjust `confidence_threshold` carefully.
- Calibrate `min_movement_meters` per camera scene scale.
- Use realistic `loitering_duration_seconds` for your environment.
- Draw precise polygons aligned with floor boundaries.

### Alert quality tuning

- Use `alert_cooldown_frames` to reduce repetitive alerts.
- Start with console mode, then enable external channels.

---

## 14) Troubleshooting

- **No window appears**
  - Check `visualization.enable_display`.
  - Verify execution environment supports GUI display.

- **Cannot open video source**
  - Validate `INPUT_SOURCE`.
  - Check camera permissions/availability.
  - Verify RTSP URL credentials and network reachability.

- **No notifications sent**
  - Confirm channel is enabled in config.
  - Confirm all required env credentials are set.
  - Check network/firewall for SMTP/Twilio access.

- **Too many false intrusion alerts**
  - Re-draw polygons to match floor area.
  - Verify centroid placement aligns with target trigger area.

- **Loitering too sensitive / not sensitive enough**
  - Tune `loitering_duration_seconds` and `min_movement_meters`.

- **High CPU usage / slow processing**
  - Reduce resolution, switch model size, use GPU.

---

## 15) Known Constraints

- Distance estimation uses approximate pixel-to-meter mapping.
- Accuracy depends on camera placement, perspective, and scene calibration.
- Heavy occlusion/crowded scenes can impact tracking consistency.
- Alert delivery is best-effort per channel credential/network availability.

---

## 16) Best Practices for Deployment

- Store credentials in `.env` only; never hardcode secrets.
- Keep `.env` out of version control.
- Start with `enable_console: true` during validation.
- Verify overlays and labels visually before enabling production alerts.
- Use intrusion-only mode first if loitering thresholds are not yet calibrated.
- Regularly test each notification channel independently.

---

## 17) Quick Start Checklist

1. Install dependencies.
2. Configure restricted polygons.
3. Set `INPUT_SOURCE`.
4. Run with display enabled and validate overlays.
5. Tune thresholds for your scene.
6. Configure notification credentials.
7. Enable required channels.
8. Run in headless/server mode if needed.

---

## 18) Notes

- For intrusion-only mode, set `thresholds.enable_loitering_detection: false`.
- In YAML, duplicate keys are overwritten by the last occurrence.
  - Keep only one `email_recipients`, one `sms_recipients`, and one `whatsapp_recipients` key.
