import base64
import os
import queue
import threading
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import requests
from dotenv import load_dotenv
from ultralytics import YOLO


DESCRIPTION_PROMPT = textwrap.dedent(
    """
    You are a visual description model.
    Look at the person in this image and write a vivid, concrete description for a roast writer.
    Focus on clothing, expression, posture, style, accessories, and overall vibe.
    Keep it factual and visually grounded.
    Avoid sensitive attributes (race, religion, disability, sexuality, etc.).
    Return 4-7 short bullet points.
    """
).strip()

ROAST_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a sharp but safe roast comedian.
    Write playful roasts based only on the provided visual description.
    Rules:
    - Keep it funny, witty, and non-hateful.
    - Never target protected characteristics or sensitive traits.
    - No slurs, threats, or explicit sexual content.
    - Keep it to 4-6 short lines.
    """
).strip()


@dataclass
class Settings:
    camera_index: int = 0
    detection_confidence: float = 0.45
    min_box_area_ratio: float = 0.08
    stable_frames_required: int = 3
    cooldown_seconds: float = 12.0
    snapshot_path: str = "snapshot.jpg"
    ollama_host: str = "http://127.0.0.1:11434"
    vision_model: str = "qwen3.5:397b-cloud"
    roast_model: str = "qwen3.5:397b-cloud"
    request_timeout_seconds: int = 90


def read_settings() -> Settings:
    load_dotenv()
    return Settings(
        camera_index=int(os.getenv("CAMERA_INDEX", "0")),
        detection_confidence=float(os.getenv("DETECTION_CONFIDENCE", "0.45")),
        min_box_area_ratio=float(os.getenv("MIN_BOX_AREA_RATIO", "0.08")),
        stable_frames_required=int(os.getenv("STABLE_FRAMES_REQUIRED", "3")),
        cooldown_seconds=float(os.getenv("COOLDOWN_SECONDS", "12")),
        snapshot_path=os.getenv("SNAPSHOT_PATH", "snapshot.jpg"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        vision_model=os.getenv("VISION_MODEL", "qwen3.5:397b-cloud"),
        roast_model=os.getenv("ROAST_MODEL", "qwen3.5:397b-cloud"),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90")),
    )


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ensure_ollama_online(host: str, timeout: int) -> None:
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {host}. Start Ollama first."
        ) from exc


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    timeout: int,
    *,
    system: Optional[str] = None,
    images: Optional[list[str]] = None,
) -> str:
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images

    resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned an empty response for model '{model}'.")
    return text


def describe_person(image_path: str, cfg: Settings) -> str:
    image_b64 = encode_image_base64(image_path)
    return ollama_generate(
        cfg.ollama_host,
        cfg.vision_model,
        DESCRIPTION_PROMPT,
        cfg.request_timeout_seconds,
        images=[image_b64],
    )


def roast_person(description: str, cfg: Settings) -> str:
    user_prompt = f"Visual description:\n{description}\n\nNow roast this person."
    return ollama_generate(
        cfg.ollama_host,
        cfg.roast_model,
        user_prompt,
        cfg.request_timeout_seconds,
        system=ROAST_SYSTEM_PROMPT,
    )


def person_detected(frame, detector: YOLO, confidence: float, min_box_area_ratio: float) -> bool:
    results = detector.predict(source=frame, classes=[0], conf=confidence, verbose=False)
    if not results:
        return False

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False

    h, w = frame.shape[:2]
    frame_area = float(w * h)
    max_ratio = 0.0
    for xyxy in boxes.xyxy:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ratio = area / frame_area
        if ratio > max_ratio:
            max_ratio = ratio

    return max_ratio >= min_box_area_ratio


def open_camera(camera_index: int):
    # On Windows, DirectShow is often more stable than the default backend.
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def run_model_pipeline(snapshot_path: str, cfg: Settings, out_q: "queue.Queue[dict]") -> None:
    try:
        description = describe_person(snapshot_path, cfg)
        roast = roast_person(description, cfg)
        out_q.put({"description": description, "roast": roast})
    except Exception as exc:
        out_q.put({"error": str(exc)})


def run() -> None:
    cfg = read_settings()
    ensure_ollama_online(cfg.ollama_host, cfg.request_timeout_seconds)

    snapshot_file = Path(cfg.snapshot_path).resolve()
    detector = YOLO("yolov8n.pt")
    cap = open_camera(cfg.camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {cfg.camera_index}.")

    print("Camera ready. Show one person to trigger snapshot.")
    print("Press 'q' in the preview window to quit.")

    stable_count = 0
    person_present = False
    last_trigger_ts = 0.0
    last_roast = "No roast yet."
    pipeline_q: "queue.Queue[dict]" = queue.Queue()
    pipeline_thread: Optional[threading.Thread] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            if pipeline_thread and not pipeline_thread.is_alive():
                pipeline_thread = None

            while not pipeline_q.empty():
                result = pipeline_q.get_nowait()
                if "error" in result:
                    print(f"Model pipeline failed: {result['error']}")
                else:
                    description = result["description"]
                    roast = result["roast"]
                    last_roast = roast
                    print("\nDescription:\n")
                    print(description)
                    print("\nRoast:\n")
                    print(roast)
                    print("\n" + "-" * 60)

            detected = person_detected(
                frame, detector, cfg.detection_confidence, cfg.min_box_area_ratio
            )
            if detected:
                stable_count += 1
            else:
                stable_count = 0
                person_present = False

            now = time.time()
            trigger_ready = (
                not person_present
                and stable_count >= cfg.stable_frames_required
                and (now - last_trigger_ts) >= cfg.cooldown_seconds
                and pipeline_thread is None
            )

            if trigger_ready:
                person_present = True
                last_trigger_ts = now
                cv2.imwrite(str(snapshot_file), frame)
                print(f"\nSnapshot saved: {snapshot_file}")
                pipeline_thread = threading.Thread(
                    target=run_model_pipeline,
                    args=(str(snapshot_file), cfg, pipeline_q),
                    daemon=True,
                )
                pipeline_thread.start()

            # Small status overlay for live preview.
            model_status = "processing" if pipeline_thread is not None else "idle"
            status = (
                f"detected={detected} stable={stable_count}/{cfg.stable_frames_required} "
                f"cooldown={max(0, int(cfg.cooldown_seconds - (now - last_trigger_ts)))}s "
                f"model={model_status}"
            )
            cv2.putText(
                frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            roast_preview = (last_roast[:85] + "...") if len(last_roast) > 88 else last_roast
            cv2.putText(
                frame,
                roast_preview,
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Roast Cam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
