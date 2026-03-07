import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from camera import open_camera
from config import Settings, read_settings
from dialog import show_text_dialog_async
from ollama_client import (
    describe_person,
    ensure_models_available,
    ensure_ollama_online,
    roast_person,
)
from prompts import mode_label
from ui import ModeWheel, draw_mode_wheel


def run_model_pipeline(
    snapshot_path: str, cfg: Settings, roast_mode: str, out_q: "queue.Queue[dict]"
) -> None:
    try:
        description = describe_person(snapshot_path, cfg)
        roast = roast_person(description, roast_mode, cfg)
        out_q.put({"description": description, "roast": roast, "mode": roast_mode})
    except Exception as exc:
        out_q.put({"error": str(exc)})


def stdin_listener(cmd_q: "queue.Queue[str]") -> None:
    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            cmd = "q"
        cmd_q.put(cmd)
        if cmd == "q":
            return


def append_history(history_file: Path, mode: str, description: str, roast: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with history_file.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Mode: {mode_label(mode)} ({mode})\n")
        f.write(f"Description:\n{description}\n")
        f.write(f"Roast:\n{roast}\n")
        f.write("-" * 60 + "\n\n")


def run() -> None:
    cfg = read_settings()
    available_models = ensure_ollama_online(cfg.ollama_host, cfg.request_timeout_seconds)
    ensure_models_available(available_models, [cfg.vision_model, cfg.roast_model])

    snapshot_file = Path(cfg.snapshot_path).resolve()
    history_file = Path(cfg.history_path).resolve()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    cap = open_camera(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {cfg.camera_index}.")

    wheel = ModeWheel(selected_mode=cfg.default_roast_mode)

    print("Camera ready.")
    print("Type 'r' + Enter to spin mode wheel and roast.")
    print("Type 'w' + Enter to spin wheel only.")
    print("Type 'q' + Enter or press 'q' in the preview window to quit.")
    print(f"Default mode: {mode_label(wheel.selected_mode)}")

    cmd_q: "queue.Queue[str]" = queue.Queue()
    input_thread = threading.Thread(target=stdin_listener, args=(cmd_q,), daemon=True)
    input_thread.start()

    pipeline_q: "queue.Queue[dict]" = queue.Queue()
    pipeline_thread: Optional[threading.Thread] = None

    countdown_end_ts = 0.0
    countdown_active = False
    pending_capture_after_spin = False
    last_roast = "No roast yet."
    last_roast_mode = wheel.selected_mode

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
                    mode = result["mode"]
                    last_roast_mode = mode
                    last_roast = roast
                    show_text_dialog_async("Roast Cam", roast)

                    append_history(history_file, mode, description, roast)

                print("Waiting for 'r' + Enter to spin and roast again.")

            while not cmd_q.empty():
                cmd = cmd_q.get_nowait().strip().lower()
                if cmd == "q":
                    return
                if cmd == "r":
                    if pipeline_thread is not None:
                        print("Still generating roast. Please wait.")
                    elif countdown_active:
                        print("Snapshot timer is already running.")
                    elif wheel.active:
                        print("Mode wheel is already spinning.")
                    else:
                        pending_capture_after_spin = True
                        wheel.spin(time.time(), cfg.wheel_spin_seconds)
                        print("Spinning mode wheel for next roast...")
                elif cmd == "w":
                    if countdown_active:
                        print("Cannot spin while snapshot timer is running.")
                    elif wheel.active:
                        print("Mode wheel is already spinning.")
                    else:
                        pending_capture_after_spin = False
                        wheel.spin(time.time(), cfg.wheel_spin_seconds)
                        print("Spinning mode wheel...")
                elif cmd:
                    print("Unknown command. Use 'r' to roast, 'w' to spin mode, 'q' to quit.")

            now = time.time()
            selected_mode = wheel.update(now)
            if selected_mode:
                print(f"Mode selected: {mode_label(selected_mode)} ({selected_mode})")
                if pending_capture_after_spin:
                    pending_capture_after_spin = False
                    countdown_active = True
                    countdown_end_ts = now + cfg.snapshot_delay_seconds
                    print(f"Snapshot timer started ({cfg.snapshot_delay_seconds:.0f}s).")

            if countdown_active and pipeline_thread is None and not wheel.active:
                now = time.time()
                if now >= countdown_end_ts:
                    cv2.imwrite(str(snapshot_file), frame)
                    countdown_active = False
                    active_mode = wheel.selected_mode
                    print(f"\nSnapshot saved: {snapshot_file}")
                    print(f"Generating roast in mode: {mode_label(active_mode)}")
                    pipeline_thread = threading.Thread(
                        target=run_model_pipeline,
                        args=(str(snapshot_file), cfg, active_mode, pipeline_q),
                        daemon=True,
                    )
                    pipeline_thread.start()
                else:
                    remaining = max(0.0, countdown_end_ts - now)
                    cv2.putText(
                        frame,
                        f"Snapshot in {remaining:.1f}s",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )

            model_status = "processing" if pipeline_thread is not None else "idle"
            countdown_text = "countdown=active" if countdown_active else "countdown=idle"
            status = (
                f"{countdown_text} model={model_status} mode={mode_label(wheel.selected_mode)}"
            )
            cv2.putText(
                frame,
                status,
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            hint = "stdin: r spin+roast | w spin mode | q quit"
            cv2.putText(
                frame,
                hint,
                (12, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

            roast_preview = f"[{mode_label(last_roast_mode)}] {last_roast}"
            roast_preview = (
                roast_preview[:95] + "..." if len(roast_preview) > 98 else roast_preview
            )
            cv2.putText(
                frame,
                roast_preview,
                (12, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

            draw_mode_wheel(frame, wheel)

            cv2.imshow("Roast Cam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
