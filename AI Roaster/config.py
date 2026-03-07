from dataclasses import dataclass
from pathlib import Path

import yaml

from prompts import normalize_roast_mode


@dataclass
class Settings:
    camera_index: int = 0
    snapshot_delay_seconds: float = 3.0
    snapshot_path: str = "snapshot.jpg"
    history_path: str = "history.txt"
    ollama_host: str = "http://127.0.0.1:11434"
    vision_model: str = "qwen3-vl:4b"
    roast_model: str = "llama3.2:3b"
    roast_temperature: float = 0.9
    request_timeout_seconds: int = 90
    ollama_keep_alive: str = "0s"
    default_roast_mode: str = "normal"
    wheel_spin_seconds: float = 2.6


def normalize_ollama_host(host: str) -> str:
    cleaned = host.strip().rstrip("/")
    if cleaned.endswith("/api"):
        cleaned = cleaned[:-4]
    return cleaned


def _read_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(data).__name__}")
    return data


def read_settings() -> Settings:
    data = _read_yaml_config(Path("config.yaml"))
    host = normalize_ollama_host(str(data.get("ollama_host", "http://127.0.0.1:11434")))
    return Settings(
        camera_index=int(data.get("camera_index", 0)),
        snapshot_delay_seconds=float(data.get("snapshot_delay_seconds", 3)),
        snapshot_path=str(data.get("snapshot_path", "snapshot.jpg")),
        history_path=str(data.get("history_path", "history.txt")),
        ollama_host=host,
        vision_model=str(data.get("vision_model", "qwen3-vl:4b")),
        roast_model=str(data.get("roast_model", "llama3.2:3b")),
        roast_temperature=float(data.get("roast_temperature", 0.9)),
        request_timeout_seconds=int(data.get("request_timeout_seconds", 90)),
        ollama_keep_alive=str(data.get("ollama_keep_alive", "0s")),
        default_roast_mode=normalize_roast_mode(str(data.get("default_roast_mode", "normal"))),
        wheel_spin_seconds=float(data.get("wheel_spin_seconds", 2.6)),
    )
