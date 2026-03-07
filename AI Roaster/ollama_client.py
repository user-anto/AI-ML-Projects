import base64
from typing import Optional

import requests

from config import Settings
from prompts import DESCRIPTION_PROMPT, build_roast_system_prompt, mode_label, normalize_roast_mode


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ensure_ollama_online(host: str, timeout: int) -> set[str]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return {
            item["name"]
            for item in data.get("models", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {host}. Start Ollama (`ollama serve`) first."
        ) from exc


def ensure_models_available(available_models: set[str], required_models: list[str]) -> None:
    missing = sorted({m for m in required_models if m not in available_models})
    if not missing:
        return

    install_lines = "\n".join(f"  ollama pull {m}" for m in missing)
    raise RuntimeError(
        "Missing Ollama model(s): "
        + ", ".join(missing)
        + "\nInstall them with:\n"
        + install_lines
    )


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    timeout: int,
    *,
    system: Optional[str] = None,
    images: Optional[list[str]] = None,
    keep_alive: Optional[str] = None,
    options: Optional[dict] = None,
) -> str:
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images
    if keep_alive:
        payload["keep_alive"] = keep_alive
    if options:
        payload["options"] = options

    resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = str(body.get("error", "")).strip()
        except Exception:
            detail = resp.text.strip()

        if resp.status_code == 404 and "model" in detail.lower():
            detail = f"{detail} Install it with: ollama pull {model}"
        elif not detail:
            detail = resp.reason

        raise RuntimeError(
            f"Ollama generate failed for model '{model}' ({resp.status_code}): {detail}"
        ) from exc

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
        keep_alive=cfg.ollama_keep_alive,
    )


def roast_person(description: str, mode: str, cfg: Settings) -> str:
    mode_key = normalize_roast_mode(mode)
    user_prompt = (
        f"Roast mode: {mode_label(mode_key)}\n"
        f"Visual description:\n{description}\n\n"
        "Now roast this person."
    )
    return ollama_generate(
        cfg.ollama_host,
        cfg.roast_model,
        user_prompt,
        cfg.request_timeout_seconds,
        system=build_roast_system_prompt(mode_key),
        keep_alive=cfg.ollama_keep_alive,
        options={"temperature": cfg.roast_temperature},
    )
