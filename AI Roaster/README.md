# Roast Cam (Ollama)

Webcam pipeline:
1. Person enters frame
2. Mode wheel spins (`normal`, `sports-commentary`, `mean-girl`, `corporate-review`)
3. Snapshot is captured
4. Vision model describes the person
5. Text model generates a mode-specific roast

## Requirements

- Python 3.10+
- Ollama running locally
- Webcam

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Pull required models in Ollama:

```bash
ollama pull qwen3-vl:4b
ollama pull llama3.2:3b
```

## Run

```bash
python main.py
```

## Configuration

Settings are loaded from `config.yaml` (not `.env`).
Edit these keys to customize behavior:
- `default_roast_mode`
- `wheel_spin_seconds`
- `history_path`
- `snapshot_path`
- `vision_model`, `roast_model`, and `ollama_host`

- Type `r` + Enter to spin the wheel and run a roast.
- Type `w` + Enter to spin the wheel only (set next mode).
- Type `q` + Enter or press `q` in the video window to quit.
- Snapshot is saved to the path in `config.yaml` under `snapshot_path`.
