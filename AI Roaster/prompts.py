import textwrap

ROAST_MODES = [
    "normal",
    "sports-commentary",
    "mean-girl",
    "corporate-review",
]

ROAST_MODE_LABELS = {
    "normal": "Normal",
    "sports-commentary": "Sports Commentary",
    "mean-girl": "Mean Girl",
    "corporate-review": "Corporate Review",
}

MODE_STYLE_GUIDANCE = {
    "normal": "Sharp, clever one-liners with a playful sting.",
    "sports-commentary": "High-energy sports caster voice with scoreboard-style punchlines.",
    "mean-girl": "Snarky, trendy, queen-bee gossip tone with playful shade.",
    "corporate-review": "Performance-review tone, passive-aggressive but funny.",
}

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


def normalize_roast_mode(mode: str) -> str:
    if not isinstance(mode, str):
        return "normal"

    cleaned = mode.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "sports": "sports-commentary",
        "sports-commentator": "sports-commentary",
        "mean": "mean-girl",
        "meangirl": "mean-girl",
        "corp": "corporate-review",
        "corporate": "corporate-review",
    }
    canonical = aliases.get(cleaned, cleaned)
    if canonical in ROAST_MODES:
        return canonical
    return "normal"


def mode_label(mode: str) -> str:
    return ROAST_MODE_LABELS[normalize_roast_mode(mode)]


def build_roast_system_prompt(mode: str) -> str:
    mode_key = normalize_roast_mode(mode)
    return textwrap.dedent(
        f"""
        You are a sharp but safe roast comedian.
        Mode: {ROAST_MODE_LABELS[mode_key]}
        Style guidance: {MODE_STYLE_GUIDANCE[mode_key]}

        Rules:
        - Keep it funny, witty, and playful.
        - Roast only what is visible in the provided visual description.
        - Never target protected characteristics or sensitive traits.
        - No slurs, threats, or explicit sexual content.
        - Generate 3-4 short lines.
        """
    ).strip()
