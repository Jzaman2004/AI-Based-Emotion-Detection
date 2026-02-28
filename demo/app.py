import base64
import io
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq
from PIL import Image, UnidentifiedImageError

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_MEGAPIXELS = 33_000_000
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

ALLOWED_GENDERS = {"male", "female", "unknown"}
ALLOWED_RACES = {
    "Caucasian White",
    "Black American",
    "South Asian",
    "East Asian",
    "Latino",
    "unknown",
}
EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "hate"]

SYSTEM_PROMPT = """You are an ethical facial analysis assistant. Analyze the provided facial image and output ONLY a valid JSON object with the following structure. Do not include any other text, explanations, or markdown.

{
  "person_visible": boolean,
  "gender": "male" | "female" | "unknown",
  "race": "Caucasian White" | "Black American" | "South Asian" | "East Asian" | "Latino" | "unknown",
  "emotions": {
    "joy": integer (0-100),
    "sadness": integer (0-100),
    "anger": integer (0-100),
    "fear": integer (0-100),
    "disgust": integer (0-100),
    "surprise": integer (0-100),
    "hate": integer (0-100)
  },
  "dominant_emotion": "joy" | "sadness" | "anger" | "fear" | "disgust" | "surprise" | "hate"
}

RULES:
1. All emotion percentages MUST sum to exactly 100.
2. If the face is unclear, occluded, or not visible, set person_visible=false and all other fields to "unknown" or 0.
3. For gender: ONLY output "male", "female", or "unknown". No other values.
4. For race: ONLY output one of: "Caucasian White", "Black American", "South Asian", "East Asian", "Latino", or "unknown". No other values.
5. dominant_emotion must be the key from "emotions" with the highest integer value. If tie, pick the first alphabetically.
6. If confidence is low, default to "unknown" rather than guessing.
7. Output ONLY the JSON object. No preamble, no code blocks, no explanations."""


def safe_default_result() -> dict[str, Any]:
    """Return a safe fallback result when parsing or model output fails."""
    return {
        "person_visible": False,
        "gender": "unknown",
        "race": "unknown",
        "emotions": {emotion: 0 for emotion in EMOTION_KEYS},
        "dominant_emotion": "joy",
    }


def infer_dominant_emotion(emotions: dict[str, int]) -> str:
    """Return the dominant emotion, using alphabetical tie-break for equal scores."""
    return min(
        sorted(EMOTION_KEYS),
        key=lambda emotion: (-emotions.get(emotion, 0), emotion),
    )


def normalize_emotions(raw: Any) -> dict[str, int]:
    """Coerce emotion values to valid integers in [0, 100] that sum to exactly 100."""
    normalized: dict[str, int] = {}
    for emotion in EMOTION_KEYS:
        value = raw.get(emotion, 0) if isinstance(raw, dict) else 0
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            value = 0
        normalized[emotion] = max(0, min(100, value))

    total = sum(normalized.values())
    if total == 100:
        return normalized

    if total <= 0:
        normalized["joy"] = 100
        for emotion in EMOTION_KEYS:
            if emotion != "joy":
                normalized[emotion] = 0
        return normalized

    scaled = {emotion: (normalized[emotion] * 100.0 / total) for emotion in EMOTION_KEYS}
    floors = {emotion: int(scaled[emotion]) for emotion in EMOTION_KEYS}
    remainder = 100 - sum(floors.values())

    if remainder > 0:
        order = sorted(EMOTION_KEYS, key=lambda key: (scaled[key] - floors[key]), reverse=True)
        for i in range(remainder):
            floors[order[i % len(order)]] += 1
    elif remainder < 0:
        order = sorted(EMOTION_KEYS, key=lambda key: (scaled[key] - floors[key]))
        for i in range(abs(remainder)):
            candidate = order[i % len(order)]
            if floors[candidate] > 0:
                floors[candidate] -= 1

    return floors


def sanitize_result(raw: Any) -> dict[str, Any]:
    """Validate schema and constraints, then return safe structured output."""
    fallback = safe_default_result()
    if not isinstance(raw, dict):
        return fallback

    person_visible = bool(raw.get("person_visible", False))
    if not person_visible:
        return fallback

    gender = raw.get("gender", "unknown")
    if gender not in ALLOWED_GENDERS:
        gender = "unknown"

    race = raw.get("race", "unknown")
    if race not in ALLOWED_RACES:
        race = "unknown"

    emotions = normalize_emotions(raw.get("emotions", {}))
    dominant_emotion = infer_dominant_emotion(emotions)

    return {
        "person_visible": True,
        "gender": gender,
        "race": race,
        "emotions": emotions,
        "dominant_emotion": dominant_emotion,
    }


def decode_and_validate_image() -> tuple[bytes, str]:
    """Read multipart/form-data image input (file or base64 string) and validate it."""
    image_bytes: bytes | None = None
    mime_type = "image/jpeg"

    upload = request.files.get("image")
    if upload:
        image_bytes = upload.read()
        mime_type = upload.mimetype or mime_type
    else:
        data = request.form.get("image", "").strip()
        if not data:
            raise ValueError("No image provided")

        if data.startswith("data:"):
            match = re.match(r"^data:(image/(?:jpeg|png));base64,(.+)$", data, re.IGNORECASE | re.DOTALL)
            if not match:
                raise ValueError("Unsupported data URL format")
            mime_type = match.group(1).lower()
            payload = match.group(2)
        else:
            payload = data

        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("Invalid base64 image") from exc

    if image_bytes is None or len(image_bytes) == 0:
        raise ValueError("Empty image")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise OverflowError("Image exceeds 4MB size limit")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            img.verify()
            if width * height > MAX_MEGAPIXELS:
                raise ValueError("Image exceeds 33MP resolution limit")
            image_format = (img.format or "").upper()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Unsupported image file") from exc

    if image_format == "JPEG":
        mime_type = "image/jpeg"
    elif image_format == "PNG":
        mime_type = "image/png"
    else:
        raise ValueError("Only JPEG and PNG images are allowed")

    return image_bytes, mime_type


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse model text response into a JSON object."""
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object")
    return parsed


def analyze_with_groq(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    """Send image to Groq vision model and return sanitized JSON output."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this facial image according to the system rules."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.1,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    message = response.choices[0].message.content if response.choices else ""
    raw_result = parse_json_object(message or "")
    return sanitize_result(raw_result)


@app.route("/")
def index() -> Any:
    """Serve the capture page."""
    return send_from_directory(".", "demo.html")


@app.route("/results")
def results() -> Any:
    """Serve the results page."""
    return send_from_directory(".", "demoresults.html")


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    """Accept an image, proxy analysis to Groq, and return safe structured output."""
    try:
        image_bytes, mime_type = decode_and_validate_image()
        result = analyze_with_groq(image_bytes, mime_type)
        return jsonify({"success": True, "data": result, "message": "Analysis complete"}), 200
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except OverflowError as exc:
        return jsonify({"success": False, "message": str(exc)}), 413
    except RuntimeError:
        app.logger.exception("Configuration error")
        return jsonify({"success": False, "message": "Server configuration error"}), 500
    except Exception:
        app.logger.exception("Unexpected analysis error")
        return jsonify({"success": False, "message": "Unable to analyze image right now"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
