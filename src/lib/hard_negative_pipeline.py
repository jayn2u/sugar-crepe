from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


EDIT_TYPES = (
    "add_obj",
    "add_att",
    "replace_obj",
    "replace_att",
    "replace_rel",
    "swap_obj",
    "swap_att",
)

SUPPORTED_VLM_MODELS = (
    "qwen3-vl:8b",
    "gemma4:26b",
    "gemma3:27b",
)


@dataclass(slots=True)
class OllamaConfig:
    model: str = "qwen3-vl:8b"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    timeout: int = 180
    retries: int = 3


@dataclass(slots=True, frozen=True)
class ModelProfile:
    model_name: str
    prompt_version: str
    family: str
    system_style: str


def resolve_model_profile(model_name: str) -> ModelProfile:
    if model_name == "qwen3-vl:8b":
        return ModelProfile(
            model_name=model_name,
            prompt_version="qwen3_vl_sugarcrepe_v1",
            family="qwen3_vl",
            system_style="strict_json_minimal_edit",
        )
    if model_name == "gemma4:26b":
        return ModelProfile(
            model_name=model_name,
            prompt_version="gemma4_sugarcrepe_v1",
            family="gemma4",
            system_style="strict_json_grounded_edit",
        )
    if model_name == "gemma3:27b":
        return ModelProfile(
            model_name=model_name,
            prompt_version="gemma3_sugarcrepe_v1",
            family="gemma3",
            system_style="strict_json_grounded_edit",
        )
    raise ValueError(
        f"Unsupported model '{model_name}'. Supported models: {', '.join(SUPPORTED_VLM_MODELS)}"
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._-") or "unknown"


def build_dated_output_path(
    root_dir: str | Path,
    *,
    model_name: str,
    suffix: str,
    extension: str,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    time_prefix = now.strftime("%H%M%S")
    model_part = sanitize_name(model_name)
    file_name = f"{time_prefix}_{model_part}_{suffix}.{extension.lstrip('.')}"
    return Path(root_dir) / date_dir / file_name


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_coco_captions(
    annotation_file: str | Path,
    image_root: str | Path,
    instances_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    annotation_data = read_json(annotation_file)
    images = annotation_data.get("images", [])
    annotations = annotation_data.get("annotations", [])

    captions_by_image_id: dict[int, list[str]] = {}
    for annotation in annotations:
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")
        if image_id is None or caption is None:
            continue
        captions_by_image_id.setdefault(image_id, []).append(caption)

    grouped_rows: list[dict[str, Any]] = []
    image_root = Path(image_root)
    for image in images:
        image_id = image.get("id")
        file_name = image.get("file_name")
        if image_id is None or file_name is None:
            continue
        captions = captions_by_image_id.get(image_id, [])
        if not captions:
            continue
        grouped_rows.append(
            {
                "image_id": image_id,
                "filename": file_name,
                "image_path": str(image_root / file_name),
                "all_captions": captions,
                "instances_file": str(instances_file) if instances_file else None,
            }
        )
    return grouped_rows


def encode_image_base64(image_path: str | Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def validate_image_path(image_path: str | Path) -> None:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"image path is not a file: {path}")

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"invalid or unreadable image: {path}") from exc


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc


def chat_with_ollama(
    config: OllamaConfig,
    system_prompt: str,
    user_prompt: str,
    image_path: str | Path | None = None,
    response_format: str | None = "json",
) -> str:
    user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
    if image_path is not None:
        user_message["images"] = [encode_image_base64(image_path)]

    payload = {
        "model": config.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            user_message,
        ],
        "options": {"temperature": config.temperature},
    }
    if response_format == "json":
        payload["format"] = "json"

    last_error: Exception | None = None
    for _ in range(config.retries):
        try:
            data = _post_json(f"{config.host}/api/chat", payload, timeout=config.timeout)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Ollama request failed after {config.retries} attempts") from last_error
    try:
        return data["message"]["content"]
    except KeyError as exc:
        raise RuntimeError(f"Unexpected Ollama response: {data}") from exc


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model response")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    markdown_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", stripped, flags=re.DOTALL)
    if markdown_match:
        return json.loads(markdown_match.group(1))

    start = min([idx for idx in (stripped.find("{"), stripped.find("[")) if idx != -1], default=-1)
    if start == -1:
        raise ValueError(f"no JSON object found in response: {text[:200]}")

    candidate = stripped[start:]
    for end in range(len(candidate), 0, -1):
        try:
            return json.loads(candidate[:end])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"failed to parse JSON from response: {text[:200]}")


def normalize_caption(text: str) -> str:
    return " ".join(text.strip().split())


def token_edit_ratio(source: str, target: str) -> float:
    return 1.0 - SequenceMatcher(None, normalize_caption(source), normalize_caption(target)).ratio()


def length_delta(source: str, target: str) -> int:
    return abs(len(normalize_caption(source).split()) - len(normalize_caption(target).split()))


def generation_system_prompt(model_profile: ModelProfile) -> str:
    if model_profile.family == "qwen3_vl":
        return (
            "You create SugarCrepe-style hard negative captions for vision-language evaluation. "
            "Return only valid JSON. Keep captions natural, concise, and close to the input style."
        )
    if model_profile.family in {"gemma4", "gemma3"}:
        return (
            "You are generating image-grounded SugarCrepe-style hard negative captions. "
            "Return valid JSON only. Keep the rewrite minimal, natural, and visually grounded."
        )
    raise ValueError(f"Unsupported model family: {model_profile.family}")


def build_generation_prompt(
    *,
    record: dict[str, Any],
    anchor_caption: str,
    anchor_index: int,
    edit_type: str,
    num_candidates: int,
) -> str:
    captions = "\n".join(
        f"{idx + 1}. {caption}"
        for idx, caption in enumerate(record["all_captions"])
    )
    return f"""
Image filename: {record["filename"]}
Anchor caption index: {anchor_index}
Anchor caption: {anchor_caption}
All human captions for the same image:
{captions}

Task: generate {num_candidates} hard negative captions for edit_type="{edit_type}".

Rules:
- Keep the sentence fluent and natural.
- Keep the style and length close to the anchor caption.
- Change exactly one semantic fact.
- The edited caption must be false for the image.
- The edited caption must not become true under any of the other human captions.
- Do not mention uncertainty or the editing process.

Allowed edit semantics:
- add_obj: add one object that is not in the image
- add_att: add one attribute that is not true
- replace_obj: replace one key object with another object
- replace_att: replace one attribute with another attribute
- replace_rel: replace one relation with another relation
- swap_obj: swap two object roles or positions
- swap_att: swap two attributes

Return JSON with this exact shape:
{{
  "candidates": [
    {{
      "negative_caption": "...",
      "edited_fact": "...",
      "why_false_for_image": "..."
    }}
  ]
}}
""".strip()


def judge_system_prompt(model_profile: ModelProfile) -> str:
    if model_profile.family == "qwen3_vl":
        return (
            "You are a strict verifier for SugarCrepe-style hard negative captions. "
            "Return only valid JSON."
        )
    if model_profile.family in {"gemma4", "gemma3"}:
        return (
            "You are a strict multimodal verifier for SugarCrepe-style hard negative captions. "
            "Return valid JSON only and make grounded decisions from the image."
        )
    raise ValueError(f"Unsupported model family: {model_profile.family}")


def build_semantic_judge_prompt(candidate: dict[str, Any]) -> str:
    return f"""
Positive caption: {candidate["anchor_caption"]}
Negative caption: {candidate["negative_caption"]}
Edit type: {candidate["edit_type"]}

Decide whether the negative caption is a minimal edit of the positive caption
and changes exactly one semantic fact consistent with the edit type.

Return JSON:
{{
  "passes_single_edit": true,
  "reason": "...",
  "edit_distance_ok": true
}}
""".strip()


def build_faithfulness_prompt(candidate: dict[str, Any]) -> str:
    all_captions = "\n".join(f"{idx + 1}. {caption}" for idx, caption in enumerate(candidate["all_captions"]))
    return f"""
All human captions for the image:
{all_captions}

Positive caption: {candidate["anchor_caption"]}
Negative caption: {candidate["negative_caption"]}

Estimate how true each caption is for the image on a 0 to 1 scale.
The positive caption should be scored by image evidence, not by whether it is human-written.

Return JSON:
{{
  "positive_truth": 0.0,
  "negative_truth": 0.0,
  "reason": "..."
}}
""".strip()


def build_consistency_prompt(candidate: dict[str, Any]) -> str:
    all_captions = "\n".join(f"{idx + 1}. {caption}" for idx, caption in enumerate(candidate["all_captions"]))
    return f"""
Human captions for the same image:
{all_captions}

Candidate negative caption: {candidate["negative_caption"]}

Decide whether the candidate negative caption could still be compatible with
any of the human captions for the same image.

Return JSON:
{{
  "compatible_with_any_caption": false,
  "reason": "..."
}}
""".strip()


def structural_filter(
    positive_caption: str,
    negative_caption: str,
    *,
    max_edit_ratio: float,
    max_length_delta: int,
) -> tuple[bool, str]:
    positive = normalize_caption(positive_caption)
    negative = normalize_caption(negative_caption)
    if not negative:
        return False, "empty_negative"
    if positive == negative:
        return False, "identical_caption"
    edit_ratio = token_edit_ratio(positive, negative)
    if edit_ratio > max_edit_ratio:
        return False, "edit_ratio_too_large"
    if length_delta(positive, negative) > max_length_delta:
        return False, "length_delta_too_large"
    return True, "ok"


def score_priority_key(candidate: dict[str, Any]) -> tuple[float, int, float, int]:
    scores = candidate["scores"]
    margin = float(scores["faithful_positive"]) - float(scores["faithful_negative"])
    length_gap = int(candidate.get("selection_meta", {}).get("length_delta", 0))
    edit_ratio = float(candidate.get("selection_meta", {}).get("edit_ratio", 0.0))
    rank = int(candidate.get("generation_meta", {}).get("candidate_rank", 999999))
    return (margin, -length_gap, -edit_ratio, -rank)
