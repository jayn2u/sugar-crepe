#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-${SCRIPT_DIR}/mscoco}"
IMAGE_DIR="${TARGET_ROOT}/val2017"
ANNOTATION_DIR="${TARGET_ROOT}/annotations"
IMAGE_ZIP_PATH="${TARGET_ROOT}/val2017.zip"
ANNOTATION_ZIP_PATH="${TARGET_ROOT}/annotations_trainval2017.zip"
IMAGE_URL="http://images.cocodataset.org/zips/val2017.zip"
ANNOTATION_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

die() {
  echo "Error: $*" >&2
  exit 1
}

log() {
  echo "[setup_mscoco] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar "$url" -o "$out"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
    return
  fi

  die "Neither curl nor wget is available for downloading files"
}

require_cmd unzip

if [[ -d "${TARGET_ROOT}" ]] && [[ -n "$(find "${TARGET_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  die "Target directory already contains files: ${TARGET_ROOT}. Remove or rename it before running again."
fi

mkdir -p "${TARGET_ROOT}"

if [[ -e "${IMAGE_ZIP_PATH}" || -e "${ANNOTATION_ZIP_PATH}" || -e "${IMAGE_DIR}" || -e "${ANNOTATION_DIR}" ]]; then
  die "Found existing download artifacts in ${TARGET_ROOT}. Refusing to continue."
fi

log "Downloading MS COCO 2017 validation images into ${TARGET_ROOT}"
download_file "${IMAGE_URL}" "${IMAGE_ZIP_PATH}"

log "Downloading MS COCO 2017 annotations into ${TARGET_ROOT}"
download_file "${ANNOTATION_URL}" "${ANNOTATION_ZIP_PATH}"

log "Extracting ${IMAGE_ZIP_PATH}"
unzip -q "${IMAGE_ZIP_PATH}" -d "${TARGET_ROOT}"

log "Extracting ${ANNOTATION_ZIP_PATH}"
unzip -q "${ANNOTATION_ZIP_PATH}" -d "${TARGET_ROOT}"

if [[ ! -d "${IMAGE_DIR}" ]]; then
  die "Extraction completed, but ${IMAGE_DIR} was not created"
fi

if [[ ! -f "${ANNOTATION_DIR}/captions_val2017.json" ]]; then
  die "Extraction completed, but ${ANNOTATION_DIR}/captions_val2017.json was not created"
fi

IMAGE_COUNT="$(find "${IMAGE_DIR}" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')"

if [[ "${IMAGE_COUNT}" -eq 0 ]]; then
  die "No JPG files were found in ${IMAGE_DIR}"
fi

CAPTION_COUNT="$(ANNOTATION_FILE="${ANNOTATION_DIR}/captions_val2017.json" python - <<'PY'
import json
import os

with open(os.environ["ANNOTATION_FILE"], "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data.get("annotations", [])))
PY
)"

log "Extraction complete: ${IMAGE_COUNT} images available under ${IMAGE_DIR}"
log "Captions ready: ${CAPTION_COUNT} caption annotations available under ${ANNOTATION_DIR}"
log "You can now pass --coco_image_root ${IMAGE_DIR} to main_eval.py"
