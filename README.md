# <img src="https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugar_crepe.png?raw=true" height="50"> SugarCrepe: A benchmark for faithful vision-language compositionality evaluation

This repository contains the official SugarCrepe benchmark plus an additional local pipeline for generating SugarCrepe-style hard negatives from MS COCO with `Ollama + qwen3-vl:8b`.

## 프로젝트 요약

### 이 프로젝트가 다루는 핵심 문제

CLIP 같은 비전-언어 모델을 평가하는 기존 벤치마크(CREPE, ARO 등)에는 두 가지 치명적 편향이 있었습니다.

- **플러시빌리티(Plausibility) 편향**: 오답 문장이 너무 어색해서 이미지 없이 텍스트만 봐도 오답을 골라낼 수 있음
- **문법(Grammar) 편향**: 오답 문장이 문법적으로 어색해서 언어 모델만으로도 쉽게 구분 가능

이로 인해 모델이 실제 구성적 이해(compositionality)를 하지 않아도 높은 점수를 받을 수 있었습니다.

### SugarCrepe의 해결책

ChatGPT + 인간 검수 + 적대적 정제(Adversarial Refinement) 파이프라인으로 고품질 하드 네거티브를 생성합니다. 오답 문장도 자연스럽고 문법적으로 올바르게 만들어 오직 이미지와의 관계를 실제로 이해해야만 정답을 고를 수 있도록 설계했습니다.

### 7가지 평가 유형

| 유형 | 설명 |
|---|---|
| `add_obj` | 없는 객체를 추가 |
| `add_att` | 없는 속성을 추가 |
| `replace_obj` | 객체를 다른 것으로 치환 |
| `replace_att` | 속성을 다른 것으로 치환 |
| `replace_rel` | 관계를 다른 것으로 치환 |
| `swap_obj` | 두 객체의 위치/역할을 교환 |
| `swap_att` | 두 속성을 교환 |

## 코드 구조

실행용 스크립트와 재사용 로직을 분리했습니다.

### `src/cli`

연구원이 직접 실행하는 엔트리포인트입니다.

| 파일 | 역할 |
|---|---|
| `src/cli/main_eval.py` | OpenCLIP 기반 이미지-텍스트 유사도 평가 |
| `src/cli/text_model_eval.py` | 텍스트만으로 편향 분석 (Vera, Grammar 모델) |
| `src/cli/inspect_mscoco_sample.py` | COCO 이미지 파일명과 캡션 확인 |
| `src/cli/build_coco_caption_groups.py` | COCO captions를 이미지 단위 JSONL로 묶기 |
| `src/cli/generate_hard_negatives.py` | `qwen3-vl:8b`로 hard negative 후보 생성 |
| `src/cli/filter_hard_negatives.py` | Qwen judge + text bias score로 후보 필터링 |
| `src/cli/run_adversarial_refinement.py` | 합성 데이터셋 경로를 받아 적대적 정제 및 SugarCrepe 형식 export |
| `src/cli/export_sugarcrepe_style.py` | 필터된 JSONL을 SugarCrepe 형식 JSON으로 export |

### `src/lib`

클래스, 메서드, 공통 유틸이 들어 있는 라이브러리 코드입니다.

| 파일 | 역할 |
|---|---|
| `src/lib/hard_negative_pipeline.py` | Ollama 호출, 프롬프트, JSONL 입출력, 파일 네이밍 유틸 |
| `src/lib/adversarial_refine.py` | 적대적 정제 알고리즘 |
| `src/lib/mscoco_dataloader.py` | COCO 이미지/캡션 Dataset 및 DataLoader |

### 기타

| 파일 | 역할 |
|---|---|
| `scripts/setup_mscoco.sh` | MS COCO 2017 validation 이미지와 캡션 어노테이션 다운로드 |
| `data/*.json` | 정제 완료된 원본 SugarCrepe 데이터 |
| `data_unrefined/*.json` | 적대적 정제 이전 데이터 |

## 데이터셋 재현 가능 여부

원 논문 기준의 SugarCrepe 데이터셋은 이 저장소만으로 완전히 재현되지 않습니다. 다만 이 저장소에는 로컬에서 MS COCO 기반 합성 hard negative를 생성하고 적대적 정제까지 수행할 수 있는 추가 파이프라인이 포함되어 있습니다.

원 논문 파이프라인은 아래와 같습니다.

```text
[1] COCO 2017 val captions
        ↓
[2] ChatGPT로 하드 네거티브 생성
        ↓
[3] 인간 검수 (Human Validation)
        ↓
[4] 적대적 정제 (Adversarial Refinement)
        ↓
    최종 SugarCrepe
```

현재 레포에 추가된 로컬 파이프라인은 아래 흐름입니다.

```text
[1] COCO 2017 val captions
        ↓
[2] qwen3-vl:8b 로 hard negative 후보 생성
        ↓
[3] qwen3-vl:8b judge + text bias scoring
        ↓
[4] adversarial_refine 적용
        ↓
    SugarCrepe-style JSON export
```

## Installation

This project uses `torch`, `open_clip`, `transformers`, `Pillow`, and `tqdm`.

```bash
pip install open_clip_torch
```

If you are using the local hard-negative pipeline, make sure:

- `ollama` is installed
- `qwen3-vl:8b` is pulled
- MS COCO `val2017` images and `captions_val2017.json` are available under `mscoco/`

Check the model:

```bash
ollama list
```

Download COCO files:

```bash
bash scripts/setup_mscoco.sh
```

## 기본 평가 사용법

Evaluate a pretrained CLIP model:

```bash
python src/cli/main_eval.py --model RN50 \
    --pretrained openai \
    --output ./output \
    --coco_image_root ./mscoco/val2017 \
    --data_root ./data
```

Evaluate all pretrained CLIP models:

```bash
python src/cli/main_eval.py --all \
    --output ./output \
    --coco_image_root ./mscoco/val2017 \
    --data_root ./data
```

Evaluate text-only bias models:

```bash
python src/cli/text_model_eval.py \
    --output ./output \
    --data_root ./data
```

Inspect a sample COCO image-caption mapping:

```bash
python src/cli/inspect_mscoco_sample.py --show-captions --limit 3
```

## Hard Negative 생성 파이프라인

### 1. COCO captions를 이미지 단위로 묶기

```bash
python src/cli/build_coco_caption_groups.py \
    --root ./mscoco \
    --output artifacts/grouped_coco_captions.jsonl
```

출력 row 예시:

```json
{
  "image_id": 397133,
  "filename": "000000397133.jpg",
  "image_path": "mscoco/val2017/000000397133.jpg",
  "all_captions": [
    "A man is in a kitchen making pizzas.",
    "Man in apron standing on front of oven with pans and bakeware"
  ]
}
```

### 2. `qwen3-vl:8b`로 hard negative 후보 생성

```bash
python src/cli/generate_hard_negatives.py \
    --input artifacts/grouped_coco_captions.jsonl \
    --model qwen3-vl:8b
```

기본 출력은 자동 네이밍 규칙을 따릅니다.

```text
artifacts/generated_hard_negatives/YYYY-MM-DD/HHMMSS_qwen3-vl_8b_generated_hard_negatives.jsonl
```

직접 경로를 지정하고 싶으면 `--output`을 사용하면 됩니다.

### 3. 후보 필터링 및 점수화

```bash
python src/cli/filter_hard_negatives.py \
    --input artifacts/generated_hard_negatives/2026-04-05/214112_qwen3-vl_8b_generated_hard_negatives.jsonl \
    --model qwen3-vl:8b
```

기본 출력:

```text
artifacts/scored_hard_negatives/YYYY-MM-DD/HHMMSS_qwen3-vl_8b_scored_hard_negatives.jsonl
```

필터 단계는 아래를 수행합니다.

- single-edit semantic judge
- image-grounded faithfulness judge
- same-image caption consistency judge
- Vera / Grammar 기반 text-only bias score 계산

이미지가 손상되었거나 모델이 이미지를 받지 못하는 경우, 해당 샘플은 `stderr` 로그를 남기고 스킵합니다.

### 4. 적대적 정제 + SugarCrepe 형식 export

생성된 합성 데이터셋 경로를 직접 넘겨서 실행할 수 있습니다.

```bash
python src/cli/run_adversarial_refinement.py \
    --synthetic-dataset artifacts/scored_hard_negatives/2026-04-05/214200_qwen3-vl_8b_scored_hard_negatives.jsonl
```

기본 출력:

```text
artifacts/refined_sugarcrepe/YYYY-MM-DD/HHMMSS_qwen3-vl_8b_refined_sugarcrepe/
```

이 폴더 안에는 아래와 같은 파일이 생성됩니다.

- `add_obj.json`
- `add_att.json`
- `replace_obj.json`
- `replace_att.json`
- `replace_rel.json`
- `swap_obj.json`
- `swap_att.json`

출력 형식은 기존 SugarCrepe 평가 코드와 호환됩니다.

```json
{
  "0": {
    "filename": "000000397133.jpg",
    "caption": "A man is in a kitchen making pizzas.",
    "negative_caption": "A man is in a kitchen making cakes."
  }
}
```

## 파일 네이밍 규칙

모델에 따라 생성되는 hard negative가 달라질 수 있으므로, 생성물은 날짜별 폴더와 시간 기반 파일명으로 저장됩니다.

- 생성 결과:
  `HHMMSS_<model>_generated_hard_negatives.jsonl`
- 필터 결과:
  `HHMMSS_<model>_scored_hard_negatives.jsonl`
- 적대적 정제 결과:
  `HHMMSS_<model>_refined_sugarcrepe/`

예:

```text
artifacts/generated_hard_negatives/2026-04-05/214112_qwen3-vl_8b_generated_hard_negatives.jsonl
artifacts/scored_hard_negatives/2026-04-05/214130_qwen3-vl_8b_scored_hard_negatives.jsonl
artifacts/refined_sugarcrepe/2026-04-05/214235_qwen3-vl_8b_refined_sugarcrepe/
```

## 운영 시 참고 사항

- COCO는 이미지당 5개 캡션을 제공하므로, 생성은 5개 캡션 전체를 컨텍스트로 사용합니다.
- 최종 샘플 단위는 `이미지 1장 + positive caption 1개 + negative caption 1개`입니다.
- 필터와 적대적 정제를 거친 뒤 이미지당 최종 1개만 남기도록 구성되어 있습니다.
- `instances_val2017.json`은 현재 v1 파이프라인에서 필수는 아니며, 향후 객체 검증 보조용으로 확장 가능합니다.

---

**GPT-4V on SugarCrepe is [here](https://github.com/RAIVNLab/sugar-crepe/tree/main/gpt-4v-results)**

This is the official repository of SugarCrepe, a benchmark for faithful vision-language compositionality evaluation introduced in our paper [SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality](https://arxiv.org/abs/2306.14610).

On SugarCrepe, given an image, a model is required to select the positive caption that correctly describes the image, against another hard negative text distractor that differs from the positive text only by small compositional changes.

❗:**You can also use SugarCrepe in the [clip-benchmark](https://github.com/LAION-AI/CLIP_benchmark#compositionality-evaluation)**

## Why SugarCrepe?

### Biases in existing benchmarks

Many existing benchmarks contain artifacts in hard negatives that can be easily exploited to achieve high performances. These artifacts, that the hard negatives are "not plausible" and "non-fluent", render the benchmarks unreliable for compositionality evaluation: Blind models, a plausibility estimation model (Vera) and a grammar-scoring model, can outperform state-of-the-art CLIP models on nearly all of these benchmarks.

![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/existing_eval.png?raw=true)

### SugarCrepe removes the biases

In SugarCrepe, we remove the artifacts by leveraging ChatGPT to generate plausible and fluent hard negatives, followed by human validation and an adversarial refinement mechanism to maximally reduce the identified biases. We show some comparisons between SugarCrepe and other benchmarks:

![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugarcrepe_vs_existing.png?raw=true)

## What we found

### Re-evaluating NegCLIP

We find that NegCLIP's improvements on existing benchmarks, e.g., ARO and CREPE, are overestimated, where its improvements are much smaller on SugarCrepe. The overestimation is particularly large when the test hard negative type matches the one used in training, which we attribute to models' unintentionally overfitting to the artifacts.

![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/re_eval.png?raw=true)

The models we trained can be found [here](https://drive.google.com/drive/folders/1n2ZNldxBteltuqx__id43sQCyzq0of09?usp=drive_link).

### Benchmarking pretrained CLIP models

On SugarCrepe, we benchmark 17 pretrained CLIP models and present 4 findings:

- The best pretrained CLIP models demonstrate some compositional understanding but still have overall large rooms for improvements.
- All models struggle at identifying SWAP hard negatives, regardless of their pertaining dataset and model size.
- Existing models are object-centric, struggling to compose attributes and relations.
- Models’ performance on SugarCrepe correlates with their ImageNet zero-shot accuracy.

![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugarcrepe_eval.png?raw=true)

## Cite

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{hsieh2023sugarcrepe,
  title={SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality},
  author={Hsieh, Cheng-Yu and Zhang, Jieyu and Ma, Zixian and Kembhavi, Aniruddha and Krishna, Ranjay},
  booktitle={Thirty-Seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```
