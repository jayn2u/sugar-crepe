# <img src="https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugar_crepe.png?raw=true" height="50"> SugarCrepe: A benchmark for faithful vision-language compositionality evaluation

## 프로젝트 분석 요약

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

### 코드 구성

| 파일 | 역할 |
|---|---|
| `main_eval.py` | OpenCLIP 기반 이미지-텍스트 유사도 비교 평가 |
| `text_model_eval.py` | 이미지 없이 텍스트만으로 편향 분석 (Vera, Grammar 모델) |
| `adversarial_refine.py` | 적대적 정제 알고리즘 (논문의 정제 절차 구현) |
| `data/*.json` | 정제 완료된 최종 벤치마크 데이터 |
| `data_unrefined/*.json` | 적대적 정제 이전 데이터 (비교용) |

### 데이터셋 재현 가능 여부

이 저장소의 코드만으로는 데이터셋을 처음부터 재현할 수 없습니다. 데이터 생성 파이프라인은 아래와 같으며, 1-2단계의 코드가 공개되지 않았습니다.

```
[1] COCO 2017 val 캡션
        ↓  (ChatGPT 프롬프팅 코드 미공개)
[2] ChatGPT로 하드 네거티브 생성
        ↓  (인간 작업, 코드 없음)
[3] 인간 검수 (Human Validation)
        ↓  (adversarial_refine.py 에 알고리즘 존재, 단 파이프라인 미연결)
[4] 적대적 정제 (Adversarial Refinement)
        ↓
    최종 data/*.json
```

이 저장소는 데이터셋을 생성하는 저장소가 아니라 **이미 만들어진 데이터셋으로 모델을 평가하는** 저장소입니다.

---

**GPT-4V on SugarCrepe is [here](https://github.com/RAIVNLab/sugar-crepe/tree/main/gpt-4v-results)**

This is the official repository of SugarCrepe, a benchmark for faithful vision-language compositionality evaluation introduced in our paper [SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality](https://arxiv.org/abs/2306.14610).

On SugarCrepe, given an image, a model is required to select the positive caption that correctly describes the image, against another hard negative text distractor that differs from the positive text only by small compositional changes.

## :wrench: Installation

We use [open_clip](https://github.com/mlfoundations/open_clip) for loading pretrained models. Install it with:
```
pip install open_clip_torch
```

We use images of [COCO-2017](https://cocodataset.org/#download) validation set. Download and extract it to anywhere you want, for example, `data/coco/images/val2017/`.


## :keyboard: Usage

Evaluate a pretrained model using the following command:
```python
python main_eval.py --model RN50 \ 
    --pretrained openai \
    --output ./output \ 
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/ \
```

To evaluate the 17 pretrained CLIP models included in the paper, run:
```python
python main_eval.py --all
    --output ./output \ 
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/ \
```

To evaluate the text models included in the paper ([Vera](https://huggingface.co/liujch1998/vera) & Grammar model), run:
```python
python text_model_eval.py
    --output ./output \ 
    --data_root ./data/ \
```

❗:**You can also use SugarCrepe in the [clip-benchmark](https://github.com/LAION-AI/CLIP_benchmark#compositionality-evaluation)**

## :open_book: Why SugarCrepe?

### Biases in existing benchmarks
Many existing benchmarks contain artifacts in hard negatives that can be easily exploited to achieve high performances.
These artifacts, that the hard negatives are "not plausible" and "non-fluent", render the benchmarks unreliable for compositionality evaluation: Blind models, a plausibility estimation model (Vera) and a grammar-scoring model, can outperform state-of-the-art CLIP models on nearly all of these benchmarks.
![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/existing_eval.png?raw=true)


### SugarCrepe removes the biases
In SugarCrepe, we remove the artifacts by leveraging ChatGPT to generate plausible and fluent hard negatives, followed by human validation and an adversarial refinement mechanism to maximally reduce the identified biases. We show some comparisons between SugarCrepe and other benchmarks:
![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugarcrepe_vs_existing.png?raw=true)


## :bulb: What we found

### Re-evaluating NegCLIP
We find that NegCLIP's improvements on existing benchmarks, e.g., ARO and CREPE, are overestimated, where its improvements are much smaller on SugarCrepe.
The overestimation is particularly large when the test hard negative type matches the one used in training, which we attribute to models' unintentionally overfitting to the artifacts.
![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/re_eval.png?raw=true)

The models we trained can be found [here](https://drive.google.com/drive/folders/1n2ZNldxBteltuqx__id43sQCyzq0of09?usp=drive_link).

### Benchmarking pretrained CLIP models
On SugarCrepe, we benchmark 17 pretrained CLIP models and present 4 findings:
- The best pretrained CLIP models demonstrate some compositional understanding but still
have overall large rooms for improvements.
- All models struggle at identifying SWAP hard negatives, regardless of their pertaining dataset
and model size.
- Existing models are object-centric, struggling to compose attributes and relations.
- Models’ performance on SugarCrepe correlates with their ImageNet zero-shot accuracy.
  
![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugarcrepe_eval.png?raw=true)


## :paperclip: Cite
If you find this repository useful, please consider citing:
```bibtex
@inproceedings{hsieh2023sugarcrepe,
  title={SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality},
  author={Hsieh, Cheng-Yu and Zhang, Jieyu and Ma, Zixian and Kembhavi, Aniruddha and Krishna, Ranjay},
  booktitle={Thirty-Seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```
