# Tomato SoM VLM

YOLO 1-class proposal + Qwen3-VL-4B SoM으로 토마토 숙성도(ripe / semi_ripe / unripe)를 분류합니다.
GT 박스를 VLM에 넣지 않습니다. `not_tomato`로 나온 proposal은 버립니다.

개념·그림 튜토리얼: [`tutorial/README.md`](tutorial/README.md)

## 준비

GPU 1장 (SFT ~24GB, RL은 28GB+면 k=4). Python 3.12+.

```bash
uv sync
# 또는: pip install -e .
```

Qwen 가중치는 첫 실행 때 Hugging Face에서 받아집니다 (`Qwen/Qwen3-VL-4B-Instruct`).

## 데이터 / 가중치

기본 데이터셋은 Laboro **little** (`--ds little`). 이미지 원본(`data/`, `laboro-tomato/`)은 git에 없고, 서버 홈 복사본에 있다.

| 역할 | 경로 |
|---|---|
| 이미지·GT (COCO) | `data/raw/laboro_tomato_little/laboro_little/` |
| YOLO 학습용 | `data/yolo/laboro_little_yolo_1cls/`, `..._3cls/` |
| YOLO 1cls (proposal, conf=0.1) | `weights/little/yolo11_1cls.pt` |
| YOLO 3cls (baseline) | `weights/little/yolo11.pt` |
| SFT/RL LoRA | `results/som_little/sft_lora/`, `rl_lora/` |

다른 셋: `--ds kaggle` (`laboro-tomato/`), `--ds laboro_tomatod`.

YOLO를 다시 학습하려면 (이미 `weights/`에 있으면 생략):

```bash
python train_yolo.py --ds little --cls 3
python train_yolo.py --ds little --cls 1
```

## 한 번에 재현 (little, 32장 eval)

```bash
python vs_yolo.py --ds little
python zeroshot_som.py --ds little
python zeroshot_som.py --ds little --icl
python dump_qwen_inputs.py --ds little          # GPU 불필요 (YOLO cache 있으면)
python sft_som.py --ds little --steps 400
python rl_som.py --ds little --steps 200        # SFT LoRA에서 이어서
python report.py --ds little
```

저장된 LoRA만 다시 평가:

```bash
python eval_lora.py --ds little --tag sft
python eval_lora.py --ds little --tag rl
```

## 스크립트가 만드는 것

모두 `results/som_little/` 아래.

| 명령 | 결과 |
|---|---|
| `vs_yolo.py` | `yolo32.json` — YOLO 3cls vs 같은 32장 |
| `zeroshot_som.py` | `zeroshot.json`, `qual/zeroshot/` (GT \| YOLO3 \| YOLO1 \| VLM) |
| `zeroshot_som.py --icl` | `icl.json`, `qual/icl/` — train ICL 2장 + 쿼리 |
| `dump_qwen_inputs.py` | `qwen_inputs/` — 아래 그림 설명 |
| `sft_som.py` | `sft.json`, `sft.log`, `sft_metrics.jsonl`, `sft_lora_best/`, `sft_lora_last/` |
| `rl_som.py` | `rl.json`, `rl.log`, `rl_metrics.jsonl`, `rl_lora_best/`, `rl_lora_last/` |
| `report.py` | 표 출력 |

SFT/RL은 **20 step마다** 32장 eval. 가중치는 **best**(macro-F1)와 **last**만 남깁니다. 중간 점수는 `sft.log` / `rl.log`로 보면 됩니다.

현재 폴더의 `sft_lora/`, `rl_lora/`는 첫 완성 런의 last 가중치입니다 (중간 ckpt 없이 끝난 버전). 새로 돌리면 `*_lora_best` / `*_lora_last`가 생깁니다. `eval_lora.py`는 best → last → `*_lora` 순으로 찾습니다.

## VLM이 보는 그림 (`qwen_inputs/`)

`index.html` 또는 `overview.jpg`.

Eval (zeroshot/ICL/SFT eval)은 이미지당 **한 번** 호출합니다.

- `icl_0/`, `icl_1/` — in-context 예시. 각 폴더 `input.jpg` / `som.jpg` / `mosaic.jpg` / `strip.jpg` / gold `label.json`
- `query_*` — 테스트 이미지. 박스는 YOLO 1cls @ 0.1 (최대 20)

SFT/RL 학습은 위 full JSON이 아니라 **랜덤 1개 stage + 그 stage ICL 1장**입니다. 같은 쿼리 이미지에 대해 4과제가 `vqa/`와 `vqa_overview.jpg`에 있습니다.

| stage | 질문 | gold 값 |
|---|---|---|
| `dedup` | 같은 열매에 박스가 여러 개면? | `keep` / `dup` |
| `filt` | 토마토인가? 너무 작으면? | `tomato` / `not_tomato` |
| `unripe` | 남은 토마토가 미숙인가? | `unripe` / `later` |
| `ripe_semi` | 나머지가 완숙/반숙? | `ripe` / `semi_ripe` |

각 stage 폴더에 `icl/` (worked example)과 `query/` (학습 샘플)가 있습니다. `prompt.txt`가 실제 질문, `label.json`이 그 stage의 정답입니다.

## 프로토콜

- Proposal: YOLO 1cls, conf 0.1, 최대 20 박스. GT 박스 미사용.
- SoM = 번호 박스 전체 장면, mosaic = 크롭(칸 순서 shuffle). 색은 mosaic, 가려진 열매·중복은 SoM.
- 작음: train GT p5보다 작은 proposal은 학습 gold가 `not_tomato`. 하드 필터로 버리지는 않음.
- Eval: 32장, det IoU 0.5, tomato 3클래스 macro-F1 / acc.
- SFT: LoRA r=16, 400 step, JSON only (reasoning 없음).
- RL: SFT 위에 GRPO-lite, 200 step, k는 VRAM에 따라 2–4.

## little 32장 참고 점수

det IoU 0.5. VLM 쪽은 모두 YOLO 1cls @ 0.1 proposal.

| method | acc | macroF1 | ripe | semi | unripe |
|---|---|---|---|---|---|
| YOLO 3cls | 0.604 | 0.720 | 0.715 | 0.659 | 0.787 |
| YOLO 1cls + zero-shot VLM | 0.583 | 0.698 | 0.761 | 0.560 | 0.772 |
| YOLO 1cls + ICL VLM | 0.566 | 0.685 | 0.719 | 0.575 | 0.760 |
| YOLO 1cls + VLM SFT | 0.637 | 0.762 | 0.808 | 0.695 | 0.784 |
| YOLO 1cls + VLM SFT + RL | **0.670** | **0.795** | 0.840 | 0.738 | 0.806 |

## 코드

| 파일 | |
|---|---|
| `som_common.py` | 데이터, SoM, prompt, eval |
| `train_yolo.py` | YOLO 1/3cls 학습 |
| `vs_yolo.py` | YOLO 3cls 베이스라인 |
| `zeroshot_som.py` | zero-shot / ICL |
| `sft_som.py` | LoRA SFT |
| `rl_som.py` | SFT 이어 RL |
| `dump_qwen_inputs.py` | 입력·ICL·VQA 그림 |
| `eval_lora.py` | 저장 LoRA 재평가 |
| `report.py` | 결과 표 |
