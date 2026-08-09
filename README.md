# EXP 1 — 형태소 분석 Demo: 환경 설정 안내

본 논문 실험 1(형태소 분석)의 SFT/추론 파이프라인 재현 예시를 실행하기 위한
환경 설정 안내입니다. **아래 설정을 모두 마친 뒤 `notebook/` 폴더의 노트북을 실행하세요.**

## 폴더 구조

```
프로젝트_루트/
├── readme/
│   └── README.md                                  (이 파일)
├── notebook/
│   └── exp1_morpheme_4way_demo.ipynb
└── data/
    └── nikl_mp_v1_1_repeated_100_sample.json
```

노트북은 `../data/nikl_mp_v1_1_repeated_100_sample.json` 상대경로로 데이터를 찾으므로,
위 구조를 그대로 유지해야 합니다. 노트북을 다른 위치로 옮기는 경우
노트북 안의 `NIKL_JSON_PATH` 값을 직접 수정하세요.

## 실행 환경 (참고용)

다른 환경(다른 CUDA/Python 버전 등)에서는 `torch`/`bitsandbytes` 설치 방법이 달라질 수 있습니다.

| 항목 | 권장/검증 사항 |
|---|---|
| Python | 3.12.3 |
| GPU VRAM | **24GB 이상 권장** (Gemma 4bit 양자화 로드 + LoRA 학습 기준) |
| CUDA | 12.8 이상 (드라이버가 지원하는 최대 버전이 12.8 이상이면 하위 호환으로 정상 동작) |
| PyTorch CUDA 빌드 | cu128 (아래 "1. 패키지 설치" 참고) |

VRAM이 24GB보다 적으면 4bit 양자화를 쓰더라도 결과 1(추론)·결과 3(SFT) 섹션에서
메모리 부족(OOM) 오류가 날 수 있습니다. 이 경우:
- `Cell 2`(패키지 설치)는 그대로 두고, `Cell 8`/`Cell 17`의 `model_id`를
  더 작은 모델로 바꾸거나
- 결과 2(GPT), 결과 4(Kiwi)만 실행하는 것으로 범위를 좁히세요 (GPU 자체가 불필요).

자신의 CUDA 버전은 아래 명령으로 확인할 수 있습니다.

```bash
nvidia-smi
```

출력 우측 상단의 `CUDA Version`이 드라이버가 지원하는 최대 버전입니다.
이 값이 PyTorch 빌드 버전(아래 설치 명령의 `cu128`) 이상이면 하위 호환으로 정상 동작합니다.

> GPU가 없거나 다른 GPU를 쓰는 환경이어도 **결과 2(GPT)와 결과 4(Kiwi)는 CPU만으로 실행 가능**합니다.
> 이 경우 아래 "GPU 없이 실행하는 경우" 섹션을 참고하세요.

## 1. 패키지 설치

아래 버전은 실제 실험 환경(`requirements.txt` 기준)에서 사용한 버전을 그대로 명시한 것입니다.

### PyTorch (CUDA 12.8 빌드)
PyPI 기본 인덱스에는 `+cu128` 빌드가 없으므로 PyTorch 전용 인덱스에서 설치해야 합니다.

```bash
pip install torch==2.10.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### Gemma 추론/SFT용 (결과 1, 3)
```bash
pip install transformers==5.5.4 peft==0.19.1 trl==1.2.0 bitsandbytes==0.49.2 \
    accelerate==1.13.0 datasets==4.8.4
```

### Kiwi 형태소 분석기 (결과 4)
```bash
pip install kiwipiepy==0.23.1 kiwipiepy_model==0.23.0
```

### GPT API (결과 2)
```bash
pip install openai==2.36.0 python-dotenv==1.2.2
```

### 공통 데이터 처리·집계
```bash
pip install pandas==3.0.2 numpy==2.4.3 tqdm==4.67.3 openpyxl==3.1.5
```

### 설치 확인
아래를 실행해 torch가 의도한 CUDA 빌드로 설치됐는지, GPU를 정상 인식하는지 확인하세요.

```python
import torch
print(f"torch 버전:       {torch.__version__}")
print(f"CUDA available:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 개수:          {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}  "
              f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
```

## 2. 환경 변수 설정

| 변수 | 용도 | 필요한 섹션 |
|---|---|---|
| `HF_TOKEN` | Gemma 모델(gated model) 다운로드 인증 | 결과 1, 3 |
| `OPENAI_API_KEY` | OpenAI GPT API 인증 | 결과 2 |

셸 환경 변수로 설정하거나, `notebook/` 폴더(또는 프로젝트 루트)에 `.env` 파일을 만들어도 됩니다.

```bash
# .env 파일 예시
HF_TOKEN=hf_...
OPENAI_API_KEY=sk-...
```

> `.env` 파일은 절대 git에 커밋하지 마세요. `.gitignore`에 `.env`를 추가하는 것을 권장합니다.

## 3. GPU 없이 실행하는 경우

- 결과 2(GPT), 결과 4(Kiwi) 섹션은 GPU 없이도 정상 실행됩니다.
- 결과 1, 3(Gemma) 섹션을 CPU로 실행하려면 위 PyTorch 설치 명령에서
  `--index-url https://download.pytorch.org/whl/cu128` 대신 CPU 전용 빌드를 사용하세요.
  단, `bitsandbytes`의 4bit 양자화는 GPU 전용 기능이므로 CPU 환경에서는 해당 코드가
  동작하지 않거나 매우 느릴 수 있습니다.

## 4. Docker 컨테이너에서 실행하는 경우

Jupyter가 컨테이너 안에서 실행 중이라면, 로컬 PC의 파일이 컨테이너 안에서
그대로 보이지 않습니다. 아래 중 하나로 프로젝트 폴더 전체를 컨테이너 내부로 가져오세요.

- **볼륨 마운트 (권장)**: 컨테이너 실행 시 프로젝트 루트 폴더를 통째로 마운트
  ```bash
  docker run -v /로컬/프로젝트_루트:/workspace/project ... 이미지명
  ```
  이렇게 하면 `readme/`, `notebook/`, `data/` 폴더 구조가 컨테이너 안에서도 그대로 유지됩니다.
- **Jupyter 업로드**: Jupyter 웹 UI의 파일 브라우저에서 각 폴더를 만들고 파일을 업로드
- **`docker cp`**: 이미 실행 중인 컨테이너에 개별 복사
  ```bash
  docker cp /로컬/프로젝트_루트 컨테이너이름:/workspace/project
  ```

## 5. 문제 해결 (Troubleshooting)

### HuggingFace 캐시 폴더 권한 오류 (PermissionError)

```
PermissionError: [Errno 13] Permission denied: '/home/사용자명/.cache/huggingface/hub/models--google--gemma-...'
OSError: PermissionError at ... when downloading google/gemma-4-E2B-it. Check cache directory permissions.
```

**원인**: Gemma 모델을 처음 다운로드할 때 huggingface_hub이 기본적으로
`~/.cache/huggingface`(홈 디렉토리 하위)에 캐시를 만드는데, 아래와 같은 경우 이 폴더에
쓰기 권한이 없어서 발생합니다.

- 홈 디렉토리가 NFS 등 네트워크 파일시스템이라 소유권/권한이 꼬여 있는 경우
- 이전에 다운로드가 중간에 취소되어, 다른 사용자·프로세스 소유의 lock/임시 폴더가 남아있는 경우
- 여러 사용자가 같은 서버를 공유하는데 캐시 폴더가 특정 사용자 소유로 생성된 경우

**해결**: 이 노트북은 **`Cell 7`에서 `HF_HOME`을 프로젝트 폴더 안(`../.hf_cache`)으로
자동 설정**하므로, 위 구조(`readme/`, `notebook/`, `data/`)를 그대로 쓰면 기본적으로
이 문제를 피할 수 있습니다. 그래도 같은 오류가 발생한다면:

1. **이미 `HF_HOME`이 다른 값으로 설정되어 있는지 확인** — 셸에서 `echo $HF_HOME`
   실행 시 값이 나온다면, 노트북의 `os.environ.setdefault(...)`는 그 값을 그대로 존중하므로
   그 경로의 권한을 확인해야 합니다. 임시로 해제하려면: `unset HF_HOME` 후 노트북 커널 재시작.
2. **문제되는 캐시 폴더를 직접 삭제 후 재시도**
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it
   ```
3. **권한을 직접 확인**
   ```bash
   ls -la ~/.cache/huggingface/hub/
   ```
   소유자가 본인이 아니거나 쓰기 권한이 없다면, 관리자에게 문의하거나
   `HF_HOME` 환경변수를 본인이 쓰기 권한을 가진 다른 경로로 지정하세요.
   ```bash
   export HF_HOME=/원하는/쓰기가능한/경로
   ```

### 그 외 문제

- **`FileNotFoundError` (데이터 파일)**: 노트북의 `Cell 0-0`(경로 진단) 출력을 확인하고,
  위 "폴더 구조" 섹션대로 `data/` 폴더가 배치되어 있는지 확인하세요.
- **`bitsandbytes` 관련 오류**: GPU/CUDA 버전과 `bitsandbytes` 빌드가 맞지 않을 때 발생합니다.
  `nvidia-smi`로 CUDA 버전을 다시 확인하고, 위 "1. 패키지 설치" 섹션의 버전을 그대로 맞춰 설치하세요.


## 6. 실행 순서

1. 위 1~4 단계로 환경 설정을 모두 마칩니다.
2. `notebook/exp1_morpheme_4way_demo.ipynb`를 엽니다.
3. 노트북 상단의 **Cell 0-0(경로 진단)**을 먼저 실행해 데이터 파일을 정상적으로 찾는지 확인합니다.
4. 이후 노트북에 적힌 순서대로 위에서부터 아래로 실행합니다.
   - 결과 1(Gemma 추론), 결과 2(GPT), 결과 3(Gemma SFT), 결과 4(Kiwi) 순서이며,
     각 섹션은 서로 독립적으로 실행 가능하므로 필요한 섹션만 골라 실행해도 됩니다.
