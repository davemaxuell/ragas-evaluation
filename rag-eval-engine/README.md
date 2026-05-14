# RAG 평가 엔진 (RAG Evaluation Engine)

RAG(Retrieval-Augmented Generation) 시스템의 성능을 평가하는 Python 기반 도구입니다.

## 주요 기능

- **5가지 핵심 RAG 메트릭**: Faithfulness, Context Recall, Answer Relevancy, Context Precision, Answer Correctness
- **Google Gemini 지원**: `score_json.py` 및 `score_json_multiturn.py`는 Google Gemini API 사용
- **Flask API (OpenAI)**: `api.py`는 OpenAI GPT 모델 사용
- **커스텀 프롬프트**: JSON 설정 파일로 프롬프트 커스터마이징 지원
- **RAGAS 인스펙터**: 메트릭 내부 구조 분석 도구 제공

---

## 프로젝트 구조

```
rag-eval-engine/
├── api.py                    # Flask 서버 및 RAGAS 평가 로직 (메인 파일)
├── test_api.py               # API 테스트 클라이언트
├── score_json.py             # 🆕 JSON 파일 직접 평가 (Single-turn)
├── score_json_multiturn.py   # 🆕 JSON 파일 직접 평가 (Multi-turn + AspectCritic)
├── requirements.txt          # 필수 라이브러리 목록
├── .env.example              # API 키 템플릿 (.env로 복사 후 입력)
├── eval_data/                # 평가 데이터셋 폴더 (git 제외 — 직접 준비)
│   └── test_sample.csv
│
├── custom_prompts/           # 커스텀 프롬프트 설정
│   ├── README.md             # 커스텀 프롬프트 사용법
│   ├── faithfulness.json     # Faithfulness 프롬프트 설정
│   ├── answer_correctness.json
│   ├── context_precision.json
│   ├── context_recall.json
│   ├── answer_relevancy.json
│   └── defaults/             # 기본 instruction 참조용
│       ├── README.md
│       └── *.json            # 각 메트릭 기본값
│
└── RAGAS-inspect/            # RAGAS 메트릭 분석 도구
    ├── README.md             # 인스펙터 사용법
    ├── PROMPTS_GUIDE.md      # 프롬프트 가이드
    ├── run_all_inspectors.py # 전체 실행 스크립트
    └── inspect_*.py          # 각 메트릭별 인스펙터
```

---

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 라이브러리 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.env.example`을 복사하여 `.env` 파일 생성 후 키 입력:
```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL_NAME=gpt-4o-2024-08-06
EMBEDDING_MODEL_NAME=text-embedding-3-small
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL_NAME=gemini-2.5-flash
```

### 3. 서버 실행

```bash
python api.py
```

출력 예시:
```
[성공] OPENAI API 키 로드 완료.
[초기화] 모델 및 메트릭 설정 중
[성공] 모델 및 메트릭 설정 완료.
[데이터 로드] samples.csv 파일 읽는 중...
[데이터 준비] 총 15개의 샘플 로드 완료.
[서버 시작] http://0.0.0.0:5001/evaluate 에서 요청 대기 중...
```

### 4. 평가 요청

```bash
python test_api.py
```

또는 curl:
```bash
curl -X POST http://127.0.0.1:5001/evaluate
```

---

## JSON 파일 직접 평가 (Standalone Scripts) - Gemini 기반

Flask API 없이 JSON 파일을 직접 평가할 수 있는 스크립트입니다. **Google Gemini API**를 사용합니다.

### `score_json.py` - Single-Turn 평가

첫 번째 turn만 평가하는 간단한 스크립트입니다.

```bash
python score_json.py "path/to/your/data.json"
```

**특징:**
- 5개 메트릭 평가 (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness)
- **Google Gemini API 사용** (기본 모델: `gemini-2.0-flash`)
- 커스텀 프롬프트 사용 (`custom_prompts/` 폴더)
- Rate limiting 및 자동 재시도 지원
- 출력 파일명에 모델명 포함 (예: `*_scored(gemini-2.0-flash).json`)

### `score_json_multiturn.py` - Multi-Turn 평가 ⭐

모든 turn을 평가하고 Item별 평균을 계산하는 고급 스크립트입니다.

```bash
python score_json_multiturn.py "path/to/your/data.json"
```

**특징:**
- 모든 turn 순회하며 개별 평가
- **Google Gemini API 사용**
- Item별 평균 점수 자동 계산
- Multi-turn 아이템에 AspectCritic 메트릭 추가 (대화 품질 평가)
- Single-turn vs Multi-turn 자동 감지

**출력 예시:**
```
[3/4] 평가 시작...
  [1/100] Item 1 (Single-turn, 1 turns) 평가 중...
    [turn_1] 평가 중...
      F=1, AR=1, CP=1, CR=1, AC=1.0
    → Item Average: faithfulness=1.00, answer_relevancy=1.00, ...

  [5/100] Item 5 (Multi-turn, 3 turns) 평가 중...
    [turn_1] 평가 중...
    [turn_2] 평가 중...
    [turn_3] 평가 중...
    [AspectCritic] = 1 (Session Level: Contradiction, Retention, Redundancy check)
    → Item Average: faithfulness=0.67, ..., aspect_critic=1.00
```

**✅ 최근 업데이트 (Prompt & Logic 변화):**
1. **Context Recall (Reasoning 강화)**: 정답(Ground Truth)과 Context를 대조할 때 `[Analysis]` 단계를 거쳐 수치 오류 등을 엄격하게 검증하도록 개선됨.
2. **Aspect Critic (Session Level)**: 기존 단순 품질 평가에서 **Session-level** 평가로 업그레이드.
   - **Contradiction**: 모순 여부 확인
   - **Retention**: 문맥 유지 여부 확인
   - **Redundancy**: 중복 여부 확인
   - 하나라도 실패하면 0점 부여
3. **Answer Correctness (Strict & Flexible)**:
   - **Strict**: 수치(연도, %, 금액)는 정확히 일치해야 함.
   - **Flexible**: 비교 서술이나 배경 설명이 포함되어도 핵심 수치가 맞으면 인정.
   - **Output**: F1-Score (0.0 ~ 1.0) 자동 추출.
```

**결과 JSON 구조:**
```json
{
  "items": [
    {
      "item_id": 5,
      "turns": [
        {"turn_id": "turn_1", "scores": {"faithfulness": 1, ...}},
        {"turn_id": "turn_2", "scores": {"faithfulness": 1, ...}},
        {"turn_id": "turn_3", "scores": {"faithfulness": 0, ...}}
      ],
      "item_average_scores": {
        "faithfulness": 0.667,
        "answer_relevancy": 0.667,
        "context_precision": 1.0,
        "context_recall": 0.667,
        "answer_correctness": 1.0,
        "aspect_critic": 1
      }
    }
  ],
  "evaluation_results": {
    "total_items": 100,
    "single_turn_items": 60,
    "multi_turn_items": 40,
    "summary_scores": {
      "faithfulness": 0.85,
      "answer_relevancy": 0.92,
      ...
    }
  }
}
```

---

## 평가 메트릭

| 메트릭 | 설명 | 필수 컬럼 |
|--------|------|----------|
| **Faithfulness** | 답변이 컨텍스트에 근거하는가? | response, retrieved_contexts |
| **AnswerRelevancy** | 답변이 질문과 관련있는가? | response, question |
| **ContextPrecision** | 검색된 컨텍스트가 유용한가? | retrieved_contexts, question |
| **ContextRecall** | 필요한 정보를 모두 검색했는가? | retrieved_contexts, ground_truth |
| **AnswerCorrectness** | 답변이 정답과 일치하는가? | response, ground_truth |

---

## 커스텀 프롬프트

### 개요

RAGAS 메트릭의 프롬프트를 커스터마이징하여 특정 도메인이나 언어에 최적화할 수 있습니다.

### 사용법

#### 1단계: 언어만 변경 (가장 간단)

`custom_prompts/faithfulness.json` 편집:

```json
{
  "metric_name": "faithfulness",
  "enabled": true,
  "language": "korean"
}
```

#### 2단계: instruction 커스터마이징

```json
{
  "metric_name": "faithfulness",
  "enabled": true,
  "language": "korean",
  "prompts": {
    "statement_generator_prompt": {
      "instruction": "주어진 답변을 개별 진술문으로 분해하세요...",
      "examples": []
    }
  }
}
```

### 설정 파일 구조

```json
{
  "metric_name": "string",        // 메트릭 이름
  "enabled": false,               // true로 변경하면 커스텀 프롬프트 적용
  "language": "english",          // 프롬프트 언어 (english, korean 등)
  "prompts": {
    "prompt_name": {
      "instruction": "",          // 빈 문자열이면 기본값 사용
      "examples": []              // 빈 배열이면 기본값 사용
    }
  }
}
```

### 기본 instruction 참조

`custom_prompts/defaults/` 폴더에서 각 메트릭의 기본 instruction을 확인할 수 있습니다.

---

## RAGAS 인스펙터

### 개요

RAGAS 메트릭의 내부 구조(프롬프트, Few-shot 예제 등)를 분석하는 도구입니다.

### 사용법

```bash
cd RAGAS-inspect

# 전체 메트릭 분석
python run_all_inspectors.py

# 특정 메트릭만 분석
python inspect_faithfulness.py
python inspect_answer_correctness.py
python inspect_context_precision.py
python inspect_context_recall.py
python inspect_answer_relevancy.py
```

### 분석 결과

각 인스펙터는 다음을 출력합니다:
- 메트릭 기본 정보
- 발견된 프롬프트 목록
- 각 프롬프트의 instruction (전체 텍스트)
- Few-shot examples (입력/출력 데이터)
- 실행 흐름 설명

---

## 데이터 형식

### CSV 필수 컬럼

| 컬럼명 | 설명 |
|--------|------|
| `question` | 사용자 질문 |
| `contents` | 검색된 컨텍스트 |
| `answer` | RAG 시스템 답변 |
| `ground_truth` | 정답 |

### 응답 형식

```json
{
  "status": "success",
  "evaluated_samples": 15,
  "summary_scores": {
    "faithfulness": 0.95,
    "answer_relevancy": 0.88,
    "context_precision": 0.92,
    "context_recall": 0.85,
    "answer_correctness": 0.78
  },
  "individual_scores": [
    { "question": "...", "faithfulness": 1.0, ... }
  ]
}
```

---

## 모델 설정

### Gemini 모델 (score_json.py, score_json_multiturn.py)

| 모델 | 특징 | 추천 용도 |
|------|------|----------|
| `gemini-2.0-flash` ⭐ | 빠름, 저렴, 좋은 품질 | 기본값, 대량 평가 |
| `gemini-1.5-pro` | 고품질 추론 | 복잡한 비교 |
| `gemini-1.5-flash` | 매우 빠름, 가장 저렴 | 간단한 작업 |

`.env`에서 변경:
```dotenv
GEMINI_MODEL_NAME=gemini-2.0-flash
```

### OpenAI 모델 (api.py)

| 용도 | 기본 모델 |
|------|------|
| 평가자 LLM | `gpt-4o-2024-08-06` |
| 임베딩 | `text-embedding-3-small` |

`.env`에서 변경:

```dotenv
OPENAI_MODEL_NAME=gpt-4o-2024-08-06
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

---

## 문제 해결

### ModuleNotFoundError: No module named 'ragas'
```bash
pip install ragas
```

### ModuleNotFoundError: No module named 'google'
```bash
pip install google-generativeai
```

### API 키 오류
- Gemini 스크립트: `.env`에 `GEMINI_API_KEY` 설정 확인
- OpenAI 스크립트: `.env`에 `OPENAI_API_KEY` 설정 확인

### Rate Limit 오류 (429)
스크립트에 자동 재시도 및 지수 백오프가 포함되어 있습니다. 지속되면 잠시 후 다시 시도하세요.

### 서버 연결 실패
`api.py`가 실행 중인지 확인 (포트: 5001)

---

## 라이선스

MIT License