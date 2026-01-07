# RAGAS 메트릭 인스펙터 도구

이 디렉토리는 RAGAS의 각 메트릭이 내부적으로 어떻게 동작하는지 분석하는 도구들입니다.
각 메트릭의 프롬프트, Few-Shot 예제, 실행 흐름을 완전히 추출합니다.

## 파일 구조

```
RAGAS-inspect/
├── README.md                      # 이 파일
├── PROMPTS_GUIDE.md               # 프롬프트 가이드
├── run_all_inspectors.py          # 전체 실행 스크립트
├── inspect_faithfulness.py        # Faithfulness 메트릭 분석
├── inspect_answer_relevancy.py   # AnswerRelevancy 메트릭 분석
├── inspect_context_precision.py  # ContextPrecision 메트릭 분석
├── inspect_context_recall.py     # ContextRecall 메트릭 분석
└── inspect_answer_correctness.py # AnswerCorrectness 메트릭 분석
```

## 🚀 사용 방법

### 1. 전체 메트릭 분석 (권장)
```bash
cd RAGAS-inspect
python run_all_inspectors.py
```

### 2. 특정 메트릭만 분석
```bash
# Faithfulness만
python run_all_inspectors.py faithfulness

# AnswerRelevancy만
python run_all_inspectors.py answer_relevancy

# ContextPrecision만
python run_all_inspectors.py context_precision

# ContextRecall만
python run_all_inspectors.py context_recall

# AnswerCorrectness만
python run_all_inspectors.py answer_correctness
```

### 3. 개별 스크립트 직접 실행
```bash
python inspect_faithfulness.py
python inspect_answer_relevancy.py
# ... 등등
```

## 각 인스펙터가 추출하는 정보

각 스크립트는 다음을 추출합니다:

### 1️⃣ 기본 정보
- 메트릭 이름
- 필요한 입력 컬럼
- 프롬프트 개수

### 2️⃣ 각 단계별 프롬프트
- **Instruction**: LLM에게 주어지는 명령문 (전체 텍스트)
- **Input Keys**: 어떤 데이터를 입력으로 받는지
- **Output Keys**: 어떤 결과를 반환하는지
- **Output Type**: 반환 데이터 타입

### 3️⃣ Few-Shot Examples (가장 중요!)
- **전체 예제 개수**: 보통 2-3개
- **각 예제의 입력 데이터**: 질문, 컨텍스트, 답변 등
- **각 예제의 출력 데이터**: verdict, reason 등
- **예제 길이**: 토큰 비용 산정에 중요

### 4️⃣ 실행 흐름 설명
- 메트릭이 몇 단계로 실행되는지
- 각 단계에서 무엇을 하는지
- 최종 점수가 어떻게 계산되는지
- 실제 예시와 함께 설명

## 메트릭별 설명

### 1. Faithfulness (충실성)
**평가 질문**: "답변이 검색된 문서에 근거했는가?"

**실행 단계**:
- Step 1: Statement Generation (답변을 개별 진술문으로 분해)
- Step 2: NLI Verification (각 진술문이 컨텍스트에 지지되는지 검증)

**사용 예**: 환각(hallucination) 탐지

---

### 2. AnswerRelevancy (답변 관련성)
**평가 질문**: "답변이 질문과 관련있는가?"

**실행 단계**:
- Step 1: Question Generation (답변으로부터 질문 역생성)
- Step 2: Semantic Similarity (원본 질문과의 유사도 계산)

**사용 예**: 엉뚱한 답변 탐지

---

### 3. ContextPrecision (컨텍스트 정밀도)
**평가 질문**: "검색된 컨텍스트가 모두 유용한가?"

**실행 단계**:
- Step 1: Context Usefulness (각 컨텍스트의 유용성 판단)

**사용 예**: 검색 품질 평가 (노이즈 측정)

---

### 4. ContextRecall (컨텍스트 재현율)
**평가 질문**: "필요한 정보를 모두 검색했는가?"

**실행 단계**:
- Step 1: Sentence Extraction (정답을 문장으로 분리)
- Step 2: Attribution (각 문장이 컨텍스트에 있는지 확인)

**사용 예**: 검색 범위 평가 (정보 누락 측정)

---

### 5. AnswerCorrectness (답변 정확성)
**평가 질문**: "답변이 정답과 일치하는가?"

**실행 단계**:
- Step 1: Statement Extraction (답변과 정답을 진술문으로 분해)
- Step 2: F1 Score Calculation (사실적 정확성)
- Step 3: Semantic Similarity (의미적 유사성)

**사용 예**: 전체적인 답변 품질 평가

## 활용 방법

### 1. Few-Shot 예제 이해하기
각 메트릭의 예제를 보면, RAGAS가 무엇을 "좋은 답변"/"나쁜 답변"으로 판단하는지 알 수 있습니다.

### 2. 커스텀 예제 만들기
한국 경제 데이터에 맞는 예제를 만들고 싶다면, 기존 예제의 형식을 참고하세요.

### 3. 비용 최적화
Few-shot 예제의 길이를 확인하여 토큰 비용을 산출할 수 있습니다.

### 4. 평가 기준 조정
프롬프트와 예제를 수정하면 평가 기준을 도메인에 맞게 조정할 수 있습니다.

## 문제 해결

### ImportError: No module named 'ragas'
```bash
pip install ragas
```

### 출력이 잘린 경우
각 스크립트의 문자열 자르기 부분 (`[:200]`, `[:300]`)을 수정하세요.

## 추가 자료

- RAGAS 공식 문서: https://docs.ragas.io/
- RAGAS GitHub: https://github.com/explodinggradients/ragas

## 체크리스트

사용 전 확인사항:
- [ ] RAGAS 설치됨 (`pip install ragas`)
- [ ] 가상환경 활성화됨
- [ ] Python 3.8 이상

---

**만든 목적**: RAGAS의 "블랙박스"를 열어 내부 동작을 완전히 이해하기 위함
