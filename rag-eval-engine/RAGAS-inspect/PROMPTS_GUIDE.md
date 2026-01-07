# RAGAS 프롬프트 완전 가이드

이 문서는 RAGAS의 모든 메트릭에서 사용되는 프롬프트와 메서드를 설명합니다.

## 📚 목차

- [프롬프트 vs 메서드 구분](#프롬프트-vs-메서드-구분)
- [공통 메서드 (모든 메트릭)](#공통-메서드)
- [메트릭별 프롬프트](#메트릭별-프롬프트)
- [프롬프트 커스터마이징](#프롬프트-커스터마이징)

---

## 프롬프트 vs 메서드 구분

인스펙터를 실행하면 많은 "prompt" 관련 항목이 나오지만, 실제로는 두 가지 타입이 있습니다:

### ✅ 실제 프롬프트 (LLM에 전송됨)
- **Instruction**: 태스크 설명
- **Few-shot Examples**: 학습 예제
- **Input/Output Keys**: 데이터 형식
- 예: `correctness_prompt`, `statement_generator_prompt`

### 🔧 유틸리티 메서드 (프롬프트 관리 도구)
- **Python 메서드**: 프롬프트를 조작하는 함수
- LLM에 직접 전송되지 않음
- 예: `adapt_prompts`, `get_prompts`, `save_prompts`

---

## 공통 메서드

모든 RAGAS 메트릭에 포함된 헬퍼 메서드들입니다.

### 1. `adapt_prompts`

**용도**: 프롬프트를 다른 언어로 자동 변환

**타입**: Method (함수)

**기능**:
- 입력 데이터의 언어를 감지
- Instruction과 예제를 해당 언어로 번역
- 자동으로 적용 (선택적)

**사용 예시**:
```python
metric = ContextPrecision()
metric.adapt_prompts(language="korean")
# 이제 모든 프롬프트가 한국어로 변환됨
```

**언제 사용?**:
- 한국어 데이터 평가 시 accuracy 향상 원함
- 다국어 지원이 필요한 경우

---

### 2. `get_prompts`

**용도**: 현재 메트릭의 모든 프롬프트를 딕셔너리로 반환

**타입**: Method

**기능**:
- 모든 프롬프트 객체를 가져옴
- 검사, 수정, 저장에 유용

**사용 예시**:
```python
metric = Faithfulness()
prompts = metric.get_prompts()
print(prompts.keys())
# Output: ['statement_prompt', 'nli_statements_message']
```

**언제 사용?**:
- 프롬프트 백업 전
- 현재 설정 확인
- 프로그래밍 방식으로 프롬프트 조작

---

### 3. `set_prompts`

**용도**: 커스텀 프롬프트를 메트릭에 설정

**타입**: Method

**기능**:
- 기존 프롬프트를 새 프롬프트로 교체
- 딕셔너리 형태로 여러 프롬프트 한 번에 변경

**사용 예시**:
```python
metric = ContextPrecision()
custom_prompts = {
    'context_precision_prompt': my_custom_prompt
}
metric.set_prompts(**custom_prompts)
```

**언제 사용?**:
- 완전히 새로운 프롬프트 적용
- 저장된 프롬프트 불러온 후

---

### 4. `save_prompts`

**용도**: 현재 프롬프트를 JSON 파일로 저장

**타입**: Method

**기능**:
- 프롬프트 설정을 파일에 저장
- 버전 관리 및 공유 가능

**사용 예시**:
```python
metric = Faithfulness()
metric.statement_prompt.language = "korean"
metric.save_prompts("my_prompts.json")
```

**언제 사용?**:
- 커스텀 프롬프트 백업
- 팀원과 공유
- 프로덕션 설정 저장

---

### 5. `load_prompts`

**용도**: 저장된 프롬프트 JSON 파일 불러오기

**타입**: Method

**기능**:
- 파일에서 프롬프트 복원
- 이전 설정 재사용

**사용 예시**:
```python
metric = Faithfulness()
metric.load_prompts("my_prompts.json")
# 저장된 설정 복원됨
```

**언제 사용?**:
- 팀원이 공유한 설정 사용
- 프로덕션 환경 배포
- A/B 테스트 시 다른 설정 전환

---

## 메트릭별 프롬프트

각 메트릭이 실제로 LLM에 보내는 프롬프트들입니다.

### 📊 AnswerCorrectness

#### 1. `statement_generator_prompt`
- **목적**: 텍스트를 개별 진술문으로 분해
- **입력**: question, answer
- **출력**: List of statements
- **예제 수**: 1개
- **사용 횟수**: 2회 (answer용 1회 + ground_truth용 1회)

**예시**:
```
입력: "He was a physicist and won Nobel Prize."
출력: ["Albert Einstein was a physicist.", 
       "Albert Einstein won Nobel Prize."]
```

#### 2. `correctness_prompt`
- **목적**: 진술문을 TP/FP/FN으로 분류
- **입력**: answer_statements, ground_truth_statements
- **출력**: TP, FP, FN 리스트
- **예제 수**: 2개
- **사용 횟수**: 1회

**예시**:
```
TP: 답변과 정답 모두에 있는 진술
FP: 답변에만 있는 진술 (잘못된 정보)
FN: 정답에만 있는 진술 (누락된 정보)
```

---

### 📊 Faithfulness

#### 1. `statement_prompt`
- **목적**: 답변을 atomic statements로 분해
- **입력**: question, response
- **출력**: List of statements
- **예제 수**: 1개

#### 2. `nli_statements_message`
- **목적**: 각 진술문이 컨텍스트에 지지되는지 검증
- **입력**: statement, retrieved_contexts
- **출력**: verdict (Supported/Not Supported/Noncommittal)
- **예제 수**: 3개

**핵심**: Faithfulness = (Supported 개수) / (전체 진술문 수)

---

### 📊 AnswerRelevancy

#### 1. `question_generation_prompt`
- **목적**: 답변으로부터 질문 역생성
- **입력**: response
- **출력**: List of generated questions (3-5개)
- **예제 수**: 1-2개

**핵심**: 생성된 질문과 원본 질문의 유사도로 relevancy 측정

---

### 📊 ContextPrecision

#### 1. `context_precision_prompt`
- **목적**: 각 컨텍스트가 답변 생성에 유용했는지 판단
- **입력**: question, answer, context
- **출력**: verdict (1=useful, 0=not useful)
- **예제 수**: 3개

**특징**: 유용한 예제 2개 + 유용하지 않은 예제 1개

**핵심**: ContextPrecision = (유용한 컨텍스트) / (전체 컨텍스트)

---

### 📊 ContextRecall

#### 1. `context_recall_prompt`
- **목적**: 정답의 각 문장이 컨텍스트에서 찾을 수 있는지 확인
- **입력**: ground_truth_sentence, retrieved_contexts
- **출력**: attributable (1=found, 0=not found)
- **예제 수**: 2-3개

**핵심**: ContextRecall = (찾을 수 있는 문장) / (전체 정답 문장)

---

## 프롬프트 커스터마이징

### 방법 1: 언어만 변경 (권장)

**가장 쉬운 방법** - 2분 소요

```python
from ragas.metrics import ContextPrecision, Faithfulness, AnswerCorrectness

# ContextPrecision
cp = ContextPrecision()
cp.context_precision_prompt.language = "korean"

# Faithfulness
faith = Faithfulness()
faith.statement_prompt.language = "korean"
faith.nli_statements_message.language = "korean"

# AnswerCorrectness
ac = AnswerCorrectness()
ac.statement_generator_prompt.language = "korean"
ac.correctness_prompt.language = "korean"

metrics = [cp, faith, ac]
```

**효과**:
- Instruction이 한국어로 변경됨
- 예제는 영어로 유지 (GPT-4는 문제없이 이해)
- 2-5% 정확도 향상

---

### 방법 2: 예제 개수 줄이기 (비용 최적화)

**30% 토큰 절감**

```python
metric = ContextPrecision()

# 3개 예제를 2개로 줄이기
original = metric.context_precision_prompt.examples
metric.context_precision_prompt.examples = original[:2]
```

**효과**:
- LLM 호출당 ~150-200 토큰 절감
- 15 samples × 5 metrics = 약 15,000 토큰 절감
- 정확도는 거의 동일 유지

---

### 방법 3: 커스텀 예제 추가 (고급)

**완전한 도메인 특화** - 4-6시간 소요

```python
from ragas.testset.prompts import QAC, Verification

# 한국 경제 특화 예제
korean_example = (
    QAC(
        question="2023년 GDP 성장률은?",
        context="2023년 경제성장률은 1.6%로 전망됩니다.",
        answer="1.6%"
    ),
    Verification(
        reason="컨텍스트에서 2023년 GDP 성장률 1.6%를 명확히 제시",
        verdict=1
    )
)

metric = ContextPrecision()
metric.context_precision_prompt.examples = [korean_example]
```

**효과**:
- 5-10% 정확도 향상 (도메인 특화)
- 유지보수 부담 증가
- RAGAS 업데이트 시 깨질 수 있음

---

## 실전 권장사항

### 🎯 대부분의 경우

**방법 1 (언어 변경)만 적용하세요**:
- 노력: 최소 (2분)
- 효과: 충분 (2-5% 향상)
- 리스크: 없음

### 💰 비용이 중요한 경우

**방법 1 + 방법 2 조합**:
- 언어를 한국어로 변경
- 예제 개수를 3개 → 2개로 축소
- 총 효과: 3-5% 향상 + 30% 비용 절감

### 🔬 최고 정확도가 필요한 경우

**방법 1 + 방법 3 조합**:
- 언어를 한국어로 변경
- 2-3개의 한국 경제 특화 예제 작성
- 총 효과: 10-15% 향상 가능
- 단, 6+ 시간 투자 필요

---

## 프롬프트 전체 목록

### AnswerCorrectness
- `statement_generator_prompt` ✅ 실제 프롬프트
- `correctness_prompt` ✅ 실제 프롬프트
- `adapt_prompts` 🔧 메서드
- `get_prompts` 🔧 메서드
- `set_prompts` 🔧 메서드
- `load_prompts` 🔧 메서드
- `save_prompts` 🔧 메서드

### Faithfulness
- `statement_prompt` ✅ 실제 프롬프트
- `nli_statements_message` ✅ 실제 프롬프트
- + 공통 메서드들

### AnswerRelevancy
- `question_generation_prompt` ✅ 실제 프롬프트
- + 공통 메서드들

### ContextPrecision
- `context_precision_prompt` ✅ 실제 프롬프트
- + 공통 메서드들

### ContextRecall
- `context_recall_prompt` ✅ 실제 프롬프트
- + 공통 메서드들

---

## FAQ

**Q: 모든 메트릭의 언어를 변경해야 하나요?**
A: 일관성을 위해 모두 변경하는 것을 권장합니다.

**Q: 예제는 영어로 두어도 되나요?**
A: 네! GPT-4는 multilingual이라 영어 예제도 잘 이해합니다.

**Q: 커스텀 예제 만들기가 너무 어려워요**
A: 대부분의 경우 언어만 변경해도 충분합니다. 커스텀 예제는 선택사항입니다.

**Q: save_prompts로 저장한 파일은 어떻게 사용하나요?**
A: 팀원에게 공유하거나 프로덕션 배포 시 load_prompts로 불러옵니다.

---

**RAGAS 버전**: Compatible with v0.1.x+
