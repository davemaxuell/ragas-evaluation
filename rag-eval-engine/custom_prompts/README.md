# Custom Prompts Configuration

이 디렉토리에는 RAGAS 메트릭의 커스텀 프롬프트 설정 파일들이 있습니다.

## 사용 방법

### 1. 커스텀 프롬프트 활성화

원하는 메트릭의 JSON 파일에서 `"enabled": true`로 변경:

```json
{
  "metric_name": "faithfulness",
  "enabled": true,  // false → true 변경
  "language": "korean",
  ...
}
```

### 2. Instruction 변경

프롬프트의 instruction을 수정:

```json
"prompts": {
  "statement_generator_prompt": {
    "instruction": "주어진 답변을 개별 진술문으로 분해하세요...",
    "examples": []
  }
}
```

### 3. 언어만 변경 (가장 간단)

```json
{
  "enabled": true,
  "language": "korean"  // RAGAS가 자동으로 한국어로 변환
}
```

## 파일 목록

| 파일명 | 메트릭 | 설명 |
|--------|--------|------|
| `faithfulness.json` | Faithfulness | 답변이 컨텍스트에 충실한지 |
| `answer_correctness.json` | AnswerCorrectness | 답변이 정답과 일치하는지 |
| `context_precision.json` | ContextPrecision | 검색된 컨텍스트가 유용한지 |
| `context_recall.json` | ContextRecall | 필요한 정보를 모두 검색했는지 |
| `answer_relevancy.json` | AnswerRelevancy | 답변이 질문과 관련있는지 |

## JSON 구조

```json
{
  "metric_name": "string",        // 메트릭 이름
  "enabled": false,               // true면 커스텀 프롬프트 사용
  "language": "english",          // 프롬프트 언어
  "prompts": {
    "prompt_name": {
      "instruction": "",          // 빈 문자열이면 기본값 사용
      "examples": []              // 빈 배열이면 기본값 사용
    }
  },
  "_comments": {}                 // 설명 (무시됨)
}
```

## 권장 사용법

### 1단계: 언어만 변경
```json
"enabled": true,
"language": "korean"
```

### 2단계: Instruction 커스터마이징
```json
"prompts": {
  "statement_generator_prompt": {
    "instruction": "한국 경제 데이터에 맞는 커스텀 명령어...",
    "examples": []
  }
}
```

### 3단계: 예제 추가 (고급)
```json
"examples": [
  {
    "input": {...},
    "output": {...}
  }
]
```

## 주의사항

- `enabled: false`면 해당 파일은 완전히 무시됩니다
- `instruction: ""`면 기본 instruction을 사용합니다
- `examples: []`면 기본 예제를 사용합니다
- 잘못된 JSON 형식이면 기본값으로 폴백합니다

---

## Custom Mode (Standalone Scripts)

`score_json.py` 및 `score_json_multiturn.py` 스크립트에서 사용하려면:

### 활성화 방법

```json
{
  "metric_name": "faithfulness",
  "enabled": true,
  "custom_mode": true,                    // ← 반드시 필요!
  "custom_prompt": "Your prompt here..."  // ← 전체 프롬프트 템플릿
}
```

### custom_prompt 템플릿 변수

| 변수 | 설명 |
|------|------|
| `{question}` | 사용자 질문 |
| `{answer}` | AI 답변 |
| `{context}` | 검색된 컨텍스트 |
| `{ground_truth}` | 정답 (AnswerCorrectness용) |

### 예시

```json
{
  "enabled": true,
  "custom_mode": true,
  "custom_prompt": "질문: {question}\n답변: {answer}\n컨텍스트: {context}\n\n답변이 컨텍스트에 근거하면 1, 아니면 0을 출력하세요."
}
```

### API vs Standalone 차이

| 항목 | API (`api.py`) | Standalone (`score_json*.py`) |
|------|----------------|-------------------------------|
| 설정 | `enabled`, `prompts` | `enabled`, `custom_mode`, `custom_prompt` |
| 프롬프트 | RAGAS 기본 구조 수정 | 전체 프롬프트 직접 작성 |
| 출력 | 0.0 ~ 1.0 (연속) | 0 또는 1 (이진) |

