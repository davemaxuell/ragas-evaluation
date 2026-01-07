# Default RAGAS Instructions Reference

이 폴더는 RAGAS 메트릭의 **기본 instruction**을 참조용으로 저장합니다.
커스텀 프롬프트 작성 시 참고하세요.

## 파일 목록

| 파일 | 메트릭 | 프롬프트 수 |
|------|--------|------------|
| `faithfulness.json` | Faithfulness | 2개 |
| `answer_correctness.json` | AnswerCorrectness | 2개 |
| `context_precision.json` | ContextPrecision | 1개 |
| `context_recall.json` | ContextRecall | 1개 |
| `answer_relevancy.json` | AnswerRelevancy | 1개 |

## 사용법

1. 기본 instruction 확인
2. 필요에 따라 수정
3. `custom_prompts/` 폴더의 해당 파일에 적용

## 주의

이 파일들은 **참조용**입니다. 실제 커스터마이징은 상위 폴더(`custom_prompts/`)에서 하세요.
