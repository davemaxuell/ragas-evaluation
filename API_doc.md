## 금융 문서 RAG API 명세

이 문서는 `rag_api/src/main.py` 기반의 신버전 API 명세입니다. 금융 문서를 대상으로 벡터 검색(+ 리랭킹)과 LLM 답변 생성을 제공합니다.

### 기본 정보
- **Base URL (예시)**: `http://210.222.65.87:39050`
- **콘텐츠 타입**: `application/json`
- **인증**: 없음 (사내망 전제)
- **컬렉션명**: `article_store`

### 공통 동작 개요
- 임베딩 검색으로 상위 후보를 조회한 뒤, 교차 인코더 리랭커로 재정렬합니다.
- 사용자가 k개 결과를 요청하면 내부적으로 2k(최대 100) 개를 우선 검색한 후 리랭킹하여 상위 k개만 반환/활용합니다.
- 리랭커 모델: `BAAI/bge-reranker-v2-m3` (환경 변수 `RE_RANKER_MODEL`로 변경 가능). 로딩 실패 시 벡터 점수만 사용합니다.

## 1) 문서 검색: POST /search
금융 문서에서 쿼리 기반으로 상위 결과를 반환합니다.

요청 바디
```json
{
  "query": "문자열",
  "k": 5
}
```

응답 바디
```json
{
  "query": "문자열",
  "took_ms": 12,
  "results": [
    {
      "id": "string",
      "score": 0.8421,
      "text": "문서 내용 일부...",
      "original_file_path": "/path/to/source.pdf"
    }
  ]
}
```

비고
- `score`는 리랭커 점수(리랭커 비활성 시 벡터 점수)를 의미합니다.

예시 (curl)
```bash
curl -X POST "http://210.222.65.87:39050/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"통화정책 방향","k":5}'
```

## 2) 답변 생성(동기): POST /answer
상위 검색 결과를 컨텍스트로 사용해 LLM 답변을 한 번에 반환합니다.

요청 바디
```json
{
  "message": "질문 내용",
  "k": 5,
  "model": "gpt-4.1-mini"
}
```

설명
- **message**: 질문 내용 (필수)
- **k**: 검색에 사용할 상위 문서 개수 (기본값 5)
- **model** (선택):
  - 미지정 시 환경 변수 `DEFAULT_LLM_MODEL` 사용 (기본 `"gpt4"`)
  - 지원 예시
    - `"gpt4"`: OpenAI GPT-4 계열 기본 설정 (내부적으로 GPT-4 계열 모델 사용, Chat Completions API)
    - `"gpt-4o"`, `"gpt-4.1"`, `"gpt-4.1-mini"`, `"gpt-3.5-turbo"` 등 `gpt`로 시작하는 OpenAI Chat Completions 모델명
    - `"gpt-5"`, `"gpt-5.2"` 등 `gpt-5`로 시작하는 OpenAI Responses API 모델명 (`/v1/responses` 사용)
    - `"vllm"`: 내부 vLLM 서버 (`bllossom_3B` 모델)
    - `"bllossom"`: Bllossom API (`bllossom_70b` 모델)
    - `"hcx007"`: HyperCLOVA HCX-007 추론(Thinking) 모델 (CLOVA Studio Chat Completions v3 API 사용)

응답 바디
```json
{
  "type": "answer",
  "data": {
    "content": "LLM 생성 답변",
    "sources": [
      {
        "id": "string",
        "score": 0.9123,
        "text": "문서 내용 일부...",
        "original_file_path": "/path/to/source.pdf"
      }
    ]
  }
}
```

예시 (curl)
```bash
# 1) 기본 GPT-4 계열 (DEFAULT_LLM_MODEL=gpt4)
curl -X POST "http://210.222.65.87:39050/answer" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"gpt4"}'

# 2) 특정 OpenAI GPT 모델 지정 (예: gpt-4.1-mini)
curl -X POST "http://210.222.65.87:39050/answer" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"gpt-4.1-mini"}'

# 3) GPT-3.5-turbo 사용
curl -X POST "http://210.222.65.87:39050/answer" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"gpt-3.5-turbo"}'

# 4) GPT-5.2 사용 (Responses API 경유)
curl -X POST "http://210.222.65.87:39050/answer" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"gpt-5.2"}'

# 5) HyperCLOVA HCX-007 사용
curl -X POST "http://210.222.65.87:39050/answer" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"hcx007"}'

# 6) 내부 vLLM 서버 사용
curl -X POST "http://210.222.65.87:39050/answer" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"vllm"}'

# 7) Bllossom API 사용
curl -X POST "http://210.222.65.87:39050/answer" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"bllossom"}'
```

## 3) 답변 생성(스트리밍): POST /answer-stream
LLM 답변을 스트리밍(서버센트 이벤트 형식 JSON 라인)으로 반환합니다.

요청 바디
```json
{
  "message": "질문 내용",
  "k": 5,
  "model": "gpt-4.1-mini"
}
```

스트리밍 이벤트 형식(각 줄은 개별 JSON)
- `model_start`: `{ "type":"model_start", "data": { "message": "..." } }`
- `model_chunk`: `{ "type":"model_chunk", "data": { "content": "..." } }`
- `model_complete`: `{ "type":"model_complete", "data": { "sources": [ ... ] } }`
- `error`: `{ "type":"error", "data": { "error": "..." } }`

예시 (curl)
```bash
# 1) 기본 GPT-4 계열 (DEFAULT_LLM_MODEL=gpt4)
curl -N -X POST "http://210.222.65.87:39050/answer-stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"gpt4"}'

# 2) 특정 OpenAI GPT 모델 지정 (예: gpt-4.1-mini)
curl -N -X POST "http://210.222.65.87:39050/answer-stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"gpt-4.1-mini"}'

# 3) GPT-3.5-turbo 사용
curl -N -X POST "http://210.222.65.87:39050/answer-stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"gpt-3.5-turbo"}'

# 4) GPT-5.2 사용 (Responses API 경유)
curl -N -X POST "http://210.222.65.87:39050/answer-stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"gpt-5.2"}'

# 5) HyperCLOVA HCX-007 사용
curl -N -X POST "http://210.222.65.87:39050/answer-stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"hcx007"}'

# 6) 내부 vLLM 서버 사용
curl -N -X POST "http://210.222.65.87:39050/answer-stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"vllm"}'

# 7) Bllossom API 사용
curl -N -X POST "http://210.222.65.87:39050/answer-stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"최근 통화정책의 핵심 포인트 요약해줘","k":5,"model":"bllossom"}'
```

## 4) 헬스체크: GET /health
서비스 및 컬렉션 상태를 확인합니다.

응답 예시
```json
{
  "status": "healthy",
  "model": "BAAI/bge-m3",
  "collection_info": {
    "points_count": 12345,
    "vector_size": 1024,
    "distance_metric": "Cosine"
  }
}
```

## 5) 서비스 정보: GET /info
배포/설정 정보를 간략히 제공합니다.

응답 예시
```json
{
  "service_name": "금융 문서 RAG API",
  "version": "1.0.0",
  "embedding_model": "BAAI/bge-m3",
  "collection_name": "article_store",
  "qdrant_url": "http://rag_database:6333"
}
```

## 에러 응답 공통 형식
```json
{ "type": "error", "data": { "error": "메시지" } }
```

## 환경 변수 요약
- `QDRANT_URL`: Qdrant 접속 URL (예: `http://rag_database:6333`)
- `EMBEDDING_MODEL`: 문서 임베딩 모델 (기본 `BAAI/bge-m3`)
- `MODEL_DEVICE`: `cuda` 또는 `cpu`
- `RE_RANKER_MODEL`: 리랭커 모델명 (기본 `BAAI/bge-reranker-v2-m3`)
- `DEFAULT_LLM_MODEL`: `/answer`, `/answer-stream`에서 기본 사용할 LLM 프로바이더/모델
  - 기본값: `"gpt4"` (OpenAI GPT-4 계열, Chat Completions API 사용)
  - 예시 값: `"gpt4"`, `"gpt-4o"`, `"gpt-4.1"`, `"gpt-4.1-mini"`, `"gpt-3.5-turbo"`, `"gpt-5.2"`
    - `"gpt-5"`로 시작하는 값은 내부적으로 OpenAI Responses API(`/v1/responses`)를 사용


