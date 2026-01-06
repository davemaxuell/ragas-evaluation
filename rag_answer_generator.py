"""
RAG Answer Generator Script
============================
이 스크립트는 2-가 RAG(멀티모달).json의 각 질문에 대해
RAG API를 호출하여 답변과 검색된 컨텍스트를 가져오고,
RAG_answer와 RAG_context 필드를 추가합니다.
"""

import json
import requests
import time
from pathlib import Path
from tqdm import tqdm

# RAG API Configuration
RAG_API_BASE_URL = "http://210.222.65.87:39050"
DEFAULT_K = 5
DEFAULT_MODEL = "gpt-3.5-turbo"

# Supported models (from API_doc.md)
SUPPORTED_MODELS = [
    # OpenAI Chat Completions API models
    "gpt4",           # GPT-4 계열 기본 설정
    "gpt-4o",         # GPT-4o
    "gpt-4.1",        # GPT-4.1
    "gpt-4.1-mini",   # GPT-4.1-mini
    "gpt-3.5-turbo",  # GPT-3.5-turbo
    # OpenAI Responses API models (gpt-5 series)
    "gpt-5",          # GPT-5 (Responses API)
    "gpt-5.2",        # GPT-5.2 (Responses API)
    # Other providers
    "vllm",           # 내부 vLLM 서버 (bllossom_3B 모델)
    "bllossom",       # Bllossom API (bllossom_70b 모델)
    "hcx007",         # HyperCLOVA HCX-007 추론 모델
]

def call_rag_answer_api(question: str, k: int = DEFAULT_K, model: str = DEFAULT_MODEL) -> dict:
    """
    RAG API의 /answer 엔드포인트를 호출하여 답변과 소스를 가져옵니다.
    
    Args:
        question: 질문 텍스트
        k: 검색할 문서 수 (기본값: 5)
        model: 사용할 LLM 모델 (기본값: gpt-4.1-mini)
    
    Returns:
        dict: {'answer': str, 'sources': list}
    """
    url = f"{RAG_API_BASE_URL}/answer"
    payload = {
        "message": question,
        "k": k,
        "model": model
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        if result.get("type") == "answer":
            data = result.get("data", {})
            return {
                "answer": data.get("content", ""),
                "sources": data.get("sources", [])
            }
        elif result.get("type") == "error":
            return {
                "answer": f"Error: {result.get('data', {}).get('error', 'Unknown error')}",
                "sources": []
            }
    except requests.exceptions.RequestException as e:
        return {
            "answer": f"Request Error: {str(e)}",
            "sources": []
        }
    except json.JSONDecodeError:
        return {
            "answer": "Error: Invalid JSON response",
            "sources": []
        }
    
    return {"answer": "", "sources": []}


def format_sources_for_context(sources: list) -> list:
    """
    RAG API에서 받은 sources를 RAG_context 형식으로 변환합니다.
    
    Args:
        sources: RAG API에서 반환된 sources 리스트
    
    Returns:
        list: 변환된 컨텍스트 리스트
    """
    formatted_contexts = []
    for source in sources:
        formatted_context = {
            "id": source.get("id", ""),
            "source_document": source.get("original_file_path", ""),
            "score": source.get("score", 0.0),
            "content": source.get("text", "")
        }
        formatted_contexts.append(formatted_context)
    return formatted_contexts


def process_json_file(input_path: str, output_path: str = None, 
                      k: int = DEFAULT_K, model: str = DEFAULT_MODEL,
                      delay: float = 0.5):
    """
    JSON 파일의 모든 질문에 대해 RAG API를 호출하고 결과를 추가합니다.
    
    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로 (None이면 input_path에 _RAG_augmented 추가)
        k: 검색할 문서 수
        model: 사용할 LLM 모델
        delay: API 호출 간 대기 시간 (초)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_RAG_augmented{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # JSON 파일 로드
    print(f"📂 Loading JSON file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get("items", [])
    total_turns = sum(len(item.get("turns", [])) for item in items)
    
    print(f"📊 Total items: {len(items)}, Total turns: {total_turns}")
    print(f"🤖 Using model: {model}, k: {k}")
    print(f"🔗 RAG API: {RAG_API_BASE_URL}")
    print("-" * 60)
    
    # 진행 상황 추적
    processed_count = 0
    error_count = 0
    
    # tqdm으로 진행률 표시
    with tqdm(total=total_turns, desc="Processing questions") as pbar:
        for item in items:
            turns = item.get("turns", [])
            
            for turn in turns:
                question = turn.get("question", "")
                
                if not question:
                    pbar.update(1)
                    continue
                
                # RAG API 호출
                result = call_rag_answer_api(question, k=k, model=model)
                
                # 결과 추가 (no filtering, use all chunks)
                formatted_context = format_sources_for_context(result["sources"])
                turn["RAG_context"] = formatted_context
                turn["RAG_answer"] = result["answer"]
                
                if result["answer"].startswith("Error") or result["answer"].startswith("Request Error"):
                    error_count += 1
                else:
                    processed_count += 1
                
                pbar.update(1)
                
                # API 부하 방지를 위한 딜레이
                time.sleep(delay)
    
    # 결과 저장
    print("-" * 60)
    print(f"💾 Saving results to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Processing complete!")
    print(f"   - Successfully processed: {processed_count}")
    print(f"   - Errors: {error_count}")
    print(f"   - Output file: {output_path}")
    
    return str(output_path)


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG API를 사용하여 JSON 파일의 질문에 답변을 추가합니다."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="2-가 RAG(멀티모달).json",
        help="입력 JSON 파일 경로 (기본값: 2-가 RAG(멀티모달).json)"
    )
    parser.add_argument(
        "-o", "--output",
        help="출력 JSON 파일 경로 (기본값: 입력파일명_RAG_augmented.json)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=DEFAULT_K,
        help=f"검색할 문서 수 (기본값: {DEFAULT_K})"
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help=f"""사용할 LLM 모델 (기본값: {DEFAULT_MODEL}). 지원 모델:
  - gpt4, gpt-4o, gpt-4.1, gpt-4.1-mini, gpt-3.5-turbo (OpenAI Chat Completions)
  - gpt-5, gpt-5.2 (OpenAI Responses API)
  - vllm (내부 vLLM 서버), bllossom (Bllossom API), hcx007 (HyperCLOVA)"""
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=0.5,
        help="API 호출 간 대기 시간(초) (기본값: 0.5)"
    )
    
    args = parser.parse_args()
    
    # 파일 경로 확인
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        return 1
    
    # 처리 실행
    try:
        output_file = process_json_file(
            input_path=str(input_path),
            output_path=args.output,
            k=args.top_k,
            model=args.model,
            delay=args.delay
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
