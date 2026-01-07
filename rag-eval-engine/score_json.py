"""
RAG 멀티모달 JSON 평가 스크립트
- 커스텀 프롬프트를 사용하여 5개 메트릭 평가
- Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
- RAG_answer를 평가 대상 답변으로 사용, answer를 ground_truth로 사용
- RAG_context를 검색된 컨텍스트로 사용
- Google Gemini API 사용
"""

import json
import os
import re
import time
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Rate limiting 설정
API_DELAY_SECONDS = 1  # API 호출 간 딜레이 (초)
MAX_RETRIES = 3  # 최대 재시도 횟수
RETRY_DELAY_BASE = 5  # 재시도 시 기본 대기 시간 (초)

# .env 파일에서 환경변수 로드
load_dotenv()

# Gemini API 키
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# Gemini 모델명 (환경변수에서 로드, 기본값: gemini-2.0-flash)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# Gemini 클라이언트 설정
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# 커스텀 프롬프트 디렉토리
CUSTOM_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "custom_prompts")

# ============================================================
# 커스텀 프롬프트 로드 함수들
# ============================================================

def load_custom_prompt(metric_name: str) -> str:
    """JSON 파일에서 커스텀 프롬프트를 로드합니다."""
    json_path = os.path.join(CUSTOM_PROMPTS_DIR, f"{metric_name}.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"커스텀 프롬프트 파일을 찾을 수 없습니다: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if not config.get("enabled", False) or not config.get("custom_mode", False):
        raise ValueError(f"{metric_name}의 custom_mode가 비활성화되어 있습니다.")
    
    custom_prompt = config.get("custom_prompt", "")
    if not custom_prompt:
        raise ValueError(f"{metric_name}의 custom_prompt가 비어있습니다.")
    
    return custom_prompt


# ============================================================
# API 호출 헬퍼 함수 (Rate Limiting 및 Retry)
# ============================================================

def call_api_with_retry(prompt: str, max_tokens: int = 5):
    """Gemini API 호출을 재시도 로직과 함께 실행합니다."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(API_DELAY_SECONDS)  # Rate limiting 딜레이
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens
                )
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"    [Rate Limit] {wait_time}초 대기 후 재시도... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("최대 재시도 횟수 초과")


# ============================================================
# 평가 함수들
# ============================================================

def evaluate_faithfulness(question: str, answer: str, context: str, prompt_template: str) -> int:
    """Faithfulness 평가: 답변이 컨텍스트에 근거하는가? (0 또는 1)"""
    try:
        formatted_prompt = prompt_template.format(
            question=question,
            answer=answer,
            context=context
        )
        
        result_text = call_api_with_retry(formatted_prompt, max_tokens=5)
        
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"  [경고] Faithfulness 예상치 못한 응답: {result_text}")
            return 0
            
    except Exception as e:
        print(f"  [오류] Faithfulness 평가 실패: {e}")
        return 0


def evaluate_answer_relevancy(question: str, answer: str, prompt_template: str) -> int:
    """AnswerRelevancy 평가: 답변이 질문과 관련있는가? (0 또는 1)"""
    try:
        formatted_prompt = prompt_template.format(
            question=question,
            answer=answer
        )
        
        result_text = call_api_with_retry(formatted_prompt, max_tokens=5)
        
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"  [경고] AnswerRelevancy 예상치 못한 응답: {result_text}")
            return 0
            
    except Exception as e:
        print(f"  [오류] AnswerRelevancy 평가 실패: {e}")
        return 0


def evaluate_context_precision(question: str, answer: str, context: str, prompt_template: str) -> int:
    """ContextPrecision 평가: 검색된 컨텍스트가 유용한가? (0 또는 1)"""
    try:
        formatted_prompt = prompt_template.format(
            question=question,
            answer=answer,
            context=context
        )
        
        result_text = call_api_with_retry(formatted_prompt, max_tokens=5)
        
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"  [경고] ContextPrecision 예상치 못한 응답: {result_text}")
            return 0
            
    except Exception as e:
        print(f"  [오류] ContextPrecision 평가 실패: {e}")
        return 0


def evaluate_context_recall(question: str, answer: str, context: str, prompt_template: str) -> int:
    """ContextRecall 평가: 필요한 정보를 모두 검색했는가? (0 또는 1)"""
    try:
        formatted_prompt = prompt_template.format(
            question=question,
            answer=answer,
            context=context
        )
        
        # Reasoning이 포함되므로 max_tokens 증가
        result_text = call_api_with_retry(formatted_prompt, max_tokens=500)
        
        # 결과 분석 (마지막 줄 확인)
        lines = result_text.strip().split('\n')
        last_line = lines[-1].strip()
        
        if "1" in last_line:
            return 1
        elif "0" in last_line:
            return 0
        else:
            print(f"  [경고] ContextRecall 예상치 못한 응답: {last_line} (Full: {result_text[:50]}...)")
            return 0
            
    except Exception as e:
        print(f"  [오류] ContextRecall 평가 실패: {e}")
        return 0


def evaluate_answer_correctness(answer: str, ground_truth: str, prompt_template: str) -> float:
    """AnswerCorrectness 평가: 답변이 정답과 일치하는가? (F1 score 0.0~1.0)"""
    try:
        formatted_prompt = prompt_template.format(
            answer=answer,
            ground_truth=ground_truth
        )
        
        # Reasoning 및 Conclusion 포함되므로 max_tokens 증가
        result_text = call_api_with_retry(formatted_prompt, max_tokens=500)
        
        # F1 Score 추출 (Regex 사용)
        # 패턴: "(최종 F1-Score: 0.8)" 또는 "F1-Score: 0.8" 등
        match = re.search(r"F1-Score[:\s]*([0-1]\.\d+|0|1)", result_text, re.IGNORECASE)
        
        if match:
            f1_score = float(match.group(1))
            # 0.0 ~ 1.0 범위 확인
            return max(0.0, min(1.0, f1_score))
        else:
            # 숫자가 명시적으로 없으면 0.0 처리 (Strict)
            print(f"    [경고] AnswerCorrectness 점수 추출 실패: {result_text[-50:]}")
            return 0.0
            
    except Exception as e:
        print(f"  [오류] AnswerCorrectness 평가 실패: {e}")
        return 0.0  # 오류 시 0점


# ============================================================
# 메인 평가 함수
# ============================================================

def evaluate_json_file(input_path: str, output_path: str = None):
    """
    JSON 파일을 읽어 각 항목을 평가하고 점수를 추가합니다.
    
    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로 (없으면 _scored.json으로 저장)
    """
    
    # 출력 경로 설정 (모델명 포함)
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        # 기존 scored 파일이면 모델명만 업데이트
        if "_scored" in base:
            # 기존 모델명 제거 후 새 모델명 추가
            base = base.split("_scored")[0]
        output_path = f"{base}_scored({GEMINI_MODEL_NAME}){ext}"
    
    print("=" * 60)
    print("RAG 멀티모달 JSON 평가 스크립트 (Gemini)")
    print("=" * 60)
    
    # 1. 커스텀 프롬프트 로드
    print("\n[1/4] 커스텀 프롬프트 로드 중...")
    try:
        faithfulness_prompt = load_custom_prompt("faithfulness")
        print("  ✓ Faithfulness 프롬프트 로드 완료")
        
        answer_relevancy_prompt = load_custom_prompt("answer_relevancy")
        print("  ✓ AnswerRelevancy 프롬프트 로드 완료")
        
        context_precision_prompt = load_custom_prompt("context_precision")
        print("  ✓ ContextPrecision 프롬프트 로드 완료")
        
        context_recall_prompt = load_custom_prompt("context_recall")
        print("  ✓ ContextRecall 프롬프트 로드 완료")
        
        answer_correctness_prompt = load_custom_prompt("answer_correctness")
        print("  ✓ AnswerCorrectness 프롬프트 로드 완료 (RAG_answer vs answer)")
        
    except Exception as e:
        print(f"  ✗ 프롬프트 로드 실패: {e}")
        return
    
    # 2. JSON 파일 로드
    print(f"\n[2/4] JSON 파일 로드 중: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get("items", [])
        total_items = len(items)
        print(f"  ✓ {total_items}개 항목 로드 완료")
        
    except Exception as e:
        print(f"  ✗ JSON 로드 실패: {e}")
        return
    
    # 3. 각 항목 평가
    print(f"\n[3/4] 평가 시작... (모델: {GEMINI_MODEL_NAME})")
    
    # 통계 초기화
    scores_summary = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
        "answer_correctness": []
    }
    
    for idx, item in enumerate(items):
        item_id = item.get("item_id", idx + 1)
        turns = item.get("turns", [])
        
        if not turns:
            print(f"  [{idx + 1}/{total_items}] Item {item_id}: 턴 데이터 없음, 스킵")
            continue
        
        # 첫 번째 턴 사용 (single_turn 가정)
        turn = turns[0]
        question = turn.get("question", "")
        ground_truth = turn.get("answer", "")  # 원본 answer를 ground_truth로 사용
        rag_answer = turn.get("RAG_answer", "")  # RAG 생성 답변을 평가 대상으로 사용
        rag_contexts = turn.get("RAG_context", [])  # RAG 검색 컨텍스트 사용
        
        # RAG_answer가 없으면 스킵
        if not rag_answer:
            print(f"  [{idx + 1}/{total_items}] Item {item_id}: RAG_answer 없음, 스킵")
            continue
        
        # 컨텍스트 결합 (RAG_context는 딕셔너리 리스트로 content 필드 추출)
        context_parts = []
        if isinstance(rag_contexts, list):
            for ctx in rag_contexts:
                if isinstance(ctx, dict):
                    content = ctx.get("content", "")
                    if content:
                        context_parts.append(content)
                elif isinstance(ctx, str):
                    context_parts.append(ctx)
        context = "\n\n".join(context_parts)
        
        print(f"  [{idx + 1}/{total_items}] Item {item_id} 평가 중...")
        
        # 각 메트릭 평가 (RAG_answer를 평가 대상으로, answer를 ground_truth로 사용)
        faithfulness_score = evaluate_faithfulness(question, rag_answer, context, faithfulness_prompt)
        answer_relevancy_score = evaluate_answer_relevancy(question, rag_answer, answer_relevancy_prompt)
        context_precision_score = evaluate_context_precision(question, ground_truth, context, context_precision_prompt)
        context_recall_score = evaluate_context_recall(question, ground_truth, context, context_recall_prompt)
        # AnswerCorrectness: RAG_answer와 ground_truth(answer) 비교
        answer_correctness_score = evaluate_answer_correctness(rag_answer, ground_truth, answer_correctness_prompt)
        
        # 점수 저장
        turn["scores"] = {
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score,
            "context_precision": context_precision_score,
            "context_recall": context_recall_score,
            "answer_correctness": answer_correctness_score
        }
        
        # 통계 수집
        scores_summary["faithfulness"].append(faithfulness_score)
        scores_summary["answer_relevancy"].append(answer_relevancy_score)
        scores_summary["context_precision"].append(context_precision_score)
        scores_summary["context_recall"].append(context_recall_score)
        scores_summary["answer_correctness"].append(answer_correctness_score)
        
        print(f"    F={faithfulness_score}, AR={answer_relevancy_score}, CP={context_precision_score}, CR={context_recall_score}, AC={answer_correctness_score}")
    
    # 4. 결과 저장
    print(f"\n[4/4] 결과 저장 중: {output_path}")
    
    # 평가 메타데이터 추가
    data["evaluation_results"] = {
        "evaluated_at": datetime.now().isoformat(),
        "model": GEMINI_MODEL_NAME,
        "total_items": total_items,
        "metrics_used": ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"],
        "note": "RAG_answer is evaluated against answer (ground_truth). RAG_context is used for context metrics.",
        "summary_scores": {
            "faithfulness": round(sum(scores_summary["faithfulness"]) / len(scores_summary["faithfulness"]), 3) if scores_summary["faithfulness"] else 0,
            "answer_relevancy": round(sum(scores_summary["answer_relevancy"]) / len(scores_summary["answer_relevancy"]), 3) if scores_summary["answer_relevancy"] else 0,
            "context_precision": round(sum(scores_summary["context_precision"]) / len(scores_summary["context_precision"]), 3) if scores_summary["context_precision"] else 0,
            "context_recall": round(sum(scores_summary["context_recall"]) / len(scores_summary["context_recall"]), 3) if scores_summary["context_recall"] else 0,
            "answer_correctness": round(sum(scores_summary["answer_correctness"]) / len(scores_summary["answer_correctness"]), 3) if scores_summary["answer_correctness"] else 0
        }
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 저장 완료")
    except Exception as e:
        print(f"  ✗ 저장 실패: {e}")
        return
    
    # 5. 최종 결과 출력
    print("\n" + "=" * 60)
    print("평가 완료!")
    print("=" * 60)
    print(f"\n총 평가 항목: {total_items}개")
    print("\n평균 점수:")
    print(f"  ✓ Faithfulness:       {data['evaluation_results']['summary_scores']['faithfulness']:.3f}")
    print(f"  ✓ AnswerRelevancy:    {data['evaluation_results']['summary_scores']['answer_relevancy']:.3f}")
    print(f"  ✓ ContextPrecision:   {data['evaluation_results']['summary_scores']['context_precision']:.3f}")
    print(f"  ✓ ContextRecall:      {data['evaluation_results']['summary_scores']['context_recall']:.3f}")
    print(f"  ✓ AnswerCorrectness:  {data['evaluation_results']['summary_scores']['answer_correctness']:.3f} (RAG_answer vs ground_truth)")
    print(f"\n결과 파일: {output_path}")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    import sys
    
    # 기본 입력 파일 경로
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "2-가 RAG(멀티모달)_scored.json"
    )
    
    # 명령줄 인자로 파일 경로 지정 가능
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"오류: 파일을 찾을 수 없습니다: {input_file}")
        print(f"\n사용법: python {sys.argv[0]} <input_json_path>")
        sys.exit(1)
    
    # 평가 실행
    evaluate_json_file(input_file)
