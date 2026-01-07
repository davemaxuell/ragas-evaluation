"""
RAG 멀티턴 JSON 평가 스크립트
- 모든 turn을 순회하여 5개 메트릭 평가
- Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
- Multi-turn 아이템에 대해 AspectCritic 평가 추가
- Item별 평균 점수 계산
- RAG_answer를 평가 대상 답변으로 사용, answer를 ground_truth로 사용
- RAG_context를 검색된 컨텍스트로 사용
"""

import json
import os
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Rate limiting 설정
API_DELAY_SECONDS = 2  # API 호출 간 딜레이 (초)
MAX_RETRIES = 3  # 최대 재시도 횟수
RETRY_DELAY_BASE = 5  # 재시도 시 기본 대기 시간 (초)

# 두 번째 OpenAI API 키 (별도 rate limit)
OPENAI_API_KEY = "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"

# OpenAI 클라이언트
client = OpenAI(api_key=OPENAI_API_KEY)

# 커스텀 프롬프트 디렉토리
CUSTOM_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "custom_prompts")

# AspectCritic 프롬프트 (Multi-turn 대화 품질 평가)
ASPECT_CRITIC_PROMPT = """당신은 다중 턴 대화의 품질을 평가하는 전문가입니다.

아래는 사용자와 AI 간의 대화입니다:

{conversation}

다음 기준에 따라 대화의 전반적인 품질을 평가하세요:
1. 일관성 (Coherence): 대화가 논리적으로 연결되고 흐름이 자연스러운가?
2. 맥락 유지 (Context Retention): AI가 이전 턴의 내용을 잘 기억하고 활용하는가?
3. 점진적 정보 제공 (Progressive Information): 각 턴에서 새로운 정보가 적절히 추가되는가?

위 기준을 종합하여 대화가 고품질이면 1, 저품질이면 0을 출력하세요.
숫자만 출력하세요: 1 또는 0"""


# ============================================================
# 커스텀 프롬프트 로드 함수
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

def call_api_with_retry(api_func, *args, **kwargs):
    """API 호출을 재시도 로직과 함께 실행합니다."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(API_DELAY_SECONDS)  # Rate limiting 딜레이
            return api_func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"      [Rate Limit] {wait_time}초 대기 후 재시도... ({attempt + 1}/{MAX_RETRIES})")
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
        
        def api_call():
            return client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_completion_tokens=5
            )
        
        response = call_api_with_retry(api_call)
        result_text = response.choices[0].message.content.strip()
        
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"    [경고] Faithfulness 예상치 못한 응답: {result_text}")
            return 0
            
    except Exception as e:
        print(f"    [오류] Faithfulness 평가 실패: {e}")
        return 0


def evaluate_answer_relevancy(question: str, answer: str, prompt_template: str) -> int:
    """AnswerRelevancy 평가: 답변이 질문과 관련있는가? (0 또는 1)"""
    try:
        formatted_prompt = prompt_template.format(
            question=question,
            answer=answer
        )
        
        def api_call():
            return client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_completion_tokens=5
            )
        
        response = call_api_with_retry(api_call)
        result_text = response.choices[0].message.content.strip()
        
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"    [경고] AnswerRelevancy 예상치 못한 응답: {result_text}")
            return 0
            
    except Exception as e:
        print(f"    [오류] AnswerRelevancy 평가 실패: {e}")
        return 0


def evaluate_context_precision(question: str, answer: str, context: str, prompt_template: str) -> int:
    """ContextPrecision 평가: 검색된 컨텍스트가 유용한가? (0 또는 1)"""
    try:
        formatted_prompt = prompt_template.format(
            question=question,
            answer=answer,
            context=context
        )
        
        def api_call():
            return client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_completion_tokens=5
            )
        
        response = call_api_with_retry(api_call)
        result_text = response.choices[0].message.content.strip()
        
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"    [경고] ContextPrecision 예상치 못한 응답: {result_text}")
            return 0
            
    except Exception as e:
        print(f"    [오류] ContextPrecision 평가 실패: {e}")
        return 0


def evaluate_context_recall(question: str, answer: str, context: str, prompt_template: str) -> int:
    """ContextRecall 평가: 필요한 정보를 모두 검색했는가? (0 또는 1)"""
    try:
        formatted_prompt = prompt_template.format(
            question=question,
            answer=answer,
            context=context
        )
        
        def api_call():
            return client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_completion_tokens=5
            )
        
        response = call_api_with_retry(api_call)
        result_text = response.choices[0].message.content.strip()
        
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"    [경고] ContextRecall 예상치 못한 응답: {result_text}")
            return 0
            
    except Exception as e:
        print(f"    [오류] ContextRecall 평가 실패: {e}")
        return 0


def evaluate_answer_correctness(answer: str, ground_truth: str, prompt_template: str) -> float:
    """AnswerCorrectness 평가: 답변이 정답과 일치하는가? (F1 score 0.0~1.0)"""
    try:
        formatted_prompt = prompt_template.format(
            answer=answer,
            ground_truth=ground_truth
        )
        
        def api_call():
            return client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_completion_tokens=200
            )
        
        response = call_api_with_retry(api_call)
        result_text = response.choices[0].message.content.strip().upper()
        
        # TP, FP, FN 개수 카운트
        tp_count = result_text.count("TP")
        fp_count = result_text.count("FP")
        fn_count = result_text.count("FN")
        
        if tp_count == 0 and fp_count == 0 and fn_count == 0:
            return 1.0
        
        if tp_count == 0:
            return 1.0
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 1.0
        
        if precision + recall == 0:
            return 1.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return round(f1_score, 2)
            
    except Exception as e:
        print(f"    [오류] AnswerCorrectness 평가 실패: {e}")
        return 1.0


def evaluate_aspect_critic(turns: list) -> int:
    """AspectCritic 평가: 다중 턴 대화의 전반적인 품질 (0 또는 1)"""
    try:
        # 대화 내용을 문자열로 변환 (RAG_answer 사용)
        conversation_parts = []
        for turn in turns:
            q = turn.get("question", "")
            # RAG_answer를 우선 사용, 없으면 answer 사용
            a = turn.get("RAG_answer", turn.get("answer", ""))
            turn_id = turn.get("turn_id", "")
            conversation_parts.append(f"[{turn_id}]\n사용자: {q}\nAI: {a}")
        
        conversation = "\n\n".join(conversation_parts)
        
        formatted_prompt = ASPECT_CRITIC_PROMPT.format(conversation=conversation)
        
        def api_call():
            return client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_completion_tokens=5
            )
        
        response = call_api_with_retry(api_call)
        result_text = response.choices[0].message.content.strip()
        
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"    [경고] AspectCritic 예상치 못한 응답: {result_text}")
            return 0
            
    except Exception as e:
        print(f"    [오류] AspectCritic 평가 실패: {e}")
        return 0


# ============================================================
# 메인 평가 함수
# ============================================================

def evaluate_json_file(input_path: str, output_path: str = None):
    """
    JSON 파일을 읽어 각 항목의 모든 턴을 평가하고 점수를 추가합니다.
    Multi-turn 아이템에 대해서는 AspectCritic 평가도 추가합니다.
    """
    
    # 출력 경로 설정
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if base.endswith("_scored"):
            output_path = input_path
        else:
            output_path = f"{base}_scored{ext}"
    
    print("=" * 70)
    print("RAG 멀티턴 JSON 평가 스크립트 (5 Metrics + AspectCritic)")
    print("=" * 70)
    
    # 1. 커스텀 프롬프트 로드
    print("\n[1/4] 커스텀 프롬프트 로드 중...")
    try:
        faithfulness_prompt = load_custom_prompt("faithfulness")
        print("  ? Faithfulness 프롬프트 로드 완료")
        
        answer_relevancy_prompt = load_custom_prompt("answer_relevancy")
        print("  ? AnswerRelevancy 프롬프트 로드 완료")
        
        context_precision_prompt = load_custom_prompt("context_precision")
        print("  ? ContextPrecision 프롬프트 로드 완료")
        
        context_recall_prompt = load_custom_prompt("context_recall")
        print("  ? ContextRecall 프롬프트 로드 완료")
        
        answer_correctness_prompt = load_custom_prompt("answer_correctness")
        print("  ? AnswerCorrectness 프롬프트 로드 완료 (RAG_answer vs answer)")
        
        print("  ? AspectCritic 프롬프트 (내장) 준비 완료")
        
    except Exception as e:
        print(f"  ? 프롬프트 로드 실패: {e}")
        return
    
    # 2. JSON 파일 로드
    print(f"\n[2/4] JSON 파일 로드 중: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get("items", [])
        total_items = len(items)
        print(f"  ? {total_items}개 항목 로드 완료")
        
        # Single-turn vs Multi-turn 카운트
        single_turn_count = sum(1 for item in items if len(item.get("turns", [])) == 1)
        multi_turn_count = total_items - single_turn_count
        print(f"  - Single-turn: {single_turn_count}개")
        print(f"  - Multi-turn: {multi_turn_count}개")
        
    except Exception as e:
        print(f"  ? JSON 로드 실패: {e}")
        return
    
    # 3. 각 항목 평가
    print(f"\n[3/4] 평가 시작...")
    
    # 전체 통계 초기화
    all_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
        "answer_correctness": [],
        "aspect_critic": []
    }
    
    for idx, item in enumerate(items):
        item_id = item.get("item_id", idx + 1)
        turns = item.get("turns", [])
        num_turns = len(turns)
        
        if not turns:
            print(f"  [{idx + 1}/{total_items}] Item {item_id}: 턴 데이터 없음, 스킵")
            continue
        
        is_multi_turn = num_turns > 1
        turn_type = "Multi-turn" if is_multi_turn else "Single-turn"
        print(f"  [{idx + 1}/{total_items}] Item {item_id} ({turn_type}, {num_turns} turns) 평가 중...")
        
        # Item 레벨 점수 수집
        item_scores = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
            "answer_correctness": []
        }
        
        # 각 턴 평가
        for turn_idx, turn in enumerate(turns):
            turn_id = turn.get("turn_id", f"turn_{turn_idx + 1}")
            question = turn.get("question", "")
            ground_truth = turn.get("answer", "")  # 원본 answer를 ground_truth로 사용
            rag_answer = turn.get("RAG_answer", "")  # RAG 생성 답변을 평가 대상으로 사용
            rag_contexts = turn.get("RAG_context", [])  # RAG 검색 컨텍스트 사용
            
            # RAG_answer가 없으면 스킵
            if not rag_answer:
                print(f"    [{turn_id}] RAG_answer 없음, 스킵")
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
            
            print(f"    [{turn_id}] 평가 중...")
            
            # 각 메트릭 평가 (RAG_answer를 평가 대상으로, answer를 ground_truth로 사용)
            f_score = evaluate_faithfulness(question, rag_answer, context, faithfulness_prompt)
            ar_score = evaluate_answer_relevancy(question, rag_answer, answer_relevancy_prompt)
            cp_score = evaluate_context_precision(question, ground_truth, context, context_precision_prompt)
            cr_score = evaluate_context_recall(question, ground_truth, context, context_recall_prompt)
            # AnswerCorrectness: RAG_answer와 ground_truth(answer) 비교
            ac_score = evaluate_answer_correctness(rag_answer, ground_truth, answer_correctness_prompt)
            
            # 턴별 점수 저장
            turn["scores"] = {
                "faithfulness": f_score,
                "answer_relevancy": ar_score,
                "context_precision": cp_score,
                "context_recall": cr_score,
                "answer_correctness": ac_score
            }
            
            # Item 레벨 수집
            item_scores["faithfulness"].append(f_score)
            item_scores["answer_relevancy"].append(ar_score)
            item_scores["context_precision"].append(cp_score)
            item_scores["context_recall"].append(cr_score)
            item_scores["answer_correctness"].append(ac_score)
            
            print(f"      F={f_score}, AR={ar_score}, CP={cp_score}, CR={cr_score}, AC={ac_score}")
        
        # Item 평균 점수 계산
        item_avg = {}
        for metric, scores in item_scores.items():
            if scores:
                item_avg[metric] = round(sum(scores) / len(scores), 3)
            else:
                item_avg[metric] = 0
        
        # Multi-turn인 경우 AspectCritic 평가 추가
        if is_multi_turn:
            aspect_score = evaluate_aspect_critic(turns)
            item_avg["aspect_critic"] = aspect_score
            all_scores["aspect_critic"].append(aspect_score)
            print(f"    [AspectCritic] = {aspect_score}")
        
        # Item 레벨 평균 저장
        item["item_average_scores"] = item_avg
        
        # 전체 통계에 추가
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]:
            all_scores[metric].extend(item_scores[metric])
        
        # 출력
        avg_str = ", ".join([f"{k}={v:.2f}" for k, v in item_avg.items()])
        print(f"    → Item Average: {avg_str}")
    
    # 4. 결과 저장
    print(f"\n[4/4] 결과 저장 중: {output_path}")
    
    # 전체 평가 결과 메타데이터
    summary = {}
    for metric, scores in all_scores.items():
        if scores:
            summary[metric] = round(sum(scores) / len(scores), 3)
    
    data["evaluation_results"] = {
        "evaluated_at": datetime.now().isoformat(),
        "total_items": total_items,
        "single_turn_items": single_turn_count,
        "multi_turn_items": multi_turn_count,
        "total_turns_evaluated": sum(len(item.get("turns", [])) for item in items),
        "metrics_used": ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"],
        "multi_turn_metrics": ["aspect_critic"],
        "note": "RAG_answer is evaluated against answer (ground_truth). RAG_context is used for context metrics.",
        "summary_scores": summary
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ? 저장 완료")
    except Exception as e:
        print(f"  ? 저장 실패: {e}")
        return
    
    # 5. 최종 결과 출력
    print("\n" + "=" * 70)
    print("평가 완료!")
    print("=" * 70)
    print(f"\n총 평가 항목: {total_items}개 (Single: {single_turn_count}, Multi: {multi_turn_count})")
    print(f"총 평가 턴: {data['evaluation_results']['total_turns_evaluated']}개")
    print("\n전체 평균 점수:")
    print(f"  ? Faithfulness:       {summary.get('faithfulness', 0):.3f}")
    print(f"  ? AnswerRelevancy:    {summary.get('answer_relevancy', 0):.3f}")
    print(f"  ? ContextPrecision:   {summary.get('context_precision', 0):.3f}")
    print(f"  ? ContextRecall:      {summary.get('context_recall', 0):.3f}")
    print(f"  ? AnswerCorrectness:  {summary.get('answer_correctness', 0):.3f} (RAG_answer vs ground_truth)")
    if "aspect_critic" in summary:
        print(f"  ? AspectCritic:       {summary.get('aspect_critic', 0):.3f} (multi-turn only)")
    print(f"\n결과 파일: {output_path}")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    import sys
    
    # 기본 입력 파일 경로
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "2-가_RAG(RAGAS).json"
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
