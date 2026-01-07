"""
RAGAS 멀티턴 JSON 평가 스크립트 (RAGAS 기본 프롬프트 사용)
- RAGAS 라이브러리의 기본 프롬프트를 사용하여 5개 메트릭 평가
- 모든 turn을 순회하여 평가
- Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
- Multi-turn 아이템에 대해 AspectCritic 평가 추가
- Item별 평균 점수 계산
- 0.0~1.0 점수 반환 (이진값 아님)
"""

import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
)
from ragas import EvaluationDataset, SingleTurnSample

# .env 파일에서 환경변수 로드
load_dotenv()

# OpenAI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 모델명 (환경변수에서 로드, 기본값: gpt-4o)
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

# AspectCritic 프롬프트 (Multi-turn 대화 품질 평가)
ASPECT_CRITIC_PROMPT = """당신은 다중 턴 대화의 품질을 평가하는 전문가입니다.

아래는 사용자와 AI 간의 대화입니다:

{conversation}

다음 기준에 따라 대화의 전반적인 품질을 0.0에서 1.0 사이의 점수로 평가하세요:
1. 일관성 (Coherence): 대화가 논리적으로 연결되고 흐름이 자연스러운가?
2. 맥락 유지 (Context Retention): AI가 이전 턴의 내용을 잘 기억하고 활용하는가?
3. 점진적 정보 제공 (Progressive Information): 각 턴에서 새로운 정보가 적절히 추가되는가?

위 기준을 종합하여 0.0에서 1.0 사이의 점수만 출력하세요 (예: 0.75).
숫자만 출력하세요."""


def evaluate_aspect_critic(turns: list) -> float:
    """AspectCritic 평가: 다중 턴 대화의 전반적인 품질 (0.0~1.0)"""
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
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_completion_tokens=10
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 숫자 추출
        try:
            score = float(result_text)
            return min(max(score, 0.0), 1.0)  # 0.0~1.0 범위로 제한
        except ValueError:
            # 숫자가 아닌 경우 기본값 사용
            if "1" in result_text and "0" not in result_text:
                return 1.0
            elif "0" in result_text:
                return 0.0
            else:
                print(f"    [경고] AspectCritic 예상치 못한 응답: {result_text}")
                return 0.5
            
    except Exception as e:
        print(f"    [오류] AspectCritic 평가 실패: {e}")
        return 0.5


def evaluate_json_file(input_path: str, output_path: str = None):
    """
    JSON 파일을 읽어 모든 턴을 RAGAS 기본 프롬프트로 평가합니다.
    Multi-turn 아이템에는 AspectCritic 평가도 추가합니다.
    """
    
    # 출력 경로 설정 (모델명 포함)
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if "_scored" in base:
            base = base.split("_scored")[0]
        output_path = f"{base}_scored_ragas({OPENAI_MODEL_NAME}){ext}"
    
    print("=" * 70)
    print("RAGAS 멀티턴 JSON 평가 스크립트 (기본 프롬프트 사용)")
    print("=" * 70)
    
    # 1. 모델 설정
    print(f"\n[1/5] 모델 설정 중... (모델: {OPENAI_MODEL_NAME})")
    try:
        langchain_llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME, 
            api_key=OPENAI_API_KEY,
            timeout=300,  # 300초 타임아웃 (5분)
            max_retries=5  # 최대 5회 재시도
        )
        evaluator_llm = LangchainLLMWrapper(langchain_llm)
        
        langchain_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        
        # 메트릭 초기화 (기본 프롬프트 사용)
        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
            AnswerCorrectness(llm=evaluator_llm),
        ]
        print("  ✓ 모델 및 메트릭 설정 완료")
        
    except Exception as e:
        print(f"  ✗ 모델 설정 실패: {e}")
        return
    
    # 2. JSON 파일 로드
    print(f"\n[2/5] JSON 파일 로드 중: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get("items", [])
        total_items = len(items)
        
        # Single-turn vs Multi-turn 카운트
        single_turn_count = sum(1 for item in items if len(item.get("turns", [])) == 1)
        multi_turn_count = total_items - single_turn_count
        
        print(f"  ✓ {total_items}개 항목 로드 완료")
        print(f"  - Single-turn: {single_turn_count}개")
        print(f"  - Multi-turn: {multi_turn_count}개")
        
    except Exception as e:
        print(f"  ✗ JSON 로드 실패: {e}")
        return
    
    # 3. 모든 턴에 대한 RAGAS 데이터셋 준비
    print(f"\n[3/5] RAGAS 데이터셋 준비 중...")
    samples = []
    sample_to_turn_map = []  # 샘플 인덱스 -> (item_idx, turn_idx) 매핑
    
    for item_idx, item in enumerate(items):
        turns = item.get("turns", [])
        
        for turn_idx, turn in enumerate(turns):
            question = turn.get("question", "")
            ground_truth = turn.get("answer", "")
            rag_answer = turn.get("RAG_answer", "")
            rag_contexts = turn.get("RAG_context", [])
            
            # RAG_answer가 없으면 스킵
            if not rag_answer:
                continue
            
            # 컨텍스트 추출
            context_list = []
            if isinstance(rag_contexts, list):
                for ctx in rag_contexts:
                    if isinstance(ctx, dict):
                        content = ctx.get("content", "")
                        if content:
                            context_list.append(content)
                    elif isinstance(ctx, str):
                        context_list.append(ctx)
            
            # RAGAS SingleTurnSample 생성
            sample = SingleTurnSample(
                user_input=question,
                response=rag_answer,
                reference=ground_truth,
                retrieved_contexts=context_list
            )
            samples.append(sample)
            sample_to_turn_map.append((item_idx, turn_idx))
    
    total_turns = len(samples)
    print(f"  ✓ {total_turns}개 턴 샘플 준비 완료")
    
    if total_turns == 0:
        print("  ✗ 평가할 샘플이 없습니다.")
        return
    
    # 4. RAGAS 평가 실행
    print(f"\n[4/5] RAGAS 평가 실행 중... (총 {total_turns}개 턴)")
    print("  ⏳ 이 작업은 시간이 걸릴 수 있습니다...")
    
    try:
        eval_dataset = EvaluationDataset(samples=samples)
        results = evaluate(dataset=eval_dataset, metrics=metrics)
        results_df = results.to_pandas()
        print("  ✓ RAGAS 평가 완료")
        
    except Exception as e:
        print(f"  ✗ RAGAS 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 결과를 원본 JSON에 추가
    print(f"\n[5/5] 결과 저장 중: {output_path}")
    
    # 전체 통계 초기화
    all_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
        "answer_correctness": [],
        "aspect_critic": []
    }
    
    # 각 턴에 점수 할당
    for sample_idx, (item_idx, turn_idx) in enumerate(sample_to_turn_map):
        row = results_df.iloc[sample_idx]
        
        # 점수 추출 (NaN 처리)
        import pandas as pd
        scores = {
            "faithfulness": round(float(row.get("faithfulness", 0)), 3) if not pd.isna(row.get("faithfulness")) else 0.0,
            "answer_relevancy": round(float(row.get("answer_relevancy", 0)), 3) if not pd.isna(row.get("answer_relevancy")) else 0.0,
            "context_precision": round(float(row.get("context_precision", 0)), 3) if not pd.isna(row.get("context_precision")) else 0.0,
            "context_recall": round(float(row.get("context_recall", 0)), 3) if not pd.isna(row.get("context_recall")) else 0.0,
            "answer_correctness": round(float(row.get("answer_correctness", 0)), 3) if not pd.isna(row.get("answer_correctness")) else 0.0,
        }
        
        # 턴에 점수 추가
        items[item_idx]["turns"][turn_idx]["scores"] = scores
        
        # 통계 수집
        for metric, score in scores.items():
            all_scores[metric].append(score)
    
    # 각 아이템별로 평균 점수 계산 및 AspectCritic 평가
    print("\n  AspectCritic 평가 중 (Multi-turn 아이템만)...")
    for item_idx, item in enumerate(items):
        turns = item.get("turns", [])
        num_turns = len(turns)
        is_multi_turn = num_turns > 1
        
        # Item 평균 점수 계산
        item_avg = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]:
            turn_scores = [t.get("scores", {}).get(metric, 0) for t in turns if "scores" in t]
            if turn_scores:
                item_avg[metric] = round(sum(turn_scores) / len(turn_scores), 3)
            else:
                item_avg[metric] = 0.0
        
        # Multi-turn인 경우 AspectCritic 평가 추가
        if is_multi_turn:
            time.sleep(0.5)  # Rate limiting
            aspect_score = evaluate_aspect_critic(turns)
            item_avg["aspect_critic"] = round(aspect_score, 3)
            all_scores["aspect_critic"].append(aspect_score)
        
        # Item 레벨 평균 저장
        item["item_average_scores"] = item_avg
    
    # 전체 평가 결과 메타데이터
    summary = {}
    for metric, scores_list in all_scores.items():
        if scores_list:
            summary[metric] = round(sum(scores_list) / len(scores_list), 3)
    
    data["evaluation_results"] = {
        "evaluated_at": datetime.now().isoformat(),
        "model": OPENAI_MODEL_NAME,
        "total_items": total_items,
        "single_turn_items": single_turn_count,
        "multi_turn_items": multi_turn_count,
        "total_turns_evaluated": total_turns,
        "metrics_used": ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"],
        "multi_turn_metrics": ["aspect_critic"],
        "prompt_type": "RAGAS default prompts",
        "note": "RAG_answer is evaluated against answer (ground_truth). RAG_context is used for context metrics.",
        "summary_scores": summary
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 저장 완료")
    except Exception as e:
        print(f"  ✗ 저장 실패: {e}")
        return
    
    # 최종 결과 출력
    print("\n" + "=" * 70)
    print("평가 완료!")
    print("=" * 70)
    print(f"\n총 평가 항목: {total_items}개 (Single: {single_turn_count}, Multi: {multi_turn_count})")
    print(f"총 평가 턴: {total_turns}개")
    print("\n전체 평균 점수:")
    print(f"  ✓ Faithfulness:       {summary.get('faithfulness', 0):.3f}")
    print(f"  ✓ AnswerRelevancy:    {summary.get('answer_relevancy', 0):.3f}")
    print(f"  ✓ ContextPrecision:   {summary.get('context_precision', 0):.3f}")
    print(f"  ✓ ContextRecall:      {summary.get('context_recall', 0):.3f}")
    print(f"  ✓ AnswerCorrectness:  {summary.get('answer_correctness', 0):.3f}")
    if "aspect_critic" in summary:
        print(f"  ✓ AspectCritic:       {summary.get('aspect_critic', 0):.3f} (multi-turn only)")
    print(f"\n결과 파일: {output_path}")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    
    # 기본 입력 파일 경로
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "2-가_RAG(RAGAS)_RAG_augmented.json"
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
