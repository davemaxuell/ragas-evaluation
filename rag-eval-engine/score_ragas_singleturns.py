"""
RAGAS 싱글턴 JSON 평가 스크립트 (RAGAS 기본 프롬프트 사용)
- RAGAS 라이브러리의 기본 프롬프트를 사용하여 5개 메트릭 평가
- Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
- RAG_answer를 평가 대상 답변으로 사용, answer를 ground_truth로 사용
- RAG_context를 검색된 컨텍스트로 사용
- 0.0~1.0 점수 반환 (이진값 아님)
"""

import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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


def evaluate_json_file(input_path: str, output_path: str = None):
    """
    JSON 파일을 읽어 RAGAS 기본 프롬프트를 사용하여 평가합니다.
    
    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로 (없으면 자동 생성)
    """
    
    # 출력 경로 설정 (모델명 포함)
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if "_scored" in base:
            base = base.split("_scored")[0]
        output_path = f"{base}_scored_ragas({OPENAI_MODEL_NAME}){ext}"
    
    print("=" * 70)
    print("RAGAS 싱글턴 JSON 평가 스크립트 (기본 프롬프트 사용)")
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
        print(f"  ✓ {total_items}개 항목 로드 완료")
        
    except Exception as e:
        print(f"  ✗ JSON 로드 실패: {e}")
        return
    
    # 3. RAGAS 데이터셋 준비
    print(f"\n[3/5] RAGAS 데이터셋 준비 중...")
    samples = []
    sample_to_item_map = []  # 샘플 인덱스 -> (item_idx, turn_idx) 매핑
    
    for item_idx, item in enumerate(items):
        turns = item.get("turns", [])
        
        if not turns:
            continue
        
        # 첫 번째 턴 사용 (single_turn 가정)
        turn = turns[0]
        question = turn.get("question", "")
        ground_truth = turn.get("answer", "")  # 원본 answer를 ground_truth로 사용
        rag_answer = turn.get("RAG_answer", "")  # RAG 생성 답변을 평가 대상으로 사용
        rag_contexts = turn.get("RAG_context", [])  # RAG 검색 컨텍스트 사용
        
        # RAG_answer가 없으면 스킵
        if not rag_answer:
            continue
        
        # 컨텍스트 추출 (RAG_context는 딕셔너리 리스트로 content 필드 추출)
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
        sample_to_item_map.append((item_idx, 0))
    
    print(f"  ✓ {len(samples)}개 샘플 준비 완료")
    
    if len(samples) == 0:
        print("  ✗ 평가할 샘플이 없습니다.")
        return
    
    # 4. RAGAS 평가 실행
    print(f"\n[4/5] RAGAS 평가 실행 중... (총 {len(samples)}개)")
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
    
    # 통계 초기화
    scores_summary = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
        "answer_correctness": []
    }
    
    for sample_idx, (item_idx, turn_idx) in enumerate(sample_to_item_map):
        row = results_df.iloc[sample_idx]
        
        # 점수 추출 (NaN 처리)
        scores = {
            "faithfulness": float(row.get("faithfulness", 0)) if not pd.isna(row.get("faithfulness")) else 0.0,
            "answer_relevancy": float(row.get("answer_relevancy", 0)) if not pd.isna(row.get("answer_relevancy")) else 0.0,
            "context_precision": float(row.get("context_precision", 0)) if not pd.isna(row.get("context_precision")) else 0.0,
            "context_recall": float(row.get("context_recall", 0)) if not pd.isna(row.get("context_recall")) else 0.0,
            "answer_correctness": float(row.get("answer_correctness", 0)) if not pd.isna(row.get("answer_correctness")) else 0.0,
        }
        
        # 턴에 점수 추가
        items[item_idx]["turns"][turn_idx]["scores"] = scores
        
        # 통계 수집
        for metric, score in scores.items():
            scores_summary[metric].append(score)
    
    # 평가 메타데이터 추가
    summary = {}
    for metric, scores_list in scores_summary.items():
        if scores_list:
            summary[metric] = round(sum(scores_list) / len(scores_list), 3)
    
    data["evaluation_results"] = {
        "evaluated_at": datetime.now().isoformat(),
        "model": OPENAI_MODEL_NAME,
        "total_items": total_items,
        "evaluated_samples": len(samples),
        "metrics_used": ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"],
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
    print(f"\n총 평가 항목: {len(samples)}개")
    print("\n평균 점수:")
    print(f"  ✓ Faithfulness:       {summary.get('faithfulness', 0):.3f}")
    print(f"  ✓ AnswerRelevancy:    {summary.get('answer_relevancy', 0):.3f}")
    print(f"  ✓ ContextPrecision:   {summary.get('context_precision', 0):.3f}")
    print(f"  ✓ ContextRecall:      {summary.get('context_recall', 0):.3f}")
    print(f"  ✓ AnswerCorrectness:  {summary.get('answer_correctness', 0):.3f}")
    print(f"\n결과 파일: {output_path}")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # 기본 입력 파일 경로
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "2-가 RAG(멀티모달)_RAG_augmented.json"
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
