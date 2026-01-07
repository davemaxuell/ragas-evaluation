"""
모든 RAGAS 메트릭 인스펙터 실행
각 메트릭의 프롬프트, Few-Shot 예제, 실행 흐름을 모두 표시합니다.
"""

import sys
import os

# 각 인스펙터 임포트
from inspect_faithfulness import inspect_faithfulness
from inspect_answer_relevancy import inspect_answer_relevancy
from inspect_context_precision import inspect_context_precision
from inspect_context_recall import inspect_context_recall
from inspect_answer_correctness import inspect_answer_correctness

def run_all_inspectors():
    """모든 메트릭 인스펙터를 순차적으로 실행"""
    
    print("\n" + "🔬" * 50)
    print("RAGAS 메트릭 완전 분해 - 전체 실행")
    print("🔬" * 50)
    
    metrics = [
        ("Faithfulness", inspect_faithfulness),
        ("AnswerRelevancy", inspect_answer_relevancy),
        ("ContextPrecision", inspect_context_precision),
        ("ContextRecall", inspect_context_recall),
        ("AnswerCorrectness", inspect_answer_correctness),
    ]
    
    for i, (name, inspector_func) in enumerate(metrics, 1):
        print(f"\n\n{'=' * 100}")
        print(f"메트릭 {i}/5: {name}")
        print(f"{'=' * 100}\n")
        
        try:
            inspector_func()
        except Exception as e:
            print(f"⚠️  오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(metrics):
            print("\n" + "─" * 100)
            print("다음 메트릭으로 이동...")
            print("─" * 100)
    
    print("\n\n" + "🎯" * 50)
    print("전체 분석 완료!")
    print("🎯" * 50)
    
    print("\n📊 5개 메트릭 요약:")
    print("""
1. Faithfulness (충실성)
   - 답변이 검색된 문서에 근거했는가?
   - 2단계: Statement 생성 → NLI 검증

2. AnswerRelevancy (답변 관련성)
   - 답변이 질문과 관련있는가?
   - 2단계: 질문 생성 → 유사도 계산

3. ContextPrecision (컨텍스트 정밀도)
   - 검색된 컨텍스트가 모두 유용한가?
   - 1단계: 각 컨텍스트 유용성 판단

4. ContextRecall (컨텍스트 재현율)
   - 필요한 정보를 모두 검색했는가?
   - 2단계: 문장 분리 → 귀속 판단

5. AnswerCorrectness (답변 정확성)
   - 답변이 정답과 일치하는가?
   - 3단계: 진술문 추출 → F1 계산 → 의미 유사도
    """)


def run_single_inspector(metric_name):
    """특정 메트릭만 실행"""
    
    inspectors = {
        "faithfulness": inspect_faithfulness,
        "answer_relevancy": inspect_answer_relevancy,
        "context_precision": inspect_context_precision,
        "context_recall": inspect_context_recall,
        "answer_correctness": inspect_answer_correctness,
    }
    
    key = metric_name.lower().replace(" ", "_")
    
    if key in inspectors:
        inspectors[key]()
    else:
        print(f"❌ 메트릭 '{metric_name}'를 찾을 수 없습니다.")
        print(f"사용 가능한 메트릭: {', '.join(inspectors.keys())}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 특정 메트릭만 실행
        metric_name = sys.argv[1]
        print(f"\n🎯 {metric_name} 메트릭만 분석합니다.\n")
        run_single_inspector(metric_name)
    else:
        # 모든 메트릭 실행
        run_all_inspectors()
    
    print("\n" + "=" * 100)
    print("사용 방법:")
    print("  - 전체 실행: python run_all_inspectors.py")
    print("  - 특정 메트릭: python run_all_inspectors.py faithfulness")
    print("=" * 100)
