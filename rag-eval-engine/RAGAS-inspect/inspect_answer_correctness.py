"""
AnswerCorrectness 메트릭 완전 분해 분석
모든 단계, 프롬프트, Few-Shot 예제를 추출합니다.
"""

from ragas.metrics import AnswerCorrectness

def inspect_answer_correctness():
    """AnswerCorrectness 메트릭의 모든 내부 구조를 추출"""
    
    print("=" * 100)
    print("ANSWER CORRECTNESS 메트릭 완전 분해")
    print("=" * 100)
    
    metric = AnswerCorrectness()
    
    # 메트릭 기본 설정
    print("\n기본 설정:")
    print(f"  메트릭 이름: {metric.name}")
    print(f"  메트릭 타입: {type(metric).__name__}")
    if hasattr(metric, '_required_columns'):
        print(f"  필요 컬럼: {metric._required_columns}")
    
    # 가중치 확인
    if hasattr(metric, 'weights'):
        print(f"  가중치 설정: {metric.weights}")
    
    # 모든 프롬프트 속성 찾기
    prompt_attrs = [attr for attr in dir(metric) 
                   if 'prompt' in attr.lower() and not attr.startswith('_')]
    
    print(f"\n발견된 프롬프트 속성: {len(prompt_attrs)}개")
    for attr in prompt_attrs:
        print(f"  - {attr}")
    
    # 각 프롬프트 상세 분석
    for step_num, attr_name in enumerate(prompt_attrs, 1):
        print("\n" + "=" * 100)
        print(f"STEP {step_num}: {attr_name}")
        print("=" * 100)
        
        prompt_obj = getattr(metric, attr_name, None)
        if prompt_obj is None:
            print("None 값")
            continue
        
        print(f"\n프롬프트 타입: {type(prompt_obj)}")
        
        # Instruction 추출
        if hasattr(prompt_obj, 'instruction'):
            print("\n" + "─" * 100)
            print("INSTRUCTION (명령 프롬프트):")
            print("─" * 100)
            instruction = prompt_obj.instruction
            print(instruction)
            print(f"\nInstruction 길이: {len(instruction)} 문자")
        
        # Input/Output Keys
        if hasattr(prompt_obj, 'input_keys'):
            print("\n" + "─" * 100)
            print(f"INPUT KEYS: {prompt_obj.input_keys}")
        
        if hasattr(prompt_obj, 'output_key'):
            print(f"OUTPUT KEY: {prompt_obj.output_key}")
        
        if hasattr(prompt_obj, 'output_type'):
            print(f"OUTPUT TYPE: {prompt_obj.output_type}")
        
        # Few-Shot Examples 추출
        examples = None
        if hasattr(prompt_obj, 'examples'):
            examples = prompt_obj.examples
        elif hasattr(prompt_obj, 'few_shot_examples'):
            examples = prompt_obj.few_shot_examples
        
        if examples:
            print("\n" + "─" * 100)
            print(f"FEW-SHOT EXAMPLES: {len(examples)}개")
            print("─" * 100)
            
            for i, example in enumerate(examples, 1):
                print(f"\n{'▼' * 50}")
                print(f"예제 {i}/{len(examples)}")
                print(f"{'▼' * 50}")
                
                if isinstance(example, tuple) and len(example) == 2:
                    input_data, output_data = example
                    
                    print("\n[입력 데이터]")
                    print("─" * 50)
                    if hasattr(input_data, '__dict__'):
                        for key, value in input_data.__dict__.items():
                            if not key.startswith('_'):
                                print(f"\n{key}:")
                                if isinstance(value, str) and len(value) > 300:
                                    print(f"{value[:300]}...")
                                    print(f"(총 {len(value)} 문자)")
                                else:
                                    print(value)
                    else:
                        print(input_data)
                    
                    print("\n[출력 데이터]")
                    print("─" * 50)
                    if hasattr(output_data, '__dict__'):
                        for key, value in output_data.__dict__.items():
                            if not key.startswith('_'):
                                print(f"{key}: {value}")
                    else:
                        print(output_data)
                else:
                    if hasattr(example, '__dict__'):
                        for key, value in example.__dict__.items():
                            if not key.startswith('_'):
                                print(f"{key}: {value}")
                    else:
                        print(example)
        
        # 기타 설정
        print("\n" + "─" * 100)
        print("기타 설정:")
        print("─" * 100)
        other_attrs = ['language', 'max_retries', 'temperature']
        for attr in other_attrs:
            if hasattr(prompt_obj, attr):
                print(f"  {attr}: {getattr(prompt_obj, attr)}")
    
    # 실행 흐름 설명
    print("\n" + "=" * 100)
    print("ANSWER CORRECTNESS 실행 흐름 요약")
    print("=" * 100)
    print("""
AnswerCorrectness는 "답변이 정답과 얼마나 일치하는가?"를 평가합니다.
이 메트릭은 두 가지 측면을 결합합니다: 사실적 정확성 + 의미적 유사성

단계별 실행:

1️⃣ STEP 1: Statement Extraction (진술문 추출)
   ├─ 입력: RAG 답변 + Ground Truth
   ├─ 처리: 두 텍스트를 각각 개별 진술문으로 분해
   └─ 출력: 답변 진술문 리스트, 정답 진술문 리스트

2️⃣ STEP 2: Statement Comparison (F1 Score)
   ├─ 입력: 답변 진술문들 vs 정답 진술문들
   ├─ 처리: 겹치는 진술문 비율 계산
   │   • TP (True Positive): 답변∩정답
   │   • FP (False Positive): 답변에만 있음
   │   • FN (False Negative): 정답에만 있음
   ├─ 계산: F1 = 2*TP / (2*TP + FP + FN)
   └─ 출력: Factual Correctness Score (0-1)

3️⃣ STEP 3: Semantic Similarity (의미 유사도)
   ├─ 입력: 전체 답변 텍스트 + 전체 정답 텍스트
   ├─ 처리: Embedding 모델로 벡터화 후 코사인 유사도 계산
   └─ 출력: Semantic Similarity Score (0-1)

4️⃣ STEP 4: Final Score Calculation
   └─ AnswerCorrectness = 
       (factual_weight × Factual Score) + 
       (semantic_weight × Semantic Score)
       
       기본 가중치: factual=0.75, semantic=0.25

핵심 아이디어:
  ✅ 사실 정확성: 내용이 맞는가?
  ✅ 의미 유사성: 표현 방식이 비슷한가?

예시:
  Ground Truth: "2023년 GDP는 2.0% 성장했다"
  답변 1: "2023년 경제성장률은 2.0%를 기록했다"
    → Factual: 1.0 (내용 동일)
    → Semantic: 0.95 (표현 거의 동일)
    → Final: 0.75×1.0 + 0.25×0.95 = 0.99
  
  답변 2: "경제가 성장했다"
    → Factual: 0.3 (정보 누락)
    → Semantic: 0.6 (의미만 유사)
    → Final: 0.75×0.3 + 0.25×0.6 = 0.375

가중치 조정 가능:
  from ragas.metrics import AnswerCorrectness
  metric = AnswerCorrectness(weights=[0.8, 0.2])  # 사실성에 더 높은 가중치
    """)
    
    print("\n" + "=" * 100)
    print("✅ AnswerCorrectness 분석 완료!")
    print("=" * 100)


if __name__ == "__main__":
    inspect_answer_correctness()
