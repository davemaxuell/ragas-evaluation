"""
AnswerRelevancy 메트릭 완전 분해 분석
모든 단계, 프롬프트, Few-Shot 예제를 추출합니다.
"""

from ragas.metrics import AnswerRelevancy

def inspect_answer_relevancy():
    """AnswerRelevancy 메트릭의 모든 내부 구조를 추출"""
    
    print("=" * 100)
    print("ANSWER RELEVANCY 메트릭 완전 분해")
    print("=" * 100)
    
    metric = AnswerRelevancy()
    
    # 메트릭 기본 정보
    print("\n📊 기본 정보:")
    print(f"  메트릭 이름: {metric.name}")
    print(f"  메트릭 타입: {type(metric).__name__}")
    if hasattr(metric, '_required_columns'):
        print(f"  필요 컬럼: {metric._required_columns}")
    
    # 모든 프롬프트 속성 찾기
    # AnswerRelevancy는 'question_generation'이라는 특별한 이름을 사용
    prompt_attrs = [attr for attr in dir(metric) 
                   if 'prompt' in attr.lower() and not attr.startswith('_')]
    
    # AnswerRelevancy 특별 처리: question_generation 추가
    if hasattr(metric, 'question_generation'):
        prompt_attrs.insert(0, 'question_generation')
    
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
            print("⚠️  None 값")
            continue
        
        print(f"\n📦 프롬프트 타입: {type(prompt_obj)}")
        
        # Instruction 추출
        if hasattr(prompt_obj, 'instruction'):
            print("\n" + "─" * 100)
            print("📝 INSTRUCTION (명령 프롬프트):")
            print("─" * 100)
            instruction = prompt_obj.instruction
            print(instruction)
            print(f"\n📏 Instruction 길이: {len(instruction)} 문자")
        
        # Input/Output Keys
        if hasattr(prompt_obj, 'input_keys'):
            print("\n" + "─" * 100)
            print(f"📥 INPUT KEYS: {prompt_obj.input_keys}")
        
        if hasattr(prompt_obj, 'output_key'):
            print(f"📤 OUTPUT KEY: {prompt_obj.output_key}")
        
        if hasattr(prompt_obj, 'output_type'):
            print(f"📦 OUTPUT TYPE: {prompt_obj.output_type}")
        
        # Few-Shot Examples 추출
        examples = None
        if hasattr(prompt_obj, 'examples'):
            examples = prompt_obj.examples
        elif hasattr(prompt_obj, 'few_shot_examples'):
            examples = prompt_obj.few_shot_examples
        
        if examples:
            print("\n" + "─" * 100)
            print(f"💡 FEW-SHOT EXAMPLES: {len(examples)}개")
            print("─" * 100)
            
            for i, example in enumerate(examples, 1):
                print(f"\n{'▼' * 50}")
                print(f"예제 {i}/{len(examples)}")
                print(f"{'▼' * 50}")
                
                # 예제가 튜플인 경우
                if isinstance(example, tuple) and len(example) == 2:
                    input_data, output_data = example
                    
                    print("\n[입력 데이터]")
                    print("─" * 50)
                    if hasattr(input_data, '__dict__'):
                        for key, value in input_data.__dict__.items():
                            if not key.startswith('_'):
                                print(f"\n{key}:")
                                if isinstance(value, str) and len(value) > 200:
                                    print(f"{value[:200]}... (총 {len(value)} 문자)")
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
        print("🔧 기타 설정:")
        print("─" * 100)
        other_attrs = ['language', 'max_retries', 'temperature', 'strictness']
        for attr in other_attrs:
            if hasattr(prompt_obj, attr):
                print(f"  {attr}: {getattr(prompt_obj, attr)}")
    
    # 실행 흐름 설명
    print("\n" + "=" * 100)
    print("🔄 ANSWER RELEVANCY 실행 흐름 요약")
    print("=" * 100)
    print("""
AnswerRelevancy는 "답변이 질문과 얼마나 관련성 있는가?"를 평가합니다.

단계별 실행:

1️⃣ STEP 1: Question Generation (역공학)
   ├─ 입력: RAG 시스템의 답변 (response)
   ├─ 처리: LLM이 "이 답변이 대답하는 질문은 무엇일까?" 추론
   ├─ 예: 답변: "GDP는 2%었다"
   │       → 생성 질문: ["GDP 성장률은?", "경제 성장은 어땠나?", "2023년 GDP는?"]
   └─ 출력: 3-5개의 생성된 질문들

2️⃣ STEP 2: Semantic Similarity (의미적 유사도)
   ├─ 입력: 원본 질문 + 생성된 질문들
   ├─ 처리: Embedding 모델로 벡터화 후 코사인 유사도 계산
   └─ 출력: 각 생성 질문과 원본 질문의 유사도 점수

3️⃣ STEP 3: Score Calculation
   └─ AnswerRelevancy = 평균 유사도

핵심 아이디어:
  ✅ 관련성 높은 답변 → 생성 질문이 원본 질문과 유사함
  ❌ 관련성 낮은 답변 → 생성 질문이 원본 질문과 다름

예시:
  원본 질문: "2023년 GDP 성장률은?"
  답변: "2023년 GDP는 2% 성장했고, 물가는 3%였다"
  
  생성 질문:
    Q1: "GDP 성장률은?" → 원본과 95% 유사
    Q2: "물가는?" → 원본과 40% 유사
    Q3: "2023년 경제는?" → 원본과 80% 유사
  
  점수: (0.95 + 0.40 + 0.80) / 3 = 0.72
  → 답변이 물가 정보도 포함해서 관련성이 약간 떨어짐
    """)
    
    print("\n" + "=" * 100)
    print("✅ AnswerRelevancy 분석 완료!")
    print("=" * 100)


if __name__ == "__main__":
    inspect_answer_relevancy()
