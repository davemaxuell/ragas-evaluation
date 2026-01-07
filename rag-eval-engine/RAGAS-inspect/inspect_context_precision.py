"""
ContextPrecision 메트릭 완전 분해 분석
모든 단계, 프롬프트, Few-Shot 예제를 추출합니다.
"""

from ragas.metrics import ContextPrecision

def inspect_context_precision():
    """ContextPrecision 메트릭의 모든 내부 구조를 추출"""
    
    print("=" * 100)
    print("CONTEXT PRECISION 메트릭 완전 분해")
    print("=" * 100)
    
    metric = ContextPrecision()
    
    # 메트릭 기본 정보
    print("\n기본 정보:")
    print(f"  메트릭 이름: {metric.name}")
    print(f"  메트릭 타입: {type(metric).__name__}")
    if hasattr(metric, '_required_columns'):
        print(f"  필요 컬럼: {metric._required_columns}")
    
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
                
                # 예제가 튜플인 경우
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
                                print(f"{key}:")
                                if isinstance(value, str) and len(value) > 200:
                                    print(f"  {value[:200]}...")
                                else:
                                    print(f"  {value}")
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
        other_attrs = ['language', 'max_retries']
        for attr in other_attrs:
            if hasattr(prompt_obj, attr):
                print(f"  {attr}: {getattr(prompt_obj, attr)}")
    
    # 실행 흐름 설명
    print("\n" + "=" * 100)
    print("CONTEXT PRECISION 실행 흐름 요약")
    print("=" * 100)
    print("""
ContextPrecision은 "검색된 컨텍스트가 답변 생성에 유용했는가?"를 평가합니다.

단계별 실행:

1️⃣ STEP 1: Context Usefulness Verification
   ├─ 입력: 질문 + 답변 + 검색된 컨텍스트
   ├─ 처리: 각 컨텍스트 조각이 답변 생성에 유용했는지 LLM이 판단
   │   • verdict = 1: 유용함
   │   • verdict = 0: 유용하지 않음 (노이즈)
   └─ 출력: 각 컨텍스트별 유용성 판정

2️⃣ STEP 2: Precision Calculation
   └─ ContextPrecision = (유용한 컨텍스트 수) / (전체 컨텍스트 수)

핵심 아이디어:
  ✅ 높은 Precision → 검색된 모든 컨텍스트가 답변에 필요함 (노이즈 없음)
  ❌ 낮은 Precision → 불필요한 컨텍스트가 많음 (검색 품질 낮음)

예시:
  질문: "2023년 GDP 성장률은?"
  
  검색된 컨텍스트 5개:
    Context 1: "2023년 GDP는 2% 성장..." → verdict=1 (유용)
    Context 2: "2023년 경제 전망..." → verdict=1 (유용)
    Context 3: "2022년 물가상승률..." → verdict=0 (불필요)
    Context 4: "실업률 통계..." → verdict=0 (불필요)
    Context 5: "2023년 성장 분석..." → verdict=1 (유용)
  
  점수: 3/5 = 0.6 (60% precision)
  → 검색 품질 개선 필요 (노이즈가 40%)

이 메트릭이 중요한 이유:
- RAG 시스템의 검색 품질 평가
- 불필요한 컨텍스트 → 토큰 낭비, 응답 품질 저하
- 개선 방향: 검색 알고리즘, 재랭킹(reranking) 등
    """)
    
    print("\n" + "=" * 100)
    print("✅ ContextPrecision 분석 완료!")
    print("=" * 100)


if __name__ == "__main__":
    inspect_context_precision()
