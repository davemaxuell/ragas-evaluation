"""
ContextRecall 메트릭 완전 분해 분석
모든 단계, 프롬프트, Few-Shot 예제를 추출합니다.
"""

from ragas.metrics import ContextRecall

def inspect_context_recall():
    """ContextRecall 메트릭의 모든 내부 구조를 추출"""
    
    print("=" * 100)
    print("CONTEXT RECALL 메트릭 완전 분해")
    print("=" * 100)
    
    metric = ContextRecall()
    
    # 메트릭 기본 정보
    print("\n📊 기본 정보:")
    print(f"  메트릭 이름: {metric.name}")
    print(f"  메트릭 타입: {type(metric).__name__}")
    if hasattr(metric, '_required_columns'):
        print(f"  필요 컬럼: {metric._required_columns}")
    
    # 모든 프롬프트 속성 찾기
    prompt_attrs = [attr for attr in dir(metric) 
                   if 'prompt' in attr.lower() and not attr.startswith('_')]
    
    print(f"\n🔍 발견된 프롬프트 속성: {len(prompt_attrs)}개")
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
        print("🔧 기타 설정:")
        print("─" * 100)
        other_attrs = ['language', 'max_retries']
        for attr in other_attrs:
            if hasattr(prompt_obj, attr):
                print(f"  {attr}: {getattr(prompt_obj, attr)}")
    
    # 실행 흐름 설명
    print("\n" + "=" * 100)
    print("🔄 CONTEXT RECALL 실행 흐름 요약")
    print("=" * 100)
    print("""
ContextRecall은 "검색된 컨텍스트가 정답 정보를 모두 포함하는가?"를 평가합니다.

단계별 실행:

1️⃣ STEP 1: Ground Truth Sentence Extraction
   ├─ 입력: Ground Truth (정답)
   ├─ 처리: 정답을 개별 문장으로 분리
   └─ 출력: 정답 문장 리스트

2️⃣ STEP 2: Sentence Attribution
   ├─ 입력: 각 정답 문장 + 검색된 컨텍스트들
   ├─ 처리: 각 문장이 컨텍스트에서 찾을 수 있는지 LLM이 판단
   │   • Attributable (1): 컨텍스트에서 찾을 수 있음
   │   • Not Attributable (0): 컨텍스트에 없음
   └─ 출력: 각 문장별 귀속 가능 여부

3️⃣ STEP 3: Recall Calculation
   └─ ContextRecall = (찾을 수 있는 문장 수) / (전체 정답 문장 수)

핵심 아이디어:
  ✅ 높은 Recall → 검색이 정답 도출에 필요한 정보를 모두 가져옴
  ❌ 낮은 Recall → 필요한 정보를 놓침 (검색 범위 부족)

예시:
  Ground Truth: "2023년 GDP는 2% 성장했고, 전년대비 0.5%p 상승했다."
  
  정답 문장 분해:
    Sentence 1: "2023년 GDP는 2% 성장했다"
    Sentence 2: "전년대비 0.5%p 상승했다"
  
  검색된 컨텍스트:
    Context 1: "2023년 경제성장률은 2.0%를 기록..."
    Context 2: "물가상승률은 3.5%..."
  
  판정:
    Sentence 1 vs Contexts → Attributable=1 (Context 1에서 발견)
    Sentence 2 vs Contexts → Attributable=0 (어느 컨텍스트에도 없음)
  
  점수: 1/2 = 0.5 (50% recall)
  → 검색 범위 확대 필요

Precision vs Recall 차이:
  Precision: 검색한 것 중 쓸모있는 것 비율 (품질)
  Recall: 필요한 것을 다 가져왔는지 (완전성)
    """)
    
    print("\n" + "=" * 100)
    print("✅ ContextRecall 분석 완료!")
    print("=" * 100)


if __name__ == "__main__":
    inspect_context_recall()
