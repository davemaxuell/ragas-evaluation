"""
Faithfulness 메트릭 완전 분해 분석
모든 단계, 프롬프트, Few-Shot 예제를 추출합니다.
"""

from ragas.metrics import Faithfulness
import json

def inspect_faithfulness():
    """Faithfulness 메트릭의 모든 내부 구조를 추출"""
    
    print("=" * 100)
    print("FAITHFULNESS 메트릭 완전 분해")
    print("=" * 100)
    
    metric = Faithfulness()
    
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
        
        # 프롬프트 타입
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
                
                # 예제가 튜플인 경우 (input, output)
                if isinstance(example, tuple) and len(example) == 2:
                    input_data, output_data = example
                    
                    print("\n[입력 데이터]")
                    print("─" * 50)
                    # 입력 객체의 모든 필드 출력
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
                    # 출력 객체의 모든 필드 출력
                    if hasattr(output_data, '__dict__'):
                        for key, value in output_data.__dict__.items():
                            if not key.startswith('_'):
                                print(f"{key}: {value}")
                    else:
                        print(output_data)
                
                # 단순 객체인 경우
                else:
                    if hasattr(example, '__dict__'):
                        for key, value in example.__dict__.items():
                            if not key.startswith('_'):
                                print(f"{key}: {value}")
                    else:
                        print(example)
        
        # 추가 속성들
        print("\n" + "─" * 100)
        print("🔧 기타 설정:")
        print("─" * 100)
        other_attrs = ['language', 'max_retries', 'temperature', 'model']
        for attr in other_attrs:
            if hasattr(prompt_obj, attr):
                print(f"  {attr}: {getattr(prompt_obj, attr)}")
    
    # 메트릭 실행 흐름 설명
    print("\n" + "=" * 100)
    print("🔄 FAITHFULNESS 실행 흐름 요약")
    print("=" * 100)
    print("""
Faithfulness는 "RAG 답변이 검색된 문서에 충실한가?"를 평가합니다.

단계별 실행:

1️⃣ STEP 1: Statement Generation (statement_prompt 사용)
   ├─ 입력: RAG 시스템의 답변 (response)
   ├─ 처리: LLM이 답변을 개별 진술문(atomic statements)으로 분해
   ├─ 예: "GDP는 2% 성장했고 인플레이션은 3%였다"
   │       → ["GDP는 2% 성장했다", "인플레이션은 3%였다"]
   └─ 출력: 진술문 리스트

2️⃣ STEP 2: NLI Verification (nli_statements_message 사용)
   ├─ 입력: 각 진술문 + 검색된 컨텍스트
   ├─ 처리: 각 진술문이 컨텍스트에 의해 지지되는지 판단
   │   • Supported (1) - 컨텍스트에 명확히 지지됨
   │   • Not Supported (0) - 컨텍스트에서 찾을 수 없음
   │   • Noncommittal - 판단 불가
   └─ 출력: 각 진술문별 검증 결과

3️⃣ STEP 3: Score Calculation
   └─ Faithfulness = (지지된 진술문 수) / (전체 진술문 수)

예시 계산:
  답변 → 5개 진술문 생성
  검증 → 4개 Supported, 1개 Not Supported
  점수 → 4/5 = 0.8 (80% faithful)
    """)
    
    print("\n" + "=" * 100)
    print("✅ Faithfulness 분석 완료!")
    print("=" * 100)


if __name__ == "__main__":
    inspect_faithfulness()
