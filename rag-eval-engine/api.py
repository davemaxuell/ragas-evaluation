import os
import json
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset

from flask import Flask, request, jsonify

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

# RAGAS 평가 라이브러리
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
)

# OPENAI API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("[오류] .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
else:
    print("[성공] OPENAI API 키 로드 완료.")

# Flask 앱 초기화
app = Flask(__name__)

# 데이터 파일 상수 정의
CSV_FILE_PATH = "samples.csv"
REQUIRED_COLUMNS = ["question", "content", "answer", "ground_truth"]
CUSTOM_PROMPTS_DIR = "custom_prompts"


# 2. 커스텀 프롬프트 로드 함수
def load_custom_prompts(metric, metric_name: str):
    """
    JSON 파일에서 커스텀 프롬프트를 로드합니다.
    enabled: false 이거나 파일이 없으면 기본값을 사용합니다.
    """
    json_path = os.path.join(CUSTOM_PROMPTS_DIR, f"{metric_name}.json")
    
    if not os.path.exists(json_path):
        return metric
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # enabled가 false면 기본값 사용
        if not config.get("enabled", False):
            return metric
        
        print(f"[커스텀 프롬프트] {metric_name} 로드 중...")
        
        # 언어 설정
        language = config.get("language", "english")
        prompts_config = config.get("prompts", {})
        
        # 각 프롬프트에 언어 및 instruction 적용
        for prompt_name, prompt_settings in prompts_config.items():
            prompt_obj = getattr(metric, prompt_name, None)
            if prompt_obj is None:
                continue
            
            # 언어 설정
            if hasattr(prompt_obj, 'language'):
                prompt_obj.language = language
            
            # instruction 커스터마이징 (빈 문자열이 아닌 경우에만)
            custom_instruction = prompt_settings.get("instruction", "")
            if custom_instruction and hasattr(prompt_obj, 'instruction'):
                prompt_obj.instruction = custom_instruction
        
        print(f"[커스텀 프롬프트] {metric_name} 적용 완료 (언어: {language})")
        return metric
        
    except Exception as e:
        print(f"[경고] {metric_name} 커스텀 프롬프트 로드 실패: {e}")
        print(f"[폴백] 기본 프롬프트 사용")
        return metric


# 2-1. 커스텀 Faithfulness 평가 함수
def load_custom_faithfulness_config():
    """
    faithfulness.json에서 custom_mode 설정을 확인합니다.
    Returns: (is_custom_mode, custom_prompt) or (False, None)
    """
    json_path = os.path.join(CUSTOM_PROMPTS_DIR, "faithfulness.json")
    
    if not os.path.exists(json_path):
        return False, None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if config.get("enabled", False) and config.get("custom_mode", False):
            custom_prompt = config.get("custom_prompt", "")
            if custom_prompt:
                return True, custom_prompt
        
        return False, None
        
    except Exception as e:
        print(f"[경고] faithfulness custom config 로드 실패: {e}")
        return False, None


def evaluate_faithfulness_custom(question: str, answer: str, context: str, custom_prompt: str) -> int:
    """
    커스텀 프롬프트를 사용하여 Faithfulness를 평가합니다.
    Returns: 0 또는 1 (이진 결과)
    """
    try:
        # 프롬프트에 변수 삽입
        formatted_prompt = custom_prompt.format(
            question=question,
            answer=answer,
            context=context
        )
        
        # OpenAI API 직접 호출
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[{"role": "user", "content": formatted_prompt}],
            
            max_completion_tokens=5
        )
        
        # 결과 파싱 (0 또는 1만 추출)
        result_text = response.choices[0].message.content.strip()
        
        # 숫자만 추출
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"[경고] 예상치 못한 응답: {result_text}, 기본값 0 사용")
            return 0
            
    except Exception as e:
        print(f"[오류] 커스텀 Faithfulness 평가 실패: {e}")
        return 0


# Custom faithfulness 설정 로드
CUSTOM_FAITHFULNESS_MODE, CUSTOM_FAITHFULNESS_PROMPT = load_custom_faithfulness_config()
if CUSTOM_FAITHFULNESS_MODE:
    print("[커스텀 평가] Faithfulness: Custom mode 활성화 (이진 평가)")


# 2-2. 커스텀 AnswerRelevancy 평가 함수
def load_custom_answer_relevancy_config():
    """
    answer_relevancy.json에서 custom_mode 설정을 확인합니다.
    Returns: (is_custom_mode, custom_prompt) or (False, None)
    """
    json_path = os.path.join(CUSTOM_PROMPTS_DIR, "answer_relevancy.json")
    
    if not os.path.exists(json_path):
        return False, None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if config.get("enabled", False) and config.get("custom_mode", False):
            custom_prompt = config.get("custom_prompt", "")
            if custom_prompt:
                return True, custom_prompt
        
        return False, None
        
    except Exception as e:
        print(f"[경고] answer_relevancy custom config 로드 실패: {e}")
        return False, None


def evaluate_answer_relevancy_custom(question: str, answer: str, custom_prompt: str) -> int:
    """
    커스텀 프롬프트를 사용하여 AnswerRelevancy를 평가합니다.
    Returns: 0 또는 1 (이진 결과)
    """
    try:
        # 프롬프트에 변수 삽입
        formatted_prompt = custom_prompt.format(
            question=question,
            answer=answer
        )
        
        # OpenAI API 직접 호출
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[{"role": "user", "content": formatted_prompt}],
            
            max_completion_tokens=5
        )
        
        # 결과 파싱 (0 또는 1만 추출)
        result_text = response.choices[0].message.content.strip()
        
        # 숫자만 추출
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"[경고] 예상치 못한 응답: {result_text}, 기본값 0 사용")
            return 0
            
    except Exception as e:
        print(f"[오류] 커스텀 AnswerRelevancy 평가 실패: {e}")
        return 0


# Custom answer_relevancy 설정 로드
CUSTOM_ANSWER_RELEVANCY_MODE, CUSTOM_ANSWER_RELEVANCY_PROMPT = load_custom_answer_relevancy_config()
if CUSTOM_ANSWER_RELEVANCY_MODE:
    print("[커스텀 평가] AnswerRelevancy: Custom mode 활성화 (이진 평가)")


# 2-3. 커스텀 ContextPrecision 평가 함수
def load_custom_context_precision_config():
    """
    context_precision.json에서 custom_mode 설정을 확인합니다.
    Returns: (is_custom_mode, custom_prompt) or (False, None)
    """
    json_path = os.path.join(CUSTOM_PROMPTS_DIR, "context_precision.json")
    
    if not os.path.exists(json_path):
        return False, None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if config.get("enabled", False) and config.get("custom_mode", False):
            custom_prompt = config.get("custom_prompt", "")
            if custom_prompt:
                return True, custom_prompt
        
        return False, None
        
    except Exception as e:
        print(f"[경고] context_precision custom config 로드 실패: {e}")
        return False, None


def evaluate_context_precision_custom(question: str, answer: str, context: str, custom_prompt: str) -> int:
    """
    커스텀 프롬프트를 사용하여 ContextPrecision을 평가합니다.
    Returns: 0 또는 1 (이진 결과)
    """
    try:
        # 프롬프트에 변수 삽입
        formatted_prompt = custom_prompt.format(
            question=question,
            answer=answer,
            context=context
        )
        
        # OpenAI API 직접 호출
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[{"role": "user", "content": formatted_prompt}],
            
            max_completion_tokens=5
        )
        
        # 결과 파싱 (0 또는 1만 추출)
        result_text = response.choices[0].message.content.strip()
        
        # 숫자만 추출
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"[경고] 예상치 못한 응답: {result_text}, 기본값 0 사용")
            return 0
            
    except Exception as e:
        print(f"[오류] 커스텀 ContextPrecision 평가 실패: {e}")
        return 0


# Custom context_precision 설정 로드
CUSTOM_CONTEXT_PRECISION_MODE, CUSTOM_CONTEXT_PRECISION_PROMPT = load_custom_context_precision_config()
if CUSTOM_CONTEXT_PRECISION_MODE:
    print("[커스텀 평가] ContextPrecision: Custom mode 활성화 (이진 평가)")


# 2-4. 커스텀 ContextRecall 평가 함수
def load_custom_context_recall_config():
    """
    context_recall.json에서 custom_mode 설정을 확인합니다.
    Returns: (is_custom_mode, custom_prompt) or (False, None)
    """
    json_path = os.path.join(CUSTOM_PROMPTS_DIR, "context_recall.json")
    
    if not os.path.exists(json_path):
        return False, None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if config.get("enabled", False) and config.get("custom_mode", False):
            custom_prompt = config.get("custom_prompt", "")
            if custom_prompt:
                return True, custom_prompt
        
        return False, None
        
    except Exception as e:
        print(f"[경고] context_recall custom config 로드 실패: {e}")
        return False, None


def evaluate_context_recall_custom(question: str, answer: str, context: str, custom_prompt: str) -> int:
    """
    커스텀 프롬프트를 사용하여 ContextRecall을 평가합니다.
    Returns: 0 또는 1 (이진 결과)
    """
    try:
        # 프롬프트에 변수 삽입
        formatted_prompt = custom_prompt.format(
            question=question,
            answer=answer,
            context=context
        )
        
        # OpenAI API 직접 호출
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[{"role": "user", "content": formatted_prompt}],
            
            max_completion_tokens=5
        )
        
        # 결과 파싱 (0 또는 1만 추출)
        result_text = response.choices[0].message.content.strip()
        
        # 숫자만 추출
        if "1" in result_text:
            return 1
        elif "0" in result_text:
            return 0
        else:
            print(f"[경고] 예상치 못한 응답: {result_text}, 기본값 0 사용")
            return 0
            
    except Exception as e:
        print(f"[오류] 커스텀 ContextRecall 평가 실패: {e}")
        return 0


# Custom context_recall 설정 로드
CUSTOM_CONTEXT_RECALL_MODE, CUSTOM_CONTEXT_RECALL_PROMPT = load_custom_context_recall_config()
if CUSTOM_CONTEXT_RECALL_MODE:
    print("[커스텀 평가] ContextRecall: Custom mode 활성화 (이진 평가)")


# 2-5. 커스텀 AnswerCorrectness 평가 함수
def load_custom_answer_correctness_config():
    """
    answer_correctness.json에서 custom_mode 설정을 확인합니다.
    Returns: (is_custom_mode, custom_prompt) or (False, None)
    """
    json_path = os.path.join(CUSTOM_PROMPTS_DIR, "answer_correctness.json")
    
    if not os.path.exists(json_path):
        return False, None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if config.get("enabled", False) and config.get("custom_mode", False):
            custom_prompt = config.get("custom_prompt", "")
            if custom_prompt:
                return True, custom_prompt
        
        return False, None
        
    except Exception as e:
        print(f"[경고] answer_correctness custom config 로드 실패: {e}")
        return False, None


def evaluate_answer_correctness_custom(answer: str, ground_truth: str, custom_prompt: str) -> float:
    """
    커스텀 프롬프트를 사용하여 AnswerCorrectness를 평가합니다.
    TP/FP/FN 라벨을 파싱하여 F1 점수를 계산합니다.
    Returns: F1 score (0.0 ~ 1.0)
    """
    try:
        # 프롬프트에 변수 삽입
        formatted_prompt = custom_prompt.format(
            answer=answer,
            ground_truth=ground_truth
        )
        
        # OpenAI API 직접 호출
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[{"role": "user", "content": formatted_prompt}],
            
            max_completion_tokens=200
        )
        
        # 결과 파싱 (TP, FP, FN 개수 추출)
        result_text = response.choices[0].message.content.strip().upper()
        
        # TP, FP, FN 개수 카운트
        tp_count = result_text.count("TP")
        fp_count = result_text.count("FP")
        fn_count = result_text.count("FN")
        
        # F1 점수 계산
        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN)
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        if tp_count == 0:
            return 0.0
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return round(f1_score, 2)
            
    except Exception as e:
        print(f"[오류] 커스텀 AnswerCorrectness 평가 실패: {e}")
        return 0.0


# Custom answer_correctness 설정 로드
CUSTOM_ANSWER_CORRECTNESS_MODE, CUSTOM_ANSWER_CORRECTNESS_PROMPT = load_custom_answer_correctness_config()
if CUSTOM_ANSWER_CORRECTNESS_MODE:
    print("[커스텀 평가] AnswerCorrectness: Custom mode 활성화 (F1 점수)")


# 3. RAGAS 모델 및 메트릭 설정
print("[초기화] 모델 및 메트릭 설정 중")

try:
    langchain_llm = ChatOpenAI(model="gpt-5-2025-08-07", api_key=OPENAI_API_KEY)
    evaluator_llm = LangchainLLMWrapper(langchain_llm)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    # 평가 메트릭 초기화 및 커스텀 프롬프트 적용
    context_precision = load_custom_prompts(ContextPrecision(), "context_precision")
    context_recall = load_custom_prompts(ContextRecall(), "context_recall")
    faithfulness = load_custom_prompts(Faithfulness(), "faithfulness")
    answer_correctness = load_custom_prompts(AnswerCorrectness(), "answer_correctness")
    answer_relevancy = load_custom_prompts(AnswerRelevancy(), "answer_relevancy")
    
    metrics_list = [
        context_precision,
        context_recall,
        faithfulness,
        answer_correctness,
        answer_relevancy,
    ]
    print("[성공] 모델 및 메트릭 설정 완료.")

except Exception as e:
    print(f"[오류] 모델 설정 중 문제가 발생했습니다: {e}")
    exit(1)


# 4. CSV 데이터 로드 및 변환
def load_and_prepare_dataset(file_path: str) -> EvaluationDataset:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[오류] 데이터 파일이 없습니다: {file_path}")

    print(f"[데이터 로드] {file_path} 파일 읽는 중...")
    
    df = pd.read_csv(file_path)
    
    # 컬럼 검증
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        raise ValueError(
            f"[오류] CSV 파일에 필수 컬럼({', '.join(REQUIRED_COLUMNS)})이 누락되었습니다."
        )

    df = df.rename(columns={
        "content": "retrieved_contexts",
        "answer": "response",
    })
    
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(
        lambda x: [str(x).strip()] if pd.notna(x) else []
    )

    # v0.3.x 호환성을 위한 별칭 추가
    df["user_input"] = df["question"]
    df["reference"] = df["ground_truth"]
    
    print(f"[데이터 준비] 총 {len(df)}개의 샘플 로드 완료.")
    return Dataset.from_pandas(df)


# 5. API 엔드포인트 정의
@app.route("/evaluate", methods=["POST"])
def run_ragas_evaluation():
    """
    RAGAS 평가를 실행하는 API 엔드포인트
    서버 시작 시 로드된 전역 데이터셋(CSV)을 사용하여 평가를 진행합니다.
    
    Method: POST
    URL: /evaluate
    Body: 요청 본문(Body)은 필요하지 않습니다.
    """
    global global_evaluation_dataset
    
    if global_evaluation_dataset is None:
        return jsonify({"error": "RAGAS 데이터셋이 서버 시작 시 로드되지 않았습니다. 파일 경로를 확인하세요."}), 500
        
    try:
        df = global_evaluation_dataset.to_pandas()
        data_count = len(df)
        print(f"[요청] 로드된 {data_count}개의 샘플에 대한 평가 요청 수신.")

        # 1. Custom Faithfulness 평가 (custom_mode가 활성화된 경우)
        custom_faithfulness_scores = []
        if CUSTOM_FAITHFULNESS_MODE and CUSTOM_FAITHFULNESS_PROMPT:
            print("[실행] Custom Faithfulness 평가 시작...")
            for idx, row in df.iterrows():
                question = row.get("question", row.get("user_input", ""))
                answer = row.get("response", "")
                # retrieved_contexts는 리스트이므로 합치기
                contexts = row.get("retrieved_contexts", [])
                context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
                
                score = evaluate_faithfulness_custom(
                    question=question,
                    answer=answer,
                    context=context,
                    custom_prompt=CUSTOM_FAITHFULNESS_PROMPT
                )
                custom_faithfulness_scores.append(score)
                print(f"  샘플 {idx + 1}/{data_count}: faithfulness = {score}")
            print(f"[완료] Custom Faithfulness 평균: {sum(custom_faithfulness_scores)/len(custom_faithfulness_scores):.2f}")

        # 2. Custom AnswerRelevancy 평가 (custom_mode가 활성화된 경우)
        custom_answer_relevancy_scores = []
        if CUSTOM_ANSWER_RELEVANCY_MODE and CUSTOM_ANSWER_RELEVANCY_PROMPT:
            print("[실행] Custom AnswerRelevancy 평가 시작...")
            for idx, row in df.iterrows():
                question = row.get("question", row.get("user_input", ""))
                answer = row.get("response", "")
                
                score = evaluate_answer_relevancy_custom(
                    question=question,
                    answer=answer,
                    custom_prompt=CUSTOM_ANSWER_RELEVANCY_PROMPT
                )
                custom_answer_relevancy_scores.append(score)
                print(f"  샘플 {idx + 1}/{data_count}: answer_relevancy = {score}")
            print(f"[완료] Custom AnswerRelevancy 평균: {sum(custom_answer_relevancy_scores)/len(custom_answer_relevancy_scores):.2f}")

        # 3. Custom ContextPrecision 평가 (custom_mode가 활성화된 경우)
        custom_context_precision_scores = []
        if CUSTOM_CONTEXT_PRECISION_MODE and CUSTOM_CONTEXT_PRECISION_PROMPT:
            print("[실행] Custom ContextPrecision 평가 시작...")
            for idx, row in df.iterrows():
                question = row.get("question", row.get("user_input", ""))
                answer = row.get("response", "")
                contexts = row.get("retrieved_contexts", [])
                context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
                
                score = evaluate_context_precision_custom(
                    question=question,
                    answer=answer,
                    context=context,
                    custom_prompt=CUSTOM_CONTEXT_PRECISION_PROMPT
                )
                custom_context_precision_scores.append(score)
                print(f"  샘플 {idx + 1}/{data_count}: context_precision = {score}")
            print(f"[완료] Custom ContextPrecision 평균: {sum(custom_context_precision_scores)/len(custom_context_precision_scores):.2f}")

        # 4. Custom ContextRecall 평가 (custom_mode가 활성화된 경우)
        custom_context_recall_scores = []
        if CUSTOM_CONTEXT_RECALL_MODE and CUSTOM_CONTEXT_RECALL_PROMPT:
            print("[실행] Custom ContextRecall 평가 시작...")
            for idx, row in df.iterrows():
                question = row.get("question", row.get("user_input", ""))
                answer = row.get("response", "")
                contexts = row.get("retrieved_contexts", [])
                context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
                
                score = evaluate_context_recall_custom(
                    question=question,
                    answer=answer,
                    context=context,
                    custom_prompt=CUSTOM_CONTEXT_RECALL_PROMPT
                )
                custom_context_recall_scores.append(score)
                print(f"  샘플 {idx + 1}/{data_count}: context_recall = {score}")
            print(f"[완료] Custom ContextRecall 평균: {sum(custom_context_recall_scores)/len(custom_context_recall_scores):.2f}")

        # 5. Custom AnswerCorrectness 평가 (custom_mode가 활성화된 경우)
        custom_answer_correctness_scores = []
        if CUSTOM_ANSWER_CORRECTNESS_MODE and CUSTOM_ANSWER_CORRECTNESS_PROMPT:
            print("[실행] Custom AnswerCorrectness 평가 시작...")
            for idx, row in df.iterrows():
                answer = row.get("response", "")
                ground_truth = row.get("ground_truth", row.get("reference", ""))
                
                score = evaluate_answer_correctness_custom(
                    answer=answer,
                    ground_truth=ground_truth,
                    custom_prompt=CUSTOM_ANSWER_CORRECTNESS_PROMPT
                )
                custom_answer_correctness_scores.append(score)
                print(f"  샘플 {idx + 1}/{data_count}: answer_correctness = {score}")
            print(f"[완료] Custom AnswerCorrectness 평균: {sum(custom_answer_correctness_scores)/len(custom_answer_correctness_scores):.2f}")

        # 6. RAGAS 평가 실행 (Custom mode 메트릭 제외)
        print("[실행] RAGAS 평가 시작...")
        
        # Custom mode인 메트릭들을 metrics_list에서 제외
        ragas_metrics = metrics_list.copy()
        if CUSTOM_FAITHFULNESS_MODE:
            ragas_metrics = [m for m in ragas_metrics if m.name != "faithfulness"]
        if CUSTOM_ANSWER_RELEVANCY_MODE:
            ragas_metrics = [m for m in ragas_metrics if m.name != "answer_relevancy"]
        if CUSTOM_CONTEXT_PRECISION_MODE:
            ragas_metrics = [m for m in ragas_metrics if m.name != "context_precision"]
        if CUSTOM_CONTEXT_RECALL_MODE:
            ragas_metrics = [m for m in ragas_metrics if m.name != "context_recall"]
        if CUSTOM_ANSWER_CORRECTNESS_MODE:
            ragas_metrics = [m for m in ragas_metrics if m.name != "answer_correctness"]
        
        result = evaluate(
            dataset=global_evaluation_dataset, 
            metrics=ragas_metrics,
            llm=evaluator_llm,
            embeddings=embedding_model,
        )

        # 7. 결과 병합
        result_df = result.to_pandas()
        
        # Custom faithfulness 점수 추가
        if CUSTOM_FAITHFULNESS_MODE and custom_faithfulness_scores:
            result_df["faithfulness"] = custom_faithfulness_scores
        
        # Custom answer_relevancy 점수 추가
        if CUSTOM_ANSWER_RELEVANCY_MODE and custom_answer_relevancy_scores:
            result_df["answer_relevancy"] = custom_answer_relevancy_scores
        
        # Custom context_precision 점수 추가
        if CUSTOM_CONTEXT_PRECISION_MODE and custom_context_precision_scores:
            result_df["context_precision"] = custom_context_precision_scores
        
        # Custom context_recall 점수 추가
        if CUSTOM_CONTEXT_RECALL_MODE and custom_context_recall_scores:
            result_df["context_recall"] = custom_context_recall_scores
        
        # Custom answer_correctness 점수 추가
        if CUSTOM_ANSWER_CORRECTNESS_MODE and custom_answer_correctness_scores:
            result_df["answer_correctness"] = custom_answer_correctness_scores
        
        # Summary scores 계산
        summary_scores = {}
        for col in result_df.columns:
            if col not in ["question", "user_input", "response", "retrieved_contexts", "ground_truth", "reference"]:
                if result_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    summary_scores[col] = float(result_df[col].mean())
        
        # 8. 결과 반환
        response_data = {
            "status": "success",
            "evaluated_samples": data_count,
            "custom_modes": {
                "faithfulness": CUSTOM_FAITHFULNESS_MODE,
                "answer_relevancy": CUSTOM_ANSWER_RELEVANCY_MODE,
                "context_precision": CUSTOM_CONTEXT_PRECISION_MODE,
                "context_recall": CUSTOM_CONTEXT_RECALL_MODE,
                "answer_correctness": CUSTOM_ANSWER_CORRECTNESS_MODE
            },
            "summary_scores": summary_scores,
            "individual_scores": result_df.to_dict(orient="records")
        }
        
        print("[응답] 평가 완료 및 결과 반환.")
        return jsonify(response_data), 200

    except Exception as e:
        print(f"[오류] 서버 내부 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "RAGAS 평가 중 내부 서버 오류가 발생했습니다."}), 500


# 6. 서버 실행
if __name__ == "__main__":
    try:
        global global_evaluation_dataset
        global_evaluation_dataset = load_and_prepare_dataset(CSV_FILE_PATH)
        
        print("[서버 시작] http://0.0.0.0:5001/evaluate 에서 요청 대기 중...")
        app.run(host="0.0.0.0", port=5001, debug=False)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"[서버 중단] 필수 파일/컬럼 오류: {e}")
        exit(1)
    except ImportError:
        print("\n[오류] 필수 라이브러리(Flask, pandas)가 설치되지 않았습니다. pip install Flask pandas")
        exit(1)
