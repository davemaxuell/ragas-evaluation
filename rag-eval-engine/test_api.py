import requests
import json

# API 서버 주소 설정 (5001 포트)
API_URL = "http://127.0.0.1:5001/evaluate"

def test_evaluation():
    print(f"요청 보내는 중... ({API_URL})")
    
    try:
        # 빈 POST 요청 전송
        response = requests.post(
            API_URL, 
            # data=None, json=None (요청 본문 삭제)
            headers={"Content-Type": "application/json"}
        )

        # 응답 확인
        if response.status_code == 200:
            result = response.json()
            print("\n평가 성공!")
            print("=" * 40)
            print(f"[총 평가 샘플]: {result.get('evaluated_samples', 'N/A')}개")
            print("[요약 점수]")
            print(json.dumps(result["summary_scores"], indent=4, ensure_ascii=False))
            print("=" * 40)
        else:
            print(f"\n오류 발생 (Code: {response.status_code})")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("\n서버에 연결할 수 없습니다. 'api.py'가 실행 중인지 확인하세요.")

if __name__ == "__main__":
    test_evaluation()