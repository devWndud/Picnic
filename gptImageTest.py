
from openai import OpenAI
import base64i
mage_path = "이미지 경로 설정"
 
   
# OpenAI API 키 설정
client = OpenAI(api_key = "개인 Chatgpt API Key 설정")


# gpt-4o 모델에 텍스트와 이미지 입력을 보내는 함수
def analyze_image_with_gpt4o(image_path):

    # 이미지 파일을 base64로 인코딩
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
   
    # OpenAI API 호출 설정
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyze the following image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        temperature=0.7
    )
   
    # 결과 출력
    return response.choices[0].message.content


analyze_image_with_gpt4o(image_path)