import requests
import sys

server_url = "222.231.24.5"

with open("./input.mp4", 'rb') as f:
    files = {'video': f}
    response = requests.post(f'http://{server_url}:5001/process_video', files=files)
    if response.status_code == 200:
        with open('result.mp4', 'wb') as out:
            out.write(response.content)
        print('결과 영상을 result.mp4로 저장했습니다.')
    else:
        print('서버 오류:', response.status_code, response.text)
