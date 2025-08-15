import requests
import sys

if len(sys.argv) < 3:
    print("사용법: python ODSSclient.py <서버주소:포트> <이미지경로>")
    exit(1)

server_url = sys.argv[1]
image_path = sys.argv[2]

with open(image_path, 'rb') as f:
    files = {'image': f}
    response = requests.post(f'http://{server_url}/process', files=files)
    if response.status_code == 200:
        with open('result.jpg', 'wb') as out:
            out.write(response.content)
        print('결과 이미지를 result.jpg로 저장했습니다.')
    else:
        print('서버 오류:', response.status_code, response.text)
