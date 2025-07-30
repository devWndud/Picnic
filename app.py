from flask import Flask, Response, render_template_string
import cv2
import threading
import time

app = Flask(__name__)

# 서버에서 YOLO가 처리한 최신 프레임을 저장할 변수
latest_frame_lock = threading.Lock()
latest_frame = None

# 외부에서 최신 프레임을 업데이트하는 함수 (server.py에서 import 해서 사용 가능)
def update_latest_frame(frame):
    global latest_frame
    with latest_frame_lock:
        latest_frame = frame.copy()

def generate_stream():
    global latest_frame
    while True:
        with latest_frame_lock:
            if latest_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', latest_frame)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # 약 30fps

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>YOLO 실시간 스트리밍</title>
        </head>
        <body>
            <h1>YOLO 실시간 감지 영상</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="360">
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
