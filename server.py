import asyncio
import websockets
import threading
import numpy as np
import cv2
import json
import time
from flask import Flask, Response
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('yolov8x.pt')  # 속도 느릴 경우 yolov8n.pt 사용 추천

latest_frame = None
frame_lock = threading.Lock()

app = Flask(__name__)

def update_latest_frame(img):
    global latest_frame
    with frame_lock:
        latest_frame = img.copy()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', latest_frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

async def handle_client(websocket):
    print("[INFO] 클라이언트 연결됨")
    try:
        while True:
            t0 = time.time()

            msg = await websocket.recv()
            t1 = time.time()

            img = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            t2 = time.time()

            results = model(img, verbose=False)
            t3 = time.time()

            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    korean_label = label
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "label": label,
                        "koreanLabel": korean_label
                    })
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, korean_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            t4 = time.time()

            update_latest_frame(img)
            t5 = time.time()

            await websocket.send(json.dumps({
                "detections": detections,
                "tts": "위험: 사람 가까움" if any(d['label'] == 'person' for d in detections) else ""
            }))
            t6 = time.time()

            print(f"\n[프레임 처리 시간]")
            print(f"수신:        {(t1 - t0)*1000:.2f} ms")
            print(f"디코딩:      {(t2 - t1)*1000:.2f} ms")
            print(f"YOLO 추론:  {(t3 - t2)*1000:.2f} ms")
            print(f"어노테이션:  {(t4 - t3)*1000:.2f} ms")
            print(f"프레임 저장: {(t5 - t4)*1000:.2f} ms")
            print(f"응답 전송:   {(t6 - t5)*1000:.2f} ms")
            print(f"전체:        {(t6 - t0)*1000:.2f} ms")

    except websockets.exceptions.ConnectionClosed:
        print("[INFO] 클라이언트 연결 종료됨")

async def start_websocket_server():
    async with websockets.serve(handle_client, "0.0.0.0", 8000, max_size=2**22):
        print("[INFO] WebSocket 서버 실행 중 (ws://0.0.0.0:8000)")
        await asyncio.Future()

def main():
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    asyncio.run(start_websocket_server())

if __name__ == '__main__':
    main()
