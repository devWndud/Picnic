import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response
import threading
import json
import time

model = YOLO("yolov8x.pt")
output_frame = None
lock = threading.Lock()

# --- 웹 스트리밍 서버 ---
app = Flask(__name__)

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            _, buffer = cv2.imencode(".jpg", output_frame)
            frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- WebSocket 서버 ---
async def handle_ws(websocket):
    global output_frame, lock
    try:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            img_bytes = bytes(data["img"])
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame_start_time = time.time()
            # YOLO 처리
            yolo_start_time = time.time()
            results = model(frame, imgsz=640, verbose=False)
            yolo_end_time = time.time()
            annotated_frame = results[0].plot()
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            names = results[0].names if hasattr(results[0], "names") else model.names
            objects = [(names[class_id], float(conf)) for class_id, conf in zip(class_ids, confidences)]
            # 결과 JSON 전송
            det_list = [
                {"label": name, "confidence": conf}
                for name, conf in objects
            ]
            await websocket.send(json.dumps({"detections": det_list}))
            # 웹 스트림용 전역 변수 업데이트
            with lock:
                output_frame = annotated_frame
            frame_end_time = time.time()
            print(f"[SERVER DEBUG] Frame total time: {(frame_end_time - frame_start_time)*1000:.2f} ms, YOLO time: {(yolo_end_time - yolo_start_time)*1000:.2f} ms")
    except websockets.exceptions.ConnectionClosed:
        print("[SERVER] WebSocket connection closed")

async def start_ws_server():
    async with websockets.serve(handle_ws, "0.0.0.0", 8000):
        print("[SERVER] WebSocket listening on ws://0.0.0.0:8000")
        await asyncio.Future()

def start_flask():
    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    asyncio.run(start_ws_server())
