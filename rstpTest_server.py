import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import threading
import json
import time
import imageio.v3 as iio
import sys

MODE = "webp"  # 'webp' or 'h264'
UDP_PORT = 5000

model = YOLO("yolov8x.pt")
output_frame = None
lock = threading.Lock()

# --- 웹 스트리밍 서버 ---
app = FastAPI()

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

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global output_frame, lock
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("end"):
                break
            encoding = data.get("encoding", "webp")
            frame_id = data.get("frame_id")
            if encoding == "webp":
                img_bytes = bytes(data["img"])
                frame = iio.imread(img_bytes, plugin="pillow")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                continue
            frame_start_time = time.time()
            yolo_start_time = time.time()
            results = model(frame, imgsz=640, verbose=False)
            yolo_end_time = time.time()
            annotated_frame = results[0].plot()
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            names = results[0].names if hasattr(results[0], "names") else model.names
            objects = [(names[class_id], float(conf)) for class_id, conf in zip(class_ids, confidences)]
            det_list = [
                {"label": name, "confidence": conf}
                for name, conf in objects
            ]
            await websocket.send_text(json.dumps({"detections": det_list, "frame_id": frame_id}))
            with lock:
                output_frame = annotated_frame
            frame_end_time = time.time()
            print(f"[SERVER DEBUG] Frame total time: {(frame_end_time - frame_start_time)*1000:.2f} ms, YOLO time: {(yolo_end_time - yolo_start_time)*1000:.2f} ms")
    except WebSocketDisconnect:
        print("[SERVER] WebSocket connection closed")

def udp_h264_receiver():
    global output_frame, lock
    cap = cv2.VideoCapture(f'udpsrc port={UDP_PORT} ! application/x-rtp, encoding-name=H264 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame_start_time = time.time()
        yolo_start_time = time.time()
        results = model(frame, imgsz=640, verbose=False)
        yolo_end_time = time.time()
        annotated_frame = results[0].plot()
        with lock:
            output_frame = annotated_frame
        frame_end_time = time.time()
        print(f"[SERVER H264] Frame total time: {(frame_end_time - frame_start_time)*1000:.2f} ms, YOLO time: {(yolo_end_time - yolo_start_time)*1000:.2f} ms")
    cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
