import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response
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
            if data.get("end"):
                break
            encoding = data.get("encoding", "webp")
            frame_id = data.get("frame_id")
            if encoding == "webp":
                img_bytes = bytes(data["img"])
                frame = iio.imread(img_bytes, plugin="pillow")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                continue  # Only handle webp in ws mode
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
            await websocket.send(json.dumps({"detections": det_list, "frame_id": frame_id}))
            with lock:
                output_frame = annotated_frame
            frame_end_time = time.time()
            print(f"[SERVER DEBUG] Frame total time: {(frame_end_time - frame_start_time)*1000:.2f} ms, YOLO time: {(yolo_end_time - yolo_start_time)*1000:.2f} ms")
    except websockets.exceptions.ConnectionClosed:
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

def start_flask():
    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    if MODE == "webp":
        def start_websocket_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            server = loop.run_until_complete(websockets.serve(handle_ws, "0.0.0.0", 8000))
            loop.run_forever()
        websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
        websocket_thread.start()
        websocket_thread.join()
    elif MODE == "h264":
        udp_thread = threading.Thread(target=udp_h264_receiver, daemon=True)
        udp_thread.start()
        udp_thread.join()
