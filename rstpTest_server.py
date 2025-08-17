import socket
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import struct
from flask import Flask, Response
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

# --- TCP 클라이언트 처리 ---
def recvall(sock, length):
    buf = bytearray(length)
    view = memoryview(buf)
    while length:
        n = sock.recv_into(view, length)
        if n == 0:
            return None
        view = view[n:]
        length -= n
    return bytes(buf)

def handle_client(conn, addr):
    global output_frame, lock
    print(f"[SERVER] Connected by {addr}")
    try:
        while True:
            frame_start_time = time.time()  # Start timing for the whole frame
            raw_len = recvall(conn, 4)
            if not raw_len:
                break
            msg_len = struct.unpack(">I", raw_len)[0]
            frame_data = recvall(conn, msg_len)
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

            # 결과 텍스트 클라이언트 전송
            text = "; ".join([f"{name}:{conf:.2f}" for name, conf in objects])
            text_bytes = text.encode()
            conn.sendall(struct.pack(">I", len(text_bytes)) + text_bytes)

            # 웹 스트림용 전역 변수 업데이트
            with lock:
                output_frame = annotated_frame

            frame_end_time = time.time()
            print(f"[SERVER DEBUG] Frame total time: {(frame_end_time - frame_start_time)*1000:.2f} ms, YOLO time: {(yolo_end_time - yolo_start_time)*1000:.2f} ms")

    except Exception as e:
        print(f"[SERVER ERROR] {e}")
    finally:
        conn.close()

def start_tcp_server(host="0.0.0.0", port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"[SERVER] TCP listening on {host}:{port}")

    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

def start_background_tcp_server():
    threading.Thread(target=start_tcp_server, daemon=True).start()

start_background_tcp_server()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
