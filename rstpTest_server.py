import socket
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import time
import struct

model = YOLO("yolov8x.pt")

def handle_client(conn, addr):
    print(f"[INFO] Connected by {addr}")
    try:
        while True:
            data_len = conn.recv(4)
            if not data_len:
                break
            length = struct.unpack(">I", data_len)[0]

            data = b""
            while len(data) < length:
                packet = conn.recv(length - len(data))
                if not packet:
                    return
                data += packet

            img_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # --- YOLO 처리 시간 측정 ---
            start_yolo = time.time()
            results = model(frame)[0]
            annotated = results.plot()
            yolo_time = time.time() - start_yolo
            print(f"[SERVER] YOLO Time: {yolo_time:.3f}s")

            _, encoded = cv2.imencode('.jpg', annotated)
            to_send = encoded.tobytes()
            conn.sendall(struct.pack(">I", len(to_send)) + to_send)

    except Exception as e:
        print(f"[ERROR] Client {addr} error: {e}")
    finally:
        conn.close()

def start_server(host='0.0.0.0', port=9999):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"[INFO] Server listening on {host}:{port}")

    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()

if __name__ == "__main__":
    start_server()
