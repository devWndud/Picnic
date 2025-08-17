import socket
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import struct

model = YOLO("yolov8x.pt")

def handle_client(conn, addr):
    print(f"[INFO] Connected by {addr}")
    try:
        while True:
            # 먼저 데이터 길이 수신 (4바이트)
            data_len = conn.recv(4)
            if not data_len:
                break
            length = struct.unpack(">I", data_len)[0]

            # 본문 이미지 데이터 수신
            data = b""
            while len(data) < length:
                packet = conn.recv(length - len(data))
                if not packet:
                    return
                data += packet

            # 디코딩
            img_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # YOLO 처리
            results = model(frame)[0]
            annotated = results.plot()

            # 다시 인코딩 후 전송
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
