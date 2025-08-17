import socket
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import struct
import time

model = YOLO('yolov8x.pt')

def handle_client(conn, addr):
    print(f"[SERVER] Connected by {addr}")
    try:
        while True:
            raw_msglen = recvall(conn, 4)
            if not raw_msglen:
                break
            msglen = struct.unpack('>I', raw_msglen)[0]
            frame_data = recvall(conn, msglen)
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            start = time.time()
            results = model(frame, imgsz=640, verbose=False)
            annotated_frame = results[0].plot()
            print(f"[SERVER] YOLO Time: {time.time() - start:.3f}s")

            _, img_encoded = cv2.imencode('.jpg', annotated_frame)
            data = img_encoded.tobytes()
            conn.sendall(struct.pack('>I', len(data)) + data)

    except Exception as e:
        print(f"[SERVER ERROR] {e}")
    finally:
        conn.close()

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

def start_server(host='0.0.0.0', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"[SERVER] Listening on {host}:{port}")

    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
