import socket
import cv2
import struct
import time

SERVER_IP = "222.231.24.5"
SERVER_PORT = 9999

def recv_all(sock, length):
    buf = bytearray(length)
    view = memoryview(buf)
    while length:
        n = sock.recv_into(view, length)
        if n == 0:
            return None
        view = view[n:]
        length -= n
    return bytes(buf)

def send_video(video_path, client_id):
    cap = cv2.VideoCapture(video_path)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 전송
        send_start_time = time.time()
        _, buffer = cv2.imencode(".jpg", frame)
        data = buffer.tobytes()
        sock.sendall(struct.pack(">I", len(data)) + data)

        # YOLO 텍스트 결과 수신
        raw_len = recv_all(sock, 4)
        if not raw_len:
            break
        msg_len = struct.unpack(">I", raw_len)[0]
        text_data = recv_all(sock, msg_len)
        send_end_time = time.time()
        text = text_data.decode()
        print(f"[Client {client_id}] Detected: {text}")
        print(f"[CLIENT DEBUG] Frame round-trip time: {(send_end_time - send_start_time)*1000:.2f} ms")

    cap.release()
    sock.close()

if __name__ == "__main__":
    videos = ["101.mp4", "102.mp4"]
    for i, v in enumerate(videos):
        send_video(v, i+1)
