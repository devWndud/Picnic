import socket
import cv2
import struct

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
        _, buffer = cv2.imencode(".jpg", frame)
        data = buffer.tobytes()
        sock.sendall(struct.pack(">I", len(data)) + data)

        # YOLO 텍스트 결과 수신
        raw_len = recv_all(sock, 4)
        if not raw_len:
            break
        msg_len = struct.unpack(">I", raw_len)[0]
        text_data = recv_all(sock, msg_len)
        text = text_data.decode()
        print(f"[Client {client_id}] Detected: {text}")

    cap.release()
    sock.close()

if __name__ == "__main__":
    videos = ["101.mp4", "102.mp4"]
    for i, v in enumerate(videos):
        send_video(v, i+1)
