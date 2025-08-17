import socket
import cv2
import struct
import time
import threading

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

def client_worker(video_path, client_id, server_ip='[AWS_PUBLIC_IP]', port=9999):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, port))

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()

        # 전송
        start = time.time()
        sock.sendall(struct.pack('>I', len(data)) + data)

        # 수신
        raw_len = recv_all(sock, 4)
        if not raw_len:
            break
        msg_len = struct.unpack('>I', raw_len)[0]
        result_data = recv_all(sock, msg_len)

        # 디코딩 및 출력
        nparr = np.frombuffer(result_data, np.uint8)
        result_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        elapsed = time.time() - start
        print(f"[Client {client_id}] Total: {elapsed:.3f}s")

        cv2.imshow(f'Client {client_id}', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_paths = ['video1.mp4', 'video2.mp4']  # 여기에 MP4 파일 경로들 입력
    threads = []
    for i, path in enumerate(video_paths):
        t = threading.Thread(target=client_worker, args=(path, i+1))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
