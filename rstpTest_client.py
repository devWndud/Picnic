import socket
import cv2
import struct
import numpy as np
import threading

SERVER_IP = '222.231.24.5'
SERVER_PORT = 9999

def send_video(video_path, window_name):
    cap = cv2.VideoCapture(video_path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[INFO] End of {video_path}")
                break

            # 인코딩 및 길이 전송
            _, encoded = cv2.imencode('.jpg', frame)
            data = encoded.tobytes()
            sock.sendall(struct.pack(">I", len(data)) + data)

            # 처리된 프레임 수신
            data_len = sock.recv(4)
            if not data_len:
                break
            length = struct.unpack(">I", data_len)[0]

            result = b""
            while len(result) < length:
                packet = sock.recv(length - len(result))
                if not packet:
                    return
                result += packet

            img_array = np.frombuffer(result, dtype=np.uint8)
            processed_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            cv2.imshow(window_name, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] {video_path}: {e}")
    finally:
        cap.release()
        sock.close()
        cv2.destroyWindow(window_name)

if __name__ == "__main__":
    videos = [
        ("video1.mp4", "Client 1"),
        ("video2.mp4", "Client 2"),
        # 더 추가 가능
    ]

    threads = []
    for path, name in videos:
        t = threading.Thread(target=send_video, args=(path, name))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
