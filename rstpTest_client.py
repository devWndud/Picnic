import websockets
import asyncio
import json
import cv2
import time

SERVER_IP = "222.231.24.5"
VIDEO_PATH = "101.mp4"

async def send_video(video_path, client_id):
    cap = cv2.VideoCapture(video_path)
    async with websockets.connect(f'ws://{SERVER_IP}:8000') as ws:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))
            _, img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            payload = {
                "img": list(img.tobytes()),
                "cam_idx": 0,
                "uniqueCode": "test"
            }
            send_start = time.time()
            await ws.send(json.dumps(payload))
            result = await ws.recv()
            send_end = time.time()
            result_dict = json.loads(result)
            print(f"[Client {client_id}] Detected: {result_dict.get('detections')}")
            print(f"[CLIENT DEBUG] Frame round-trip time: {(send_end-send_start)*1000:.2f} ms")
    cap.release()

if __name__ == "__main__":
    asyncio.run(send_video(VIDEO_PATH, 1))
