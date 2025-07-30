
import websockets
import asyncio
import json
import time
import cv2
import subprocess
import threading
import os
from gtts import gTTS

ip = '222.231.24.5'
ttsLock = threading.Lock()
tts_request = None
tts_thread_running = False
tts_block_until = 0

def tts_worker():
    global tts_request, tts_thread_running, tts_block_until
    while True:
        now = time.time()
        if tts_request is not None and now > tts_block_until:
            text = tts_request
            tts_request = None
            tts_path = f"tts_cache_{hash(text)}.mp3"
            if not os.path.exists(tts_path):
                tts = gTTS(text=text, lang='ko')
                tts.save(tts_path)
            with ttsLock:
                subprocess.run(
                    ["mpg123", tts_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            tts_block_until = time.time() + 2  # 2초 블록
        else:
            time.sleep(0.05)

def start_tts_worker():
    global tts_thread_running
    if not tts_thread_running:
        t = threading.Thread(target=tts_worker, daemon=True)
        t.start()
        tts_thread_running = True

def speak_tts_async(text):
    global tts_request
    tts_request = text

class CameraThread(threading.Thread):
    def __init__(self, device_path):
        super().__init__()
        self.device_path = device_path
        self.cap = cv2.VideoCapture(self.device_path)
        self.ret = False
        self.frame = None
        self.running = True

        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open {self.device_path}")
            self.running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        print(f"[INFO] Opened {self.device_path}")

    def run(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                print(f"[WARN] Failed to read frame from {self.device_path}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        print(f"[INFO] Released {self.device_path}")

def _findVideoDevices():
    devices = []
    for i in range(10):
        path = f"/dev/video{i}"
        if os.path.exists(path):
            devices.append(path)
    return devices

def _getDirection(idx, total):
    directionMap = ["Front", "Left", "Right"]
    if idx < len(directionMap):
        return directionMap[idx]
    else:
        return f"Extra {idx}"

DANGER_LABELS = {"person", "car", "bicycle", "motorcycle", "bus", "truck"}

async def send_frames(cam, cam_idx, total_cams):
    lastTtsMsg = None
    lastTtsTime = 0
    direction = _getDirection(cam_idx, total_cams)
    windowName = f'{direction} (Camera {cam_idx} - {cam.device_path})'

    unique_code = 'A7X2bQ9cLmY1'

    async with websockets.connect(f'ws://{ip}:8000') as ws:
        while True:
            if not cam.ret or cam.frame is None:
                await asyncio.sleep(0.01)
                continue

            frame = cam.frame.copy()
            frame = cv2.resize(frame, (640, 360))
            _, img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            payload = {
                "img": list(img.tobytes()),
                "cam_idx": cam_idx,
                "unique_code": unique_code
            }
            await ws.send(json.dumps(payload))
            result = await ws.recv()
            result_dict = json.loads(result)

            if cam_idx == 0 and "tts" in result_dict:
                ttsMsg = result_dict["tts"]
                now = time.time()
                if ttsMsg:
                    if ttsMsg != lastTtsMsg or (now - lastTtsTime) > 2:
                        print(f"[{windowName}]", ttsMsg)
                        speak_tts_async(ttsMsg)
                        lastTtsMsg = ttsMsg
                        lastTtsTime = now

            if "detections" in result_dict:
                for det in result_dict["detections"]:
                    label = det["kor_label"] if "kor_label" in det else det["label"]
                    if label not in DANGER_LABELS:
                        continue  # 바닥, 벽 등은 무시
                    x1, y1, x2, y2 = det["box"]
                    conf = det["confidence"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{direction}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'{direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow(windowName, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyWindow(windowName)

async def main():
    device_paths = _findVideoDevices()

    print("디바이스 : ", device_paths)

    cameras = [CameraThread(dev) for dev in device_paths]
    for cam in cameras:
        if cam.running:
            cam.start()

    tasks = []
    for idx, cam in enumerate(cameras):
        if cam.running:
            tasks.append(asyncio.create_task(send_frames(cam, idx, len(cameras))))

    if tasks:
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            pass
    else:
        print("카메라 없음")

    for cam in cameras:
        cam.stop()
    for cam in cameras:
        cam.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_tts_worker()
    asyncio.run(main())
