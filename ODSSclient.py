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
ttsRequest = None
ttsThreadRunning = False
ttsBlockUntil = 0

def tts_worker():
    global ttsRequest, ttsThreadRunning, ttsBlockUntil
    while True:
        now = time.time()
        if ttsRequest is not None and now > ttsBlockUntil:
            text = ttsRequest
            ttsRequest = None
            ttsPath = f"tts.mp3"
            
            tts = gTTS(text=text, lang='ko')
            tts.save(ttsPath)
            with ttsLock:
                subprocess.run(
                    ["mpg123", ttsPath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            ttsBlockUntil = time.time() + 2
        else:
            time.sleep(0.05)

def _startTtsWorker():
    global ttsThreadRunning
    if not ttsThreadRunning:
        t = threading.Thread(target=tts_worker, daemon=True)
        t.start()
        ttsThreadRunning = True

def _speakTtsAsync(text):
    global ttsRequest
    ttsRequest = text

class CameraThread(threading.Thread):
    def __init__(self, device_path):
        super().__init__()
        self.device_path = device_path
        # DirectShow 사용 (Windows)
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_DSHOW)
        self.ret = False
        self.frame = None
        self.running = True

        if not self.cap.isOpened():
            self.running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def run(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

def _findSingleCameraDevice():
    # 카메라 인덱스 탐색 (Windows용)
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return 0

def _getDirection(idx, total):
    directionMap = ["Front", "Left", "Right"]
    if idx < len(directionMap):
        return directionMap[idx]
    else:
        return f"Extra {idx}"

DANGER_LABELS = {"person", "car", "bicycle", "motorcycle", "bus", "truck"}

async def _sendFrames(cam, cam_idx, total_cams):
    lastTtsMsg = None
    lastTtsTime = 0
    direction = _getDirection(cam_idx, total_cams)
    windowName = f'{direction} (Camera {cam_idx} - {cam.device_path})'

    uniqueCode = 'A7X2bQ9cLmY1'

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
                "uniqueCode": uniqueCode
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
                        _speakTtsAsync(ttsMsg)
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
    device = _findSingleCameraDevice()
    print("사용 카메라:", device)

    cam = CameraThread(device)
    if not cam.running:
        print("카메라 없음")
        return
    cam.start()

    # WebSocket 전송 (테스트용으로 단일 카메라만)
    await _sendFrames(cam, 0, 1)

    cam.stop()
    cam.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    _startTtsWorker()
    asyncio.run(main())
