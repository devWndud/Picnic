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

def speak_tts(text):
    if ttsLock.locked():
        return
    with ttsLock:
        try:
            print(text)
            tts = gTTS(text=text, lang='ko')
            tts.save("tts.mp3")
            subprocess.run(
                ["mpg123", "tts.mp3"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print("TTS ERROR:", e)

def play_abnormal_notice():
    speak_tts("비정상 객체가 인식되었습니다")

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

def get_direction(idx, total):
    direction_map = ["Front", "Left", "Right"]
    if idx < len(direction_map):
        return direction_map[idx]
    else:
        return f"Extra {idx}"

async def send_frames(cam, cam_idx, total_cams):
    last_tts_msg = None
    last_tts_time = 0
    direction = get_direction(cam_idx, total_cams)
    window_name = f'{direction} (Camera {cam_idx} - {cam.device_path})'

    unique_code = 'A7X2bQ9cLmY1'

    async with websockets.connect(f'ws://{ip}:8000') as ws:
        while True:
            now = time.time()
            if now - last_tts_time < 2:
                await asyncio.sleep(0.01)
                continue

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

            # 2단계: 자가 진단 (TTS)
            if cam_idx == 0 and "tts" in result_dict:
                tts_msg = result_dict["tts"]
                if tts_msg:
                    if tts_msg != last_tts_msg or (now - last_tts_time) > 2:
                        print(f"[{window_name}]", tts_msg)
                        speak_tts(tts_msg)
                        last_tts_msg = tts_msg
                        last_tts_time = time.time()

            if "detections" in result_dict:
                for det in result_dict["detections"]:
                    x1, y1, x2, y2 = det["box"]
                    label = det["kor_label"] if "kor_label" in det else det["label"]
                    conf = det["confidence"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{direction}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'{direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyWindow(window_name)

async def main():
    device_paths = _findVideoDevices()
    print("[INFO] Found devices:", device_paths)
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
        print("No camera")

    for cam in cameras:
        cam.stop()
    for cam in cameras:
        cam.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())