import websockets
import asyncio
import json
import time
import cv2
import subprocess
import threading
from gtts import gTTS

ip = '222.231.24.5'  

ttsLock = threading.Lock()

def speak_tts(text):
    if ttsLock.locked():
        return
    with ttsLock:
        try:
            print("[TTS]", text)
            tts = gTTS(text=text, lang='ko')
            tts.save("tts.mp3")
            subprocess.run(
                ["mpg123", "tts.mp3"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print("[ERROR] TTS 실패:", e)

async def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 카메라 열기 실패")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 15)

    print("[INFO] 웹캠 시작됨")

    cam_idx = 0
    window_name = "Front (Camera 0)"

    last_tts_msg = None
    last_tts_time = 0

    async with websockets.connect(f'ws://{ip}:8000') as ws:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            frame = cv2.resize(frame, (640, 360))
            _, img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            payload = {
                "img": list(img.tobytes()),
                "cam_idx": cam_idx
            }
            await ws.send(json.dumps(payload))

            result = await ws.recv()
            result_dict = json.loads(result)

            if "tts" in result_dict:
                tts_msg = result_dict["tts"]
                now = time.time()
                if tts_msg:
                    if tts_msg != last_tts_msg or (now - last_tts_time) > 2:
                        threading.Thread(target=speak_tts, args=(tts_msg,), daemon=True).start()
                        last_tts_msg = tts_msg
                        last_tts_time = now

            if "detections" in result_dict:
                for det in result_dict["detections"]:
                    x1, y1, x2, y2 = det["box"]
                    label = det.get("kor_label") or det.get("koreanLabel") or det["label"]
                    conf = det["confidence"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 종료됨")

if __name__ == "__main__":
    asyncio.run(main())