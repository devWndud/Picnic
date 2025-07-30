
import asyncio
import websockets
import threading
import numpy as np
import cv2
import json
import time
from flask import Flask, Response
from ultralytics import YOLO
import base64
import requests
from openai import OpenAI
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model = YOLO('yolov8x.pt')

latestFrames = {}
frameLock = threading.Lock()

recentBottleBoxes = []

app = Flask(__name__)

def start_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

OPENAI_API_KEY = "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"  # 여기에 본인 토큰만 입력하면 됨
client = OpenAI(api_key=OPENAI_API_KEY)

gpt_lock = threading.Lock()

def updateLatestFrame(img, uniqueCode):
    global latestFrames
    with frameLock:
        latestFrames[uniqueCode] = img.copy()

@app.route('/video_feed/<uniqueCode>')
def video_feed(uniqueCode):
    def generate():
        while True:
            with frameLock:
                frame = latestFrames.get(uniqueCode)
                if frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def _isAbnormal(det, img_shape):
    return det["label"] == "bottle"

def _isRecentBottle(box):
    now = time.time()
    for prev_box, ts in recentBottleBoxes:
        if (now - ts) < 10:
            return True
    return False

def _askGptAboutBbox(label, box, croppedImg):
    if not gpt_lock.acquire(blocking=False):
        print("[DEBUG] GPT Vision API가 이미 동작 중이므로 이번 요청은 건너뜁니다.")
        return "비정상"
    try:
        import time
        start_time = time.time()
        _, buffer = cv2.imencode('.jpg', croppedImg)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        prompt = (
            f"이 이미지는 {label}의 바운딩박스({box})입니다."
            f"바운딩박스 안에 실제로 {label}이 잘 인식되어 있나요? "
            f"현재 저희는 bottle은 무조건 비정상객체로 정의하고 있습니다"
            f"'정상' 또는 '비정상'으로만 답해주세요."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10,
        )
        elapsed = time.time() - start_time
        print(f"[DEBUG] GPT Vision API 추론 시간: {elapsed:.2f}초")
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print("[ERROR] GPT Vision API 호출 실패:", e)
        return "비정상"
    finally:
        gpt_lock.release()

width, height = 640, 360

def computeRisk(det, frameWidth):
    risk = det.get("confidence", 0)
    label = det.get("label", "")

    x1, y1, x2, y2 = det["box"]

    boxArea = (x2 - x1) * (y2 - y1)
    minPersonArea = 30000

    if label == "person" and boxArea < minPersonArea:
        return 0
    
    boxCenter = (x1 + x2) / 2
    leftThird = frameWidth / 3
    rightThird = 2 * frameWidth / 3

    if leftThird < boxCenter < rightThird:
        risk += 0.3

    dangerLabels = ["car", "person", "motorcycle", "bus", "truck", "bicycle"]

    if label.lower() in dangerLabels:
        risk += 0.3

    return min(risk, 1.0)

def _getObstacleZone(detections, frameWidth):
    minPersonArea = 30000
    dangerLabels = ["car", "person", "motorcycle", "bus", "truck", "bicycle"]

    left = frameWidth / 3
    centerLeft = left + (left / 2)
    centerRight = 2 * left + (left / 2)

    right = frameWidth
    zones = []

    for det in detections:
        label = det.get("label", "")
        x1, y1, x2, y2 = det["box"]

        boxArea = (x2 - x1) * (y2 - y1)
        risk = det.get("risk", det.get("confidence", 0))

        if label.lower() not in dangerLabels:
            continue
        elif label == "person" and boxArea < minPersonArea:
            continue
        elif risk < 0.5:
            continue

        boxCenter = (x1 + x2) / 2

        if boxCenter < left:
            zones.append("left")

        elif left <= boxCenter < centerLeft:
            zones.append("centerLeft")
        elif centerLeft <= boxCenter < 2*left:
            zones.append("centerRight")
        else:
            zones.append("right")
    return zones

async def _handleClient(websocket):
    print("[INFO] Client connected")
    try:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)

            cam_idx = data.get("cam_idx", 0)
            uniqueCode = data.get("uniqueCode", "default")

            imgBytes = bytes(data["img"])
            img = cv2.imdecode(np.frombuffer(imgBytes, np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                continue

            results = model(img, verbose=False)
            detections = []
            gptCalledThisFrame = False  # 한 프레임에 한 번만 GPT 호출
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    korean_label = label
                    det = {
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "label": label,
                        "koreanLabel": korean_label
                    }
                    if _isAbnormal(det, img.shape):
                        if det["label"] == "bottle" and _isRecentBottle(det["box"]):
                            det["abnormal"] = False

                        elif det["label"] == "bottle" and not gptCalledThisFrame:
                            det["abnormal"] = True
                            recentBottleBoxes.append((det["box"], time.time()))
                            croppedImg = img[y1:y2, x1:x2]
                            gpt_result = _askGptAboutBbox(label, [x1, y1, x2, y2], croppedImg)
                            print(f"GPT : {gpt_result}", flush=True)

                            if "정상" in gpt_result:
                                det["abnormal"] = False
                                
                            gptCalledThisFrame = True
                        else:
                            det["abnormal"] = False
                    else:
                        det["abnormal"] = False

                    detections.append(det)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, korean_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if cam_idx == 0:
                updateLatestFrame(img, uniqueCode)

            for det in detections:
                det["risk"] = computeRisk(det, width)

            zones = _getObstacleZone(detections, width)
            ttsMsg = ""
            ttsDebug = None
            zoneCenter = None
            abnormal_detected = any(det.get("label", "") == "bottle" for det in detections)

            if abnormal_detected:
                ttsMsg = "비정상 객체가 인식되었습니다"
            else:
                for det in detections:
                    label = det.get("label", "")
                    x1, y1, x2, y2 = det["box"]
                    boxArea = (x2 - x1) * (y2 - y1)
                    risk = det.get("risk", det.get("confidence", 0))

                    if label.lower() not in ["car", "person", "motorcycle", "bus", "truck", "bicycle"]:
                        continue

                    if label == "person" and boxArea < 30000:
                        continue

                    if risk < 0.5:
                        continue

                    boxCenter = (x1 + x2) / 2
                    zoneCenter = boxCenter
                    break

                if "left" in zones:
                    ttsMsg = "오른쪽으로 가세요."
                    ttsDebug = zoneCenter
                elif "right" in zones:
                    ttsMsg = "왼쪽으로 가세요."
                    ttsDebug = zoneCenter
                elif "centerLeft" in zones or "centerRight" in zones:
                    ttsMsg = "잠시 정지하세요"
                    ttsDebug = zoneCenter
            if ttsMsg:
                print(f"{ttsMsg} / center : {ttsDebug}", flush=True)
            await websocket.send(json.dumps({
                "tts": ttsMsg,
                "detections": detections
            }))

    except websockets.exceptions.ConnectionClosed:
        pass

async def start_websocket_server():
    async with websockets.serve(_handleClient, "0.0.0.0", 8000):
        print("WebSocket server started on ws://0.0.0.0:8000")
        await asyncio.Future()

def main():
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    asyncio.run(start_websocket_server())

if __name__ == '__main__':
    main()