import asyncio
import threading
import time
import os
import io
import json
import base64
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import websockets
from flask import Flask, Response, request
from ultralytics import YOLO
import torch

# stdout UTF-8
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------
# Config
# ----------------------
RTSP_URLS: List[str] = [
    # 예시: 환경변수 또는 직접 입력
    os.environ.get('RTSP0', ''),
    os.environ.get('RTSP1', ''),
    os.environ.get('RTSP2', ''),
]
RTSP_URLS = [u for u in RTSP_URLS if u]

FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5001
WS_HOST = '0.0.0.0'
WS_PORT = 8001

IMG_SIZE = 960  # 1280 → 960로 지연과 품질 균형
MAX_DET = 100

# Annotated RTSP publish settings
ANNOTATED_RTSP_ENABLE = True
ANNOTATED_FPS = 30
ANNOTATED_SIZE = (640, 360)  # match processing size
ANNOTATED_BITRATE = '2M'
ANNOTATED_URL_PREFIX = 'rtsp://127.0.0.1:8554/annotated'

# ----------------------
# GPU / Model setup
# ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

model = YOLO('yolov8x.pt')
model.to(device)
try:
    model.fuse()
except Exception:
    pass

model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = MAX_DET

print("YOLO loaded.")

# ----------------------
# Shared state
# ----------------------
rawFrames: Dict[int, np.ndarray] = {}
processedFrames: Dict[int, np.ndarray] = {}
rawLock = threading.Lock()
procLock = threading.Lock()

# For WebSocket broadcasts
ws_clients: "set[websockets.WebSocketServerProtocol]" = set()
ws_lock = threading.Lock()

# ----------------------
# Annotated RTSP publisher
# ----------------------
class AnnotatedPublisher(threading.Thread):
    def __init__(self, stream_id: int, size: tuple, fps: int, bitrate: str):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.size = size
        self.fps = fps
        self.bitrate = bitrate
        self.running = True
        self.proc = None
        self.lock = threading.Lock()
        self.latest: Optional[np.ndarray] = None

    def start_ffmpeg(self):
        url = f"{ANNOTATED_URL_PREFIX}{self.stream_id}"
        w, h = self.size
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-re',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-b:v', self.bitrate,
            '-maxrate', self.bitrate,
            '-bufsize', self.bitrate,
            '-g', str(self.fps*2),
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            url
        ]
        try:
            import subprocess
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            print(f"[Annotated {self.stream_id}] ffmpeg started -> {url}")
        except Exception as e:
            print(f"[Annotated {self.stream_id}] ffmpeg start failed: {e}")
            self.proc = None

    def run(self):
        self.start_ffmpeg()
        frame_interval = 1.0 / max(1, self.fps)
        next_deadline = time.perf_counter()
        while self.running:
            with self.lock:
                frame = None if self.latest is None else self.latest.copy()
            if frame is None:
                time.sleep(0.005)
                continue
            # Ensure size
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size)
            if self.proc is None or self.proc.stdin is None or self.proc.poll() is not None:
                self.start_ffmpeg()
                time.sleep(0.2)
                continue
            try:
                self.proc.stdin.write(frame.tobytes())
            except Exception:
                # restart ffmpeg on write failure
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None
            # pace
            next_deadline += frame_interval
            now = time.perf_counter()
            if next_deadline > now:
                time.sleep(next_deadline - now)
            else:
                next_deadline = now
        # cleanup
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
            except Exception:
                pass

    def update(self, frame: np.ndarray):
        with self.lock:
            self.latest = frame

publishers: Dict[int, AnnotatedPublisher] = {}

# ----------------------
# Flask app for MJPEG
# ----------------------
app = Flask(__name__)

@app.route('/video_feed/<int:stream_id>')
def video_feed(stream_id: int):
    # Query params: w (width), q (jpeg quality), fps
    try:
        tgt_w = int(request.args.get('w', '0'))
    except Exception:
        tgt_w = 0
    try:
        jpeg_q = int(request.args.get('q', '70'))
    except Exception:
        jpeg_q = 70
    try:
        fps = float(request.args.get('fps', '15'))
    except Exception:
        fps = 15.0
    fps = max(1.0, min(60.0, fps))
    frame_interval = 1.0 / fps

    def generate():
        next_deadline = time.perf_counter()
        while True:
            with procLock:
                frame = processedFrames.get(stream_id)
            if frame is None:
                time.sleep(0.005)
                continue

            out = frame
            if tgt_w and frame.shape[1] != tgt_w:
                h = int(frame.shape[0] * (tgt_w / frame.shape[1]))
                out = cv2.resize(frame, (tgt_w, h))

            ok, jpeg = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
            if not ok:
                continue
            frame_bytes = jpeg.tobytes()
            headers = (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(frame_bytes)).encode('ascii') + b'\r\n\r\n'
            )
            yield headers + frame_bytes + b'\r\n'

            # pace
            next_deadline += frame_interval
            now = time.perf_counter()
            if next_deadline > now:
                time.sleep(next_deadline - now)
            else:
                next_deadline = now
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------
# Utility
# ----------------------
width_ref = 640

def computeRisk(det: Dict[str, Any], frameWidth: int) -> float:
    risk = float(det.get('confidence', 0))
    label = det.get('label', '')
    x1, y1, x2, y2 = det['box']
    boxArea = (x2 - x1) * (y2 - y1)
    minPersonArea = 30000
    if label == 'person' and boxArea < minPersonArea:
        return 0.0
    boxCenter = (x1 + x2) / 2
    leftThird = frameWidth / 3
    rightThird = 2 * frameWidth / 3
    if leftThird < boxCenter < rightThird:
        risk += 0.3
    if label.lower() in ['car', 'person', 'motorcycle', 'bus', 'truck', 'bicycle']:
        risk += 0.3
    return float(min(risk, 1.0))

def getZones(detections: List[Dict[str, Any]], frameWidth: int) -> List[str]:
    minPersonArea = 30000
    danger = ['car', 'person', 'motorcycle', 'bus', 'truck', 'bicycle']
    left = frameWidth / 3
    centerLeft = left + (left / 2)
    zones: List[str] = []
    for det in detections:
        label = det.get('label', '')
        x1, y1, x2, y2 = det['box']
        boxArea = (x2 - x1) * (y2 - y1)
        risk = det.get('risk', det.get('confidence', 0))
        if label.lower() not in danger:
            continue
        if label == 'person' and boxArea < minPersonArea:
            continue
        if risk < 0.5:
            continue
        boxCenter = (x1 + x2) / 2
        if boxCenter < left:
            zones.append('left')
        elif left <= boxCenter < centerLeft:
            zones.append('centerLeft')
        elif centerLeft <= boxCenter < 2*left:
            zones.append('centerRight')
        else:
            zones.append('right')
    return zones

async def ws_broadcast(message: str):
    dead = []
    with ws_lock:
        for ws in list(ws_clients):
            try:
                await ws.send(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                ws_clients.remove(ws)
            except KeyError:
                pass

# ----------------------
# Capture & Process threads
# ----------------------
class CaptureThread(threading.Thread):
    def __init__(self, stream_id: int, url: str):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.url = url
        self.running = True

    def run(self):
        while self.running:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                print(f"[Capture {self.stream_id}] open failed. retry in 2s")
                time.sleep(2)
                continue
            print(f"[Capture {self.stream_id}] started")
            try:
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[Capture {self.stream_id}] read fail, reopen")
                        break
                    # Optional downscale for stability
                    frame = cv2.resize(frame, (640, 360))
                    with rawLock:
                        rawFrames[self.stream_id] = frame
                    time.sleep(0.001)
            finally:
                cap.release()
                time.sleep(0.5)

class ProcessThread(threading.Thread):
    def __init__(self, stream_id: int):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.running = True
        if ANNOTATED_RTSP_ENABLE:
            pub = AnnotatedPublisher(stream_id, ANNOTATED_SIZE, ANNOTATED_FPS, ANNOTATED_BITRATE)
            publishers[stream_id] = pub
            pub.start()

    def run(self):
        while self.running:
            t0 = time.perf_counter()
            with rawLock:
                frame = rawFrames.get(self.stream_id)
            if frame is None:
                time.sleep(0.01)
                continue

            img = frame.copy()
            frame_w = int(img.shape[1])

            # Inference
            t_infer0 = time.perf_counter()
            with torch.inference_mode():
                results = model(img, verbose=False, device=device, half=True, imgsz=IMG_SIZE)
            t_infer1 = time.perf_counter()
            infer_ms = (t_infer1 - t_infer0) * 1000

            detections: List[Dict[str, Any]] = []
            t_post0 = time.perf_counter()
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    try:
                        conf = float(box.conf[0])
                    except Exception:
                        conf = 0.0
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    det = {
                        'box': [x1, y1, x2, y2],
                        'confidence': conf,
                        'label': label,
                        'kor_label': label,
                    }
                    detections.append(det)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for det in detections:
                det['risk'] = computeRisk(det, frame_w)
            zones = getZones(detections, frame_w)
            t_post1 = time.perf_counter()
            post_ms = (t_post1 - t_post0) * 1000

            with procLock:
                processedFrames[self.stream_id] = img

            # update annotated publisher
            if ANNOTATED_RTSP_ENABLE:
                pub = publishers.get(self.stream_id)
                if pub is not None:
                    pub.update(img)

            times = {
                'infer_ms': round(infer_ms, 1),
                'post_ms': round(post_ms, 1),
            }
            msg = json.dumps({
                'stream_id': self.stream_id,
                'detections': detections,
                'zones': zones,
                'times': times,
            })
            try:
                asyncio.run(ws_broadcast(msg))
            except RuntimeError:
                # If already running loop in main, schedule via thread-safe method
                loop = asyncio.get_event_loop()
                loop.create_task(ws_broadcast(msg))

            total_ms = (time.perf_counter() - t0) * 1000
            print(f"[Proc {self.stream_id}] infer={times['infer_ms']}ms post={times['post_ms']}ms total={total_ms:.1f}ms det={len(detections)} zones={zones}")
            time.sleep(0.001)

# ----------------------
# WebSocket server (results broadcast)
# ----------------------
async def ws_handler(websocket):
    with ws_lock:
        ws_clients.add(websocket)
    try:
        # Keep alive; no incoming messages required
        async for _ in websocket:
            pass
    finally:
        with ws_lock:
            try:
                ws_clients.remove(websocket)
            except KeyError:
                pass

async def start_ws_server():
    # Disable compression for JPEG payloads
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT, max_size=10*1024*1024, compression=None):
        print(f"WebSocket results on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()

# ----------------------
# Main
# ----------------------

def start_flask():
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)

def main():
    if not RTSP_URLS:
        print("RTSP_URLS 비어있음. 환경변수 RTSP0, RTSP1 ... 로 설정하세요.")
        return

    # Start capture/process threads
    for idx, url in enumerate(RTSP_URLS):
        CaptureThread(idx, url).start()
        ProcessThread(idx).start()

    # Start Flask and WS
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    asyncio.run(start_ws_server())

if __name__ == '__main__':
    main()
