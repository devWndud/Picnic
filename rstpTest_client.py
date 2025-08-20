import asyncio
import websockets
import json
import cv2
import time
import imageio.v3 as iio
import numpy as np
import subprocess
import sys

SERVER_IP = "222.231.24.5"
VIDEO_PATH = "101.mp4"
MODE = "webp"  # 'webp' or 'h264'

async def send_frames_webp(ws, cap, send_times):
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_bytes = iio.imwrite("<bytes>", frame, plugin="pillow", format="webp", quality=70)
        payload = {
            "img": list(img_bytes),
            "cam_idx": 0,
            "uniqueCode": "test",
            "frame_id": frame_id,
            "encoding": "webp"
        }
        send_times[frame_id] = time.time()
        await ws.send(json.dumps(payload))
        frame_id += 1
        await asyncio.sleep(1/10)  # 10fps for network safety
    await ws.send(json.dumps({"end": True}))

async def send_frames_h264():
    # Use ffmpeg to stream H.264 video to the server's UDP port
    ffmpeg_cmd = [
        "ffmpeg", "-re", "-i", VIDEO_PATH, "-vcodec", "libx264", "-f", "mpegts",
        f"udp://{SERVER_IP}:5000"
    ]
    print("[CLIENT] Starting ffmpeg for H.264 streaming:", " ".join(ffmpeg_cmd))
    proc = await asyncio.create_subprocess_exec(*ffmpeg_cmd)
    await proc.wait()

async def receive_results(ws, send_times):
    while True:
        try:
            result = await ws.recv()
            recv_time = time.time()
            result_dict = json.loads(result)
            frame_id = result_dict.get("frame_id")
            if frame_id is not None and frame_id in send_times:
                rtt = (recv_time - send_times.pop(frame_id)) * 1000
                print(f"[CLIENT DEBUG] Frame {frame_id} round-trip time: {rtt:.2f} ms")
            print(f"[Client] Detected: {result_dict.get('detections')}")
        except Exception as e:
            print(f"[CLIENT ERROR] {e}")
            break

async def main():
    if MODE == "webp":
        cap = cv2.VideoCapture(VIDEO_PATH)
        async with websockets.connect(f'ws://{SERVER_IP}:5000/ws') as ws:
            send_times = {}
            send_task = asyncio.create_task(send_frames_webp(ws, cap, send_times))
            recv_task = asyncio.create_task(receive_results(ws, send_times))
            await send_task
            await asyncio.sleep(1)
            recv_task.cancel()
        cap.release()
    elif MODE == "h264":
        await send_frames_h264()
    else:
        print("Unknown MODE. Use 'webp' or 'h264'.")

if __name__ == "__main__":
    asyncio.run(main())
