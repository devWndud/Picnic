import websockets
import asyncio
import json
import time
import cv2
import threading
import os
import base64

ip = '222.231.24.5'

# 전역 정지 이벤트 (q로 종료 공유)
stop_event = threading.Event()

class DisplayThread(threading.Thread):
    def __init__(self, window_name, shared):
        super().__init__(daemon=True)
        self.window_name = window_name
        self.shared = shared

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while not stop_event.is_set():
            frame = None
            with self.shared['lock']:
                if self.shared['latest_frame'] is not None:
                    frame = self.shared['latest_frame'].copy()
            if frame is not None:
                cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        cv2.destroyWindow(self.window_name)

class VideoThread(threading.Thread):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.ret = False
        self.frame = None
        self.running = True
        self.loop = True

        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            self.running = False
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        print(f"Video: {video_path}")
        print(f"  FPS: {fps}, Frames: {frame_count}, Duration: {duration:.2f}s")

    def run(self):
        while self.running and not stop_event.is_set():
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            time.sleep(1/30)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

def _findVideoFiles(max_videos=None):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if os.path.exists('videos'):
        for file in os.listdir('videos'):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join('videos', file))
    
    if max_videos and len(video_files) > max_videos:
        print(f"발견된 영상: {len(video_files)}개, 제한: {max_videos}개")
        video_files = video_files[:max_videos]
        print(f"처리할 영상: {video_files}")
    
    return video_files

def _getDirection(idx, total):
    directionMap = ["Front", "Left", "Right"]
    if idx < len(directionMap):
        return directionMap[idx]
    else:
        return f"Extra {idx}"

async def _sendFrames(video, video_idx, total_videos):
    direction = _getDirection(video_idx, total_videos)
    windowName = f'{direction} (Video {video_idx} - {os.path.basename(video.video_path)})'

    # 공유 버퍼 (최신 프레임만 유지)
    shared = {'latest_frame': None, 'lock': threading.Lock()}
    disp = DisplayThread(windowName, shared)
    disp.start()

    uniqueCode = 'A7X2bQ9cLmY1'

    # 적응형 전송률 변수
    target_fps = 15.0
    min_fps, max_fps = 5.0, 30.0
    jpeg_quality = 70
    min_q, max_q = 40, 85

    async with websockets.connect(f'ws://{ip}:8000', max_size=10*1024*1024) as ws:
        while not stop_event.is_set():
            if not video.ret or video.frame is None:
                await asyncio.sleep(0.01)
                continue

            t0 = time.perf_counter()
            frame = video.frame.copy()
            frame = cv2.resize(frame, (640, 360))

            t1 = time.perf_counter()
            ok, img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                continue
            img_b64 = base64.b64encode(img).decode('ascii')
            encode_ms = (time.perf_counter() - t1) * 1000

            payload = {
                "img_b64": img_b64,
                "cam_idx": video_idx,
                "uniqueCode": uniqueCode,
                "client_ts": time.time(),
            }

            t2 = time.perf_counter()
            await ws.send(json.dumps(payload))
            t3 = time.perf_counter()
            result = await ws.recv()
            t4 = time.perf_counter()

            rtt_ms = (t4 - t2) * 1000
            send_ms = (t3 - t2) * 1000
            total_cycle_ms = (t4 - t0) * 1000

            try:
                result_dict = json.loads(result)
            except Exception as e:
                print(f"[Client {video_idx}] JSON parse error: {e}")
                continue

            if "detections" in result_dict:
                for det in result_dict["detections"]:
                    label = det["kor_label"] if "kor_label" in det else det["label"]
                    x1, y1, x2, y2 = det["box"]
                    conf = det["confidence"]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{direction}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            server_times = result_dict.get("times", {})
            recv_to_infer_ms = server_times.get("recv_to_infer_ms")
            infer_ms = server_times.get("infer_ms")
            post_ms = server_times.get("post_ms")
            proc_ms = server_times.get("proc_ms")

            net_ms = None
            if proc_ms is not None:
                try:
                    net_ms = max(0.0, rtt_ms - float(proc_ms))
                except Exception:
                    net_ms = None

            # 오버레이 텍스트
            overlay_lines = [
                f"enc {encode_ms:.1f}ms | send {send_ms:.1f}ms | rtt {rtt_ms:.1f}ms | net~ {net_ms if net_ms is not None else 'NA'} | total {total_cycle_ms:.1f}ms",
                f"srv: recv->infer {recv_to_infer_ms}ms | infer {infer_ms}ms | post {post_ms}ms | proc {proc_ms}ms | fps {target_fps:.1f} q {jpeg_quality}"
            ]
            y = 20
            for line in overlay_lines:
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                y += 18

            with shared['lock']:
                shared['latest_frame'] = frame

            print(f"[Client {video_idx}] enc={encode_ms:.1f}ms send={send_ms:.1f}ms rtt={rtt_ms:.1f}ms net~={net_ms if net_ms is not None else 'NA'} total={total_cycle_ms:.1f}ms | server recv->infer={recv_to_infer_ms}ms infer={infer_ms}ms post={post_ms}ms proc={proc_ms}ms | fps={target_fps:.1f} q={jpeg_quality}")

            # 적응형 조절 (간단한 AIMD 스타일)
            # 목표: rtt <= 300ms, proc <= 200ms 근처 유지
            try:
                too_slow = (rtt_ms is not None and rtt_ms > 300) or (proc_ms is not None and proc_ms > 200)
                very_slow = (rtt_ms is not None and rtt_ms > 600) or (proc_ms is not None and proc_ms > 400)
                if very_slow:
                    target_fps = max(min_fps, target_fps * 0.8)
                    jpeg_quality = max(min_q, jpeg_quality - 5)
                elif too_slow:
                    target_fps = max(min_fps, target_fps - 1.0)
                    jpeg_quality = max(min_q, jpeg_quality - 2)
                else:
                    target_fps = min(max_fps, target_fps + 0.5)
                    jpeg_quality = min(max_q, jpeg_quality + 1)
            except Exception:
                pass

            # 프레임 속도 제한
            target_interval_ms = 1000.0 / max(min_fps, min(max_fps, target_fps))
            sleep_ms = max(0.0, target_interval_ms - total_cycle_ms)
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000.0)

async def main(max_videos=None):
    video_paths = _findVideoFiles(max_videos)

    if not video_paths:
        print("비디오 파일을 찾을 수 없습니다.")
        print("지원되는 형식: .mp4, .avi, .mov, .mkv, .wmv, .flv")
        print("비디오 파일을 현재 디렉토리나 'videos' 폴더에 넣어주세요.")
        return

    print("비디오 파일들 : ", video_paths)

    videos = [VideoThread(video_path) for video_path in video_paths]
    for video in videos:
        if video.running:
            video.start()

    tasks = []
    for idx, video in enumerate(videos):
        if video.running:
            tasks.append(asyncio.create_task(_sendFrames(video, idx, len(videos))))

    if tasks:
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            stop_event.set()
    else:
        print("재생 가능한 비디오 없음")

    for video in videos:
        video.stop()
    for video in videos:
        video.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    max_videos = None
    if len(sys.argv) > 1:
        try:
            max_videos = int(sys.argv[1])
            print(f"영상 개수 제한: {max_videos}개")
        except ValueError:
            print("올바른 숫자를 입력해주세요. 예: python yolo_client.py 3")
            sys.exit(1)
    
    asyncio.run(main(max_videos))
