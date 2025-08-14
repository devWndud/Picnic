import websockets
import asyncio
import json
import time
import cv2
import threading
import os

ip = '222.231.24.5'

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
        while self.running:
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

    uniqueCode = 'A7X2bQ9cLmY1'

    async with websockets.connect(f'ws://{ip}:8000') as ws:
        while True:
            if not video.ret or video.frame is None:
                await asyncio.sleep(0.01)
                continue

            frame = video.frame.copy()
            frame = cv2.resize(frame, (640, 360))

            _, img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            payload = {
                "img": list(img.tobytes()),
                "cam_idx": video_idx,
                "uniqueCode": uniqueCode
            }

            await ws.send(json.dumps(payload))
            result = await ws.recv()
            result_dict = json.loads(result)

            if "detections" in result_dict:
                for det in result_dict["detections"]:
                    label = det["kor_label"] if "kor_label" in det else det["label"]
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
            pass
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
