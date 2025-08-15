import io
import cv2
import numpy as np
from flask import Flask, request, send_file
from ultralytics import YOLO
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

app = Flask(__name__)

# OD 모델
od_model = YOLO('yolov8x.pt')
# SS 모델 (VOC)
ss_model = deeplabv3_resnet50(weights='DEFAULT').eval()

VOC_LABELS = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
    6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
    12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'potted plant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv/monitor'
}
SS_TARGETS = [15, 7]  # person, car
color_map = {
    15: (0, 0, 255),   # person - 빨강
    7: (255, 0, 0),    # car - 파랑
}
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def od_ss_process(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # OD
    od_results = od_model(img_rgb)
    od_img = img.copy()
    for r in od_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = od_model.names[cls]
            if label in ["person", "car", "bicycle", "motorcycle", "bus", "truck"]:
                cv2.rectangle(od_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(od_img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # SS
    input_tensor = preprocess(img_rgb).unsqueeze(0)
    with torch.no_grad():
        output = ss_model(input_tensor)['out'][0]
    seg_mask = output.argmax(0).cpu().numpy()
    seg_mask_resized = cv2.resize(seg_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ss_img = img.copy()
    for target in SS_TARGETS:
        mask = (seg_mask_resized == target)
        ss_img[mask] = (ss_img[mask] * 0.5 + np.array(color_map[target]) * 0.5).astype(np.uint8)
    # OD+SS 합성
    result_img = cv2.addWeighted(od_img, 0.7, ss_img, 0.3, 0)
    return result_img

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    file = request.files['image']
    in_memory = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory, cv2.IMREAD_COLOR)
    result_img = od_ss_process(img)
    _, img_encoded = cv2.imencode('.jpg', result_img)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
