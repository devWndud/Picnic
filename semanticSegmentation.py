import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO

# Object Detection 모델 (YOLO)
od_model = YOLO('yolov8x.pt')  # 사전학습된 모델 사용

# Semantic Segmentation 모델 (DeepLabV3)
ss_model = deeplabv3_resnet50(pretrained=True).eval()

# Semantic Segmentation 클래스 (COCO stuff 기준)
SS_LABELS = {
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
    8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'street sign', 13: 'stop sign',
    14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella',
    29: 'shoe', 30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'plate', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase',
    87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush', 91: 'hair brush', 92: 'banner',
    93: 'blanket', 94: 'branch', 95: 'bridge', 96: 'building', 97: 'bush', 98: 'cabinet', 99: 'cage',
    100: 'cardboard', 101: 'carpet', 102: 'ceiling', 103: 'clothes', 104: 'clouds', 105: 'counter', 106: 'curtain',
    107: 'desk', 108: 'dirt', 109: 'door', 110: 'fence', 111: 'floor', 112: 'floor mat', 113: 'flower',
    114: 'fog', 115: 'food', 116: 'fruit', 117: 'furniture', 118: 'grass', 119: 'gravel', 120: 'ground',
    121: 'hill', 122: 'house', 123: 'leaves', 124: 'light', 125: 'mat', 126: 'metal', 127: 'mirror',
    128: 'moss', 129: 'mountain', 130: 'mud', 131: 'napkin', 132: 'net', 133: 'paper', 134: 'pavement',
    135: 'pillow', 136: 'plant', 137: 'plastic', 138: 'platform', 139: 'playingfield', 140: 'railing',
    141: 'railroad', 142: 'river', 143: 'road', 144: 'rock', 145: 'roof', 146: 'rug', 147: 'salad',
    148: 'sand', 149: 'sea', 150: 'shelf', 151: 'sidewalk', 152: 'sky', 153: 'skyscraper', 154: 'snow',
    155: 'sofa', 156: 'staircase', 157: 'streetlight', 158: 'table', 159: 'tent', 160: 'text', 161: 'towel',
    162: 'tree', 163: 'truck', 164: 'wall', 165: 'water', 166: 'window', 167: 'wood'
}

# 관심있는 SS 클래스 (예: 땅, 계단)
SS_TARGETS = [120, 156]  # ground, staircase

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_od_ss(image_path):
    # 이미지 로드
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Object Detection (사람, 자동차 등)
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

    # Semantic Segmentation (땅, 계단 등)
    input_tensor = preprocess(img_rgb).unsqueeze(0)
    with torch.no_grad():
        output = ss_model(input_tensor)['out'][0]
    seg_mask = output.argmax(0).cpu().numpy()
    # seg_mask를 원본 이미지 크기로 리사이즈
    seg_mask_resized = cv2.resize(seg_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Segmentation 결과 시각화
    ss_img = img.copy()
    color_map = {
        120: (0, 0, 255),   # ground - 빨강
        156: (255, 0, 0),   # staircase - 파랑
    }
    for target in SS_TARGETS:
        mask = (seg_mask_resized == target)
        ss_img[mask] = cv2.addWeighted(ss_img[mask], 0.5, np.array(color_map[target]), 0.5, 0)

    # OD + SS 결과 합성
    result_img = cv2.addWeighted(od_img, 0.7, ss_img, 0.3, 0)

    # 결과 저장 및 반환
    cv2.imwrite('od_result.jpg', od_img)
    cv2.imwrite('ss_result.jpg', ss_img)
    cv2.imwrite('od_ss_result.jpg', result_img)
    print('OD, SS, OD+SS 결과가 od_result.jpg, ss_result.jpg, od_ss_result.jpg로 저장되었습니다.')
    return od_img, ss_img, result_img

if __name__ == "__main__":
    run_od_ss('./001.png')
