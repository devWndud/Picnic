import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'Camera {i}', frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"Camera index {i} is not available.")