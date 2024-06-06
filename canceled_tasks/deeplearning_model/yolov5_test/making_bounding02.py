import os
import cv2

# 크롭된 얼굴 이미지가 저장된 디렉토리
image_dir = 'C:/smu/linear_algebra/data/yongan/yongan_JHS'

# 라벨 파일을 저장할 디렉토리
label_dir = 'C:/smu/linear_algebra/data/yongan'
os.makedirs(label_dir, exist_ok=True)

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 경계 상자 좌표 (전체 이미지를 경계 상자로 설정)
    x_center = width / 2
    y_center = height / 2
    bbox_width = width
    bbox_height = height

    # YOLO 형식으로 저장 (class_id x_center y_center width height)
    label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        f.write(f"0 {x_center / width} {y_center / height} {bbox_width / width} {bbox_height / height}\n")
