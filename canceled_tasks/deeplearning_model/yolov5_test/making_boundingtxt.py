from pathlib import Path
import pandas as pd

IMG_WIDTH = 640
IMG_HEIGHT = 480
# 'dataset_path'는 다운로드 한 'archive.zip' 압축 푼 경로
#dataset_path = "../datasets/sample_dataset"
image_file_path = "C:/smu/linear_algebra/data/yongan/yongan_JHS"

df = pd.read_csv("C:/smu/linear_algebra/output.csv")

def box2d_to_yolo(box2d):
    # 0~1 사이 값으로 정규화
    x1 = box2d["xmin"] / IMG_WIDTH
    x2 = box2d["xmax"] / IMG_WIDTH
    y1 = box2d["ymin"] / IMG_HEIGHT
    y2 = box2d["ymax"] / IMG_HEIGHT

    # 모서리 좌표에서 센터 좌표로 변환
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # width, height 구함
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return cx, cy, width, height


assert Path(image_file_path).is_dir(), "Output directory doesn't exist"
labels_dir = Path("C:/smu/linear_algebra/data/yongan/yongan_JHS").absolute()

Path(labels_dir).mkdir(exist_ok=True)


# 이미지 파일('frame' 필드)로 그룹화하여 처리한다.
for frame, v in df.groupby(['frame']):
    img_name = Path(frame)
    assert img_name.suffix == ".jpg"
    frame_name = str(img_name.stem)
    annotation_file = labels_dir / (frame_name + ".txt")
    with open(annotation_file, "w") as anno_fp:  # 어노테이션 파일 생성
        for _, row in v.iterrows():
            cx, cy, width, height = box2d_to_yolo(row)
            class_id = row['class_id'] - 1
            anno_fp.write(f"{class_id} {cx} {cy} {width} {height}\n")

assert len(list(labels_dir.glob('*.txt'))) == len(df.groupby(['frame']))