import os
import csv

# 폴더 경로
folder_path = 'C:/smu/linear_algebra/data/yongan/yongan_JHS'

# CSV 파일 경로 및 이름
csv_file = 'output.csv'

# 폴더 내 파일 이름 읽기
file_names = os.listdir(folder_path)

# CSV 파일 생성 및 데이터 작성
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
    writer.writeheader()
    
    # 각 파일에 대해 작업
    for file_name in file_names:
        # 파일 이름을 frame 열에 쓰고, xmin과 xmax 값을 채움
        writer.writerow({'frame': file_name, 'xmin': 0, 'xmax': 640, 'ymin': 0, 'ymax': 640, 'class_id': 1})

print("CSV 파일이 생성되었습니다.")