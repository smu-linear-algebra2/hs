import yaml

yaml_data = {"names":['jeong_hee_sang'], # 클래스 이름
             "nc":1, # 클래스 수
             "path":root_dir, # root 경로
             "train":os.path.join(root_dir, "train.txt"), # train.txt 경로
             "val":os.path.join(root_dir, "valid.txt"), # valid.txt 경로
             "test":os.path.join(root_dir,"test.txt") # test.txt 경로
             }

with open(os.path.join(root_dir, "custom.yaml"), "w") as f:
  yaml.dump(yaml_data, f)