from os import mkdir
import os
import shutil


imagenet_eval_path = "imagenet-1k/ILSVRC2012_img_val"
datalist_path = "datalist/ILSVRC/Evaluation_2000.txt"
new_folder_path = "imagenet-1k/test/eval_2000"
mkdir(new_folder_path)
with open(datalist_path, "r") as f:
    for id, line in enumerate(f.readlines()):
        line = line.strip()
        image_name = line.split(" ")[0]
        shutil.copy(
            os.path.join(imagenet_eval_path, image_name),
            os.path.join(new_folder_path, f"{id}.jpg"),
        )
