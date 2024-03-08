import os
import shutil
from imagenet_classes import imagenet_classes


# imagenet_train_path = "imagenet-1k/ILSVRC2012_img_train"
# datalist_path = "datalist/ILSVRC/vgg16_train.txt"
# class_id = 805
# new_folder_path = f"imagenet-1k/train/vgg/{class_id}/"
# os.makedirs(new_folder_path, exist_ok=True)
# with open(datalist_path, "r") as f:
#     for img_id, line in enumerate(f.readlines()):
#         line = line.strip()
#         image_name, id = line.split(" ")
#         if int(id) == class_id:
#             shutil.copy(os.path.join(imagenet_train_path, image_name), os.path.join(new_folder_path, f"{img_id}.jpg"))

imagenet_train_path = "imagenet-1k/ILSVRC2012_img_val"
datalist_path = "datalist/ILSVRC/val.txt"
inverse_imagenet_classes = {v: k for k, v in imagenet_classes.items()}
class_id = inverse_imagenet_classes["Siamese cat, Siamese"]
new_folder_path = f"imagenet-1k/test/{class_id}/"
os.makedirs(new_folder_path, exist_ok=True)
with open(datalist_path, "r") as f:
    for img_id, line in enumerate(f.readlines()):
        line = line.strip()
        image_name, id = line.split(" ")
        if int(id) == class_id:
            shutil.copy(
                os.path.join(imagenet_train_path, image_name),
                os.path.join(new_folder_path, f"{img_id}.jpg"),
            )
