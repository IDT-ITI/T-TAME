import torchvision.datasets.imagenet as utils
from imagenet_classes import imagenet_classes

id = 561883 + 1
folder_name = ""
with open("datalist/ILSVRC/vgg16_train.txt", "r") as f:
    for i, line in enumerate(f):
        if i == id:
            folder_name = line.split(" ")[0].split("/")[0]
            print(line)

metadata = utils.load_meta_file("datalist/ILSVRC")
inverse_imagenet_classes = {v: k for k, v in imagenet_classes.items()}
print(inverse_imagenet_classes[", ".join(metadata[0][folder_name])])
