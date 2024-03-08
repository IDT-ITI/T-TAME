from pathlib import Path

import tqdm

dir_path = Path("/ssd/ntrougkas/Documents/iia/data/heatmaps")
missing_vgg = []
missing_res = []
missing_vit = []
with open(Path("/ssd/ntrougkas/Documents/iia/data/pics.txt"), "r") as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        name = line.split(" ")[0].split(".")[0] + "*"
        files = list(dir_path.glob(name))
        files = [str(f) for f in files]
        # if not any("vgg16" in file for file in files):
        #     missing_vgg.append(line)
        # if not any("resnet50" in file for file in files):
        #     missing_res.append(line)
        #     print(line, files)
        if not any("vit" in file for file in files):
            missing_vit.append(line)

for file_list, m in zip(
    [missing_vgg, missing_res, missing_vit], ["vgg16", "resnet50", "vit"]
):
    with open(f"./missing_{m}.txt", "w") as f:
        for el in file_list:
            f.write(el)
