from __future__ import annotations
from enum import EnumMeta
from genericpath import exists
from os import confstr
from matplotlib.font_manager import json_dump
import numpy as np
import cv2 as cv
import json 
from datetime import datetime
from shutil import copyfile
import random
from tqdm import tqdm

# import imantics
from imantics import Polygons, Mask
from pathlib import Path
from pycocotools import mask


# folderpath = Path("/media/wehak/Data/coco_master/test_1_d-handle")
# check_image = True

config = {
    "database_folders"  : [Path("/media/wehak/WD Passport/dataset_test_1_1/total"), Path("/media/wehak/WD Passport/dataset_test_2_1/total")],
    "output_folder"     : Path("/media/wehak/WD Passport/dataset_unified/total"), # output
    # "data_fraction"     : 1, # percent of total data to use
    # "validation_split"  : .20, # percent
    # "test_split"        : .10 # percent
}

# delete image function
def remove_img(mask_path, img_path):
    mask_path.unlink()
    img_path.unlink()


new_cats = []
new_imgs = []
new_annots = []
Path(f"{config['output_folder']}/data").mkdir(parents=True, exist_ok=True)


for i, path in enumerate(config["database_folders"], 1):
    print(path.parent)

    # read the existing labels
    with open(f"{path}/labels.json", "r") as f:
        
        previous_labels = json.load(f)
        prev_categories = previous_labels["categories"]
        prev_images = previous_labels["images"]
        prev_annotations = previous_labels["annotations"]
        n = len(prev_images)

        old_new = {}

        for j, img in enumerate(prev_images[:]):
            print(j, "/", n, end="\r")
            new_img = img.copy()
            new_img["id"] = int((i * 10**6) + j)
            old_new[ img["id"] ] = new_img["id"]

            old_path = path / "data" / img["file_name"]
            filename = f"{new_img['id']}{img['file_name'][ img['file_name'].rfind('.'): ]}"
            new_path = config["output_folder"] / "data" / filename
            new_img["file_name"] = filename
            copyfile(old_path, new_path)
            new_imgs.append(new_img)

        
        for j, annot in enumerate(prev_annotations):
            new_annot = annot
            new_annot["id"] = int((i * 10**3) + j)
            new_annot["image_id"] = old_new[int(annot["image_id"])]
            new_annots.append(new_annot)
        
        for cat in prev_categories:
            if cat["name"] not in [ncat["name"] for ncat in new_cats]:
                new_cats.append(cat)
        print()
    

# write labels.json file
labels = {
    "info": {
        "year": "2022",
        "version": "1.0",
        "description": "ROV handle database",
        "contributor": "Håkon Weydahl",
        "url": "",
        # "date_created": "2021-01-19T09:48:27"
        "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    },
    "licenses": [
        {
        "url": "https://opensource.org/licenses/GPL-3.0",
        "id": 1,
        "name": "GNU General Public License version 3"
        },
    ],
    "categories": new_cats,
    "images": new_imgs,
    "annotations": new_annots
}

with open(config['output_folder'] / "labels.json", "w") as f:
    json.dump(labels, f)

print("done")