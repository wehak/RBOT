from __future__ import annotations
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
    "database_folder"   : Path("/media/wehak/WD Passport/dataset_test_1_1/total"), # output
    "data_fraction"     : 1.0, # percent
    "validation_split"  : .20, # percent
}

# delete image function
def remove_img(mask_path, img_path):
    mask_path.unlink()
    img_path.unlink()

def make_split(ids, split):
    print(f"Making '{split}'")
    new_imgs = []
    new_annots = []

    split_path = Path(config['database_folder']).parent / split / "data"
    split_path.mkdir(parents=True, exist_ok=True)
    
    for id in tqdm(ids):
        img, annots = search_id(id, "train", split_path)

        new_imgs.append(img)
        new_annots += annots

    # write labels.json file
    labels = {
        "info": {
            "year": "2022",
            "version": "1.0",
            "description": f"ROV handle database {split.upper()}",
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
        "categories": prev_categories,
        "images": new_imgs,
        "annotations": new_annots
    }

    with open(split_path.parent / "labels.json", "w") as f:
        json.dump(labels, f)


def search_id(img_id, split, split_path):
    annots = []
    for img in prev_images:
        if int(img["id"]) == img_id:
            old_path = Path(f"{config['database_folder']}/data/{img['file_name']}")
            new_path = Path(split_path) / Path(f"{img['file_name']}")
            copyfile(old_path, new_path)

            new_image = img
            break
    
    for annot in prev_annotations:
        if int(annot["image_id"]) == img_id:
            annots.append(annot)
    
    return new_image, annots


""" read existing database """
if not Path(f"{config['database_folder']}/labels.json").is_file():
    print("error no file")
    exit()

# read the existing labels
with open(f"{config['database_folder']}/labels.json", "r") as f:
    previous_labels = json.load(f)
    prev_categories = previous_labels["categories"]
    prev_images = previous_labels["images"]
    prev_annotations = previous_labels["annotations"]


img_ids = set([int(img["id"]) for img in prev_images])
img_ids = random.sample(img_ids, round(len(img_ids) * config["data_fraction"]))

assert len(img_ids) == len(prev_images)

train_ids = set(random.sample(img_ids, round(len(prev_images) * (1-config["validation_split"]))))
val_ids = img_ids.difference(train_ids)

assert len(img_ids) == len(train_ids) + len(val_ids)

make_split(train_ids, "train")
make_split(val_ids, "val")


