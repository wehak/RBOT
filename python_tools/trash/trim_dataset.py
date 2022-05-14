from __future__ import annotations
from genericpath import exists
from os import confstr
from matplotlib.font_manager import json_dump
import numpy as np
import cv2 as cv
import json 
from datetime import datetime
from shutil import copyfile

# import imantics
from imantics import Polygons, Mask
from pathlib import Path
from pycocotools import mask


# folderpath = Path("/media/wehak/Data/coco_master/test_1_d-handle")
# check_image = True

config = {
    "database_folder"   : Path("/media/wehak/Data/coco_master/trial_dataset_4/train"), # output
    "min_annots"        : 2,
    "view_time"         : False, # False or ms
}

# delete image function
def remove_img(mask_path, img_path):
    mask_path.unlink()
    img_path.unlink()

""" read existing database """

# if there is an existing labels file
if Path(f"{config['database_folder']}/labels.json").is_file():

    # read the existing labels
    with open(f"{config['database_folder']}/labels.json", "r") as f:
        previous_labels = json.load(f)
        prev_categories = previous_labels["categories"]
        prev_images = previous_labels["images"]
        prev_annotations = previous_labels["annotations"]

id_pairs = [(int(annot['id']), int(annot['image_id'])) for annot in prev_annotations[:]]

annot_ids, img_ids = zip(*id_pairs)

new_annots = []
rejects = []
for id in set(img_ids):
    if img_ids.count(id) < config["min_annots"]:
        rejects.append(id)
    else:
        new_annots.append()


print(f"Found {len(rejects)} images, {round((len(rejects)/len(prev_annotations))*100,2)}% of annotations. Delete? [y/n] ", end="")

if input() == "y":
    for id in rejects:
        Path(f"{config['database_folder']}/data/{id}.png").unlink()
    print("Deleted")
else:
    print("Exited without deleting.")
