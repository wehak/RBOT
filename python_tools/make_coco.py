from __future__ import annotations
from genericpath import exists
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
    "input_files"       : Path("/media/wehak/Data/coco_master/test_1_d-handle"),
    "database_folder"   : Path("/media/wehak/Data/coco_master/trial_dataset_1"),
    "object_category"   : "d-handle",
    "verify_image"      : True,
    "show_image"        : True,
}

# misc
if not config['input_files'].is_dir():
    print("Wrong path?")
    exit()

config["database_folder"].mkdir(exist_ok=True, parents=True)
img_folder = Path(f"{config['database_folder']}/data")
img_folder.mkdir(exist_ok=True, parents=True)

color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
n_removed = 0
new_annotations = []
new_images = []

# delete image function
def remove_img():
    mask_path.unlink()
    img_path.unlink()

# unpack contour
def unpack_c(contour):
    new_list = []
    for row in contour:
        new_list.append(float(row[0][0]))
        new_list.append(float(row[0][1]))
    return new_list

    # return [[row[0][0], row[0][1]] for row in contour]

# read existing database
""" stuff here """
last_annot_id = 0
last_img_id = 0

prev_categories = [
    {
        "id": 2,
        "name": "d-handle",
        "supercategory": None
    },
    ]
prev_images = []
prev_annotations = []

# is category new?
new_categories = []


# find images
images = list(Path(f"{config['input_files']}/mask").glob("*.png"))
#
# DEBUG
images = images[:25]
#
n_images = len(images)
print(f"Found {n_images} images.")
for i, mask_path in enumerate(images):
    img_path = Path(f"{config['input_files']}/img/{mask_path.stem}.png")

    if config['verify_image']:
        if not mask_path.is_file():
            continue

    img = cv.imread(str(img_path))
    mask_img = cv.imread(str(mask_path))


    # verify if mask is visible
    if not mask_img.any(): 
        print("No mask:", mask_path.stem, end="\r")
        remove_img()
        n_removed += 1
        continue

    # convert to single channel
    mask_gray = cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY)
    ret, threshold = cv.threshold(mask_gray, 127, 255, 0)
    height, width = mask_gray.shape


    # verify mask is not on the image border
    if config['verify_image']:
        if mask_gray[0, :].any():
            print("White along top:", mask_path.stem, end="\r")
            remove_img()
            n_removed += 1
            continue
        elif mask_gray[height-1, :].any():
            print("White along bottom:", mask_path.stem, end="\r")
            remove_img()
            n_removed += 1
            continue
        elif mask_gray[:, 0].any():
            print("White along left:", mask_path.stem, end="\r")
            remove_img()
            n_removed += 1
            continue
        elif mask_gray[:, width-1].any():
            print("White along right:", mask_path.stem, end="\r")
            remove_img()
            n_removed += 1
            continue

    # find segmentation contour
    contours, hierarchy = cv.findContours(
        threshold, 
        cv.RETR_TREE, 
        cv.CHAIN_APPROX_SIMPLE
        )
    
    # control that the largest contour is selected
    areas = []
    for j, cnt in enumerate(contours):
        areas.append(cv.contourArea(cnt))
    max_index = areas.index(max(areas))

    # # find bounding box
    x, y, w, h = cv.boundingRect(contours[max_index])

    # find segmentation
    siluette_mask = np.zeros([height, width], np.uint8)
    cv.drawContours(siluette_mask, [contours[max_index]], -1, 255, cv.FILLED)
    encoded_obj = mask.encode(np.asfortranarray(siluette_mask))
    gt_area = mask.area(encoded_obj)
    gt_bb = mask.toBbox(encoded_obj)

    # print(encoded_obj.keys())

    # print(gt_area.tolist(), areas[max_index])
    # print(gt_bb.tolist(), [x, y, w, h])
    # cv.imshow("test", siluette_mask)
    # cv.waitKey()
    polygons = Mask(siluette_mask).polygons()
    # # print(polygons.points)
    # # print(polygons.segmentation)
    # print([x, y, x+w, y+h])
    # print(polygons.bbox)


    print(f"Image {i+1}/{n_images}", end="\r")

    # draw contour
    if config['show_image']:
        cv.drawContours(img, [contours[max_index]], -1, color[0], 2)
        cv.rectangle(img, [x, y], [x+w, y+h], color[1], 1)
        cv.imshow("name", img)
        cv.waitKey(10)

    # write annotation
    new_annotations.append({
            "segmentation": {
                "counts": encoded_obj["counts"].decode("utf-8"), 
                "size": encoded_obj["size"]
                },
            # "segmentation": [unpack_c(c) for c in contours],
            # "segmentation": polygons.segmentation,
            # "area": areas[max_index], 
            "area": gt_area.tolist(),
            "iscrowd": 0,
            "image_id": mask_path.stem,
            "bbox": [x, y, w, h],
            "category_id": 2,
            "id": last_annot_id + i
    })

    new_images.append({
            "id": mask_path.stem,
            "license": 1,
            "file_name": mask_path.name,
            "height": height,
            "width": width,
            "date_captured": None
    })

    #copy image
    copy_path = Path(f"{img_folder}/{img_path.name}")
    if not copy_path.is_file():
        copyfile(img_path, copy_path)


    

# write status
print()
if config["verify_image"]:
    print(f"Removed {n_removed} images.")


# append new and old data
all_categories = prev_categories + new_categories
all_images = prev_images + new_images
all_annotations = prev_annotations + new_annotations

# print(all_annotations)
# print(all_categories)
# print(all_images)


# write labels.json file
labels = {
    "info": {
        "year": "2022",
        "version": "1.0",
        "description": "DuckyDB",
        "contributor": "Håkon Weydahl",
        "url": "",
        # "date_created": "2021-01-19T09:48:27"
        "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    },
    "licenses": [
        {
          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
          "id": 1,
          "name": "Attribution-NonCommercial-ShareAlike License"
        },
    ],
    "categories": all_categories,
    "images": all_images,
    "annotations": all_annotations
}

with open(f"{config['database_folder']}/labels.json", "w") as f:
    json.dump(labels, f)