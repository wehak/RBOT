import fiftyone as fo

dataset = fo.Dataset.from_dir(
    dataset_dir="/media/wehak/Data/coco_master/trial_dataset_1", 
    dataset_type=fo.types.COCODetectionDataset, 
    name="DuckyDB"
)

session = fo.launch_app(dataset)
session.wait()