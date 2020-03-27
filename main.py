import numpy as np 
import random
import torch 
import torchvision
import detectron2 
import os
import json 
import cv2 
from detectron2.utils.logger import setup_logger


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

# check for cuda availability
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
print(torch.cuda.is_available())

def conv_catname_to_num(cat):
    if cat == "movable_object.barrier":
        return 0
    elif cat == "vehicle.bicycle":
        return 1
    elif cat == "vehicle.bus.rigid" or cat == "vehicle.bus.bendy":
        return 2
    elif cat == "vehicle.car":
        return 3
    elif cat == "vehicle.construction":
        return 4
    elif cat == "vehicle.motorcycle":
        return 5
    elif cat == "human.pedestrian.adult" or cat == "human.pedestrian.police_officer" or cat == "human.pedestrian.child" or cat == "human.pedestrian.construction_worker":
        return 6
    elif cat == "movable_object.trafficcone":
        return 7
    elif cat == "vehicle.trailer":
        return 8
    elif cat == "vehicle.truck":
        return 9
    else:
        return 10

def get_dicts(img_dir):
    json_file = "./v1.0-mini/image_annotations.json"
    with open(json_file) as f:
        image_anns = json.load(f)

    dataset_dicts = []
    proc_files = []
    for v in image_anns:
        if v["sample_data_token"] not in proc_files:
            proc_files.append(v["sample_data_token"])
            # print(v["sample_data_token"])
        else:
            continue
            
        record = {}
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = v["sample_data_token"]
        record["height"] = height
        record["width"] = width

        objs = []
        for ann in image_anns:
            if ann["sample_data_token"] == record["image_id"]:
                obj = {
                    "bbox": ann["bbox_corners"],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": conv_catname_to_num(ann["category_name"])
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train"]:
    DatasetCatalog.register("nusc_" + d, lambda d=d: get_dicts("./"))
    MetadataCatalog.get("nusc_" + d).set(thing_classes=["barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck", "void"])
nusc_metadata = MetadataCatalog.get("nusc_train")

dataset_dicts = get_dicts("./")
print(len(dataset_dicts))

print("Beginning Training:")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("nusc_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()