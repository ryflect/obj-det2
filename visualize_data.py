import numpy as np 
import random
import torch 
import torchvision
import detectron2 
import os
import json 
import cv2 
from detectron2.utils.logger import setup_logger
from matplotlib import pyplot as plt 

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

# inference for a single image
def infer_single_image(image_name):
    # print("Single Image Inference")
    im = cv2.imread("/mnt/nfs/scratch1/pmallya/nusc_kitti/val/image_2/" + image_name)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig("./output/" + str(image_name) + "_inferred.png")
    return outputs["instances"].to("cpu")

# save figure that outputs the ground truth as well
def show_ground_truth(dict, annotations):
    im = cv2.imread(dict["file_name"])
    v = Visualizer(im[:, :, ::-1], metadata=annotations, scale=0.8, )
    vis = v.draw_dataset_dict(dict)
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.savefig("./output/" + dict["id"] + "_ground.png")

def get_dicts(img_dir, json_path):
    with open(json_path) as f:
        image_anns = json.load(f)

    dataset_dicts = []
    for v in image_anns:
        record = {}
        filename = os.path.join(img_dir, v["id"], ".txt")
        print(filename)
        record["file_name"] = filename
        record["id"] = v["id"]
        objs = []
        for ann in v["Car"]:
            obj = {
                "bbox": ann,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0
            }
            objs.append(obj)
        # for ann in v["Pedestrian"]:
            # obj = {
                # "bbox": ann,
                # "bbox_mode": BoxMode.XYXY_ABS,
                # "category_id": 1
            # }
            # objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

DatasetCatalog.register("nusc_train", lambda d="train": get_dicts("/mnt/nfs/scratch1/pmallya/nusc_kitti/val/image_2/", "./output/ground.json"))
DatasetCatalog.register("nusc_infer", lambda d="infer": get_dicts("/mnt/nfs/scratch1/pmallya/nusc_kitti/val/image_2/", "./output/infer.json"))
# MetadataCatalog.get("nusc_train").set(thing_classes=["barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck", "void"])
MetadataCatalog.get("nusc_train").set(thing_classes=["car", "pedestrian"])
MetadataCatalog.get("nusc_infer").set(thing_classes=["car", "pedestrian"])
nusc_metadata_ground = MetadataCatalog.get("nusc_train")
nusc_metadata_infer = MetadataCatalog.get("nusc_infer")

nusc_dicts_ground = get_dicts("/mnt/nfs/scratch1/pmallya/nusc_kitti/val/image_2/", "./output/ground.json")
nusc_dicts_infer = get_dicts("/mnt/nfs/scratch1/pmallya/nusc_kitti/val/image_2/", "./output/infer.json")
# print(nusc_dicts)
print(len(nusc_dicts_ground))
print(len(nusc_dicts_infer))

# infer_single_image("b139c133286247d48093bba9151920b2.png", 15)
# show_ground_truth("b139c133286247d48093bba9151920b2.png", nusc_dicts_ground, 15)