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
def infer_single_image(image_dict, id_num):
    # print("Single Image Inference")
    im = cv2.imread(image_dict["file_name"])
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # plt.imshow(v.get_image()[:, :, ::-1])
    # plt.savefig("./output/inferred_" + str(id_num) + ".png")
    return outputs["instances"].to("cpu")

# save figure that outputs the ground truth as well
def show_ground_truth(dict, annotations, id_num):
    im = cv2.imread(dict["file_name"])
    v = Visualizer(im[:, :, ::-1], metadata=annotations, scale=0.8, )
    vis = v.draw_dataset_dict(dict)
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.savefig("./output/ground_" + str(id_num) + ".png")

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

def conv_catname_to_num_nusc(cat):
    if cat == "vehicle.car":
        return 0
    elif cat == "human.pedestrian.adult" or cat == "human.pedestrian.police_officer" or cat == "human.pedestrian.child" or cat == "human.pedestrian.construction_worker":
        return 1
    else:
        return 2

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
                    "category_id": conv_catname_to_num_nusc(ann["category_name"])
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

DatasetCatalog.register("nusc_train", lambda d="train": get_dicts("./"))
# MetadataCatalog.get("nusc_train").set(thing_classes=["barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck", "void"])
MetadataCatalog.get("nusc_train").set(thing_classes=["car", "pedestrian", "void"])
nusc_metadata = MetadataCatalog.get("nusc_train")
nusc_dicts = get_dicts("./")
# print(len(nusc_dicts))

# do inference for multiple randomly selected images from the miniset
# count = 13
plt.rcParams['figure.figsize'] = [20, 10]
random.seed(2)
for d in random.sample(nusc_dicts, 1):
    print("Ground: ")
    print("File Name: ", d['file_name'])
    print("Annotations: ")
    print(d['annotations'])
    print("---------------------------------------------------------------------------------------------------")
    inferred_output = infer_single_image(d, 1)
    print("Inferred: ")
    print("Boxes: ")
    print(inferred_output.pred_boxes)
    print("Classes: ")
    print(inferred_output.pred_classes)
    # show_ground_truth(d, nusc_metadata, 1)
    # count = count + 1
