import numpy as np 
import random
import torch 
import torchvision
import detectron2 
import os
import json 
import cv2 
from detectron2.utils.logger import setup_logger
# from train_det2 import conv_catname_to_num, get_dicts
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
def infer_single_image(image_path):
    print("Single Image Inference")
    exit(0)
    im = cv2.imread(image_path)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig("./output/inferred.png")

# save figure that outputs the ground truth as well
# def show_ground_truth(image_path, annotations_path):
    # im = cv2.imread(image_path)
    # DatasetCatalog.register("nusc_train", lambda d=d: get_dicts("./"))
    # MetadataCatalog.get("nusc_train").set(thing_classes=["barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck", "void"])
    # nusc_metadata = MetadataCatalog.get("nusc_train")
    # v = Visualizer(im[:, :, ::-1], metadata=nusc_metadata, scale=0.8, )
    # vis = v.draw_dataset_dict(d)
    # plt.imshow(vis.get_image()[:, :, ::-1])
    # plt.savefig("./output/fig_draw.png")

infer_single_image("./samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657123112404.jpg")
# show_ground_truth("./samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657123112404.jpg")
