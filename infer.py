import numpy as np 
import random
import torch 
import torchvision
import detectron2 
import os
import json 
import cv2 
from detectron2.utils.logger import setup_logger
from train_det2 import get_cfg, get_dicts
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

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("nusc_train", )
predictor = DefaultPredictor(cfg)

for d in ["train"]:
    DatasetCatalog.register("nusc_" + d, lambda d=d: get_dicts("./"))
    MetadataCatalog.get("nusc_" + d).set(thing_classes=["barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck", "void"])
nusc_metadata = MetadataCatalog.get("nusc_train")

dataset_dicts = get_dicts("./")
idx = 0
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=nusc_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig("./output/" + idx + ".png")
