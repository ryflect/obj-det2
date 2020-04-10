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

directory = "/mnt/nfs/scratch1/pmallya/nusc_kitti/val/image_2/"
infer_directory = "/mnt/nfs/scratch1/pmallya/nusc_kitti/val/infer_2/"
output_json = []
draw_output_flag = False
id_num = 0
count = 0
plt.rcParams['figure.figsize'] = [20, 10]
# generate ids for val set
val_file = open("/mnt/nfs/scratch1/pmallya/nusc_kitti/val/nusc_val.txt", 'w')
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        print("Processing File: ", filename)
        infer = {}
        infer["id"] = os.path.splitext(filename)[0]
        val_file.write(infer["id"] + "\n")
        infer_0 = []
        infer_1 = []
        infer_2 = []
        im = cv2.imread(directory + filename)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        inferred_output = outputs["instances"].to("cpu")
        
        if draw_output_flag:
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.imshow(v.get_image()[:, :, ::-1])
            plt.savefig("./output/inferred_" + str(id_num) + ".png")
            id_num = id_num + 1
            
        infer_classes = inferred_output.pred_classes.numpy()
        infer_bbox = inferred_output.pred_boxes
        box_count = 0
        with open(infer_directory + infer["id"] + ".txt", 'w') as infer_file:
            for i in infer_bbox:
                # print(i)
                # print(infer_classes[box_count])
                if infer_classes[box_count] == 2:
                    infer_0.append(i.numpy().tolist())
                    w_string = "Car 0.0 0.0 0.0 " + " ".join([str(k) for k in i.numpy().tolist()]) + " 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
                    print(w_string)
                    infer_file.write(w_string)
                elif infer_classes[box_count] == 0:
                    infer_1.append(i.numpy().tolist())
                    w_string = "Pedestrian 0.0 0.0 0.0 " + " ".join([str(k) for k in i.numpy().tolist()]) + " 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
                    print(w_string)
                    infer_file.write(w_string)
                box_count = box_count + 1
        
        # print("Boxes: ")
        # print(inferred_output.pred_boxes)
        # print("Classes: ")
        # print(inferred_output.pred_classes)        
        
        infer["Car"] = infer_0
        infer["Pedestrian"] = infer_1
        output_json.append(infer)
        count = count + 1
        # if count == 10:
            # break

val_file.close()
print("Finished")
with open("./output/infer.json", 'w', encoding='utf-8') as infer_outfile:
    json.dump(output_json, infer_outfile, ensure_ascii=False, indent=4)
        