import numpy as np 
import random
import torch 
import torchvision
import detectron2 
from detectron2.utils.logger import setup_logger


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# check for cuda availability
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
print(torch.cuda.is_available())

print("Hello")