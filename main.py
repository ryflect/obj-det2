import numpy as np 
import torch 
import torchvision

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
print(torch.cuda.is_available())
