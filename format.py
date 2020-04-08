import json
import numpy as np 
import os 
import pandas as pd 
# fff7106ca1f34c7a83e72ffbc0c0b1b5.txt
# fff7106ca1f34c7a83e72ffbc0c0b1b5.png

directory = "/mnt/nfs/scratch1/pmallya/nusc_kitti/val/label_2/"

# with open(directory + "fff7106ca1f34c7a83e72ffbc0c0b1b5.txt", 'r') as f:
data = np.loadtxt(directory + "fff7106ca1f34c7a83e72ffbc0c0b1b5.txt")
print(data)
print(data[0])
print(data[0][4:8])
