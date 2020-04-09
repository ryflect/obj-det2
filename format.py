import json
import numpy as np 
import os 
import pandas as pd 
import csv
# fff7106ca1f34c7a83e72ffbc0c0b1b5.txt
# fff7106ca1f34c7a83e72ffbc0c0b1b5.png

directory = "/mnt/nfs/scratch1/pmallya/nusc_kitti/val/label_2/"
# directory = "./Downloads/"
format_json = []

# with open(directory + "fff7106ca1f34c7a83e72ffbc0c0b1b5.txt", 'r') as f:
    # data = np.loadtxt(directory + "fff7106ca1f34c7a83e72ffbc0c0b1b5.txt")
    # reader = csv.reader(f, delimiter=' ')
    # for row in reader:
        # print(row[0])
        # bbox = list(float(i) for i in row[4:8])
        # print(bbox)

forgotten_labels = []       
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(directory + filename, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            print("Processing File: ", filename)
            ground = {}
            ground["id"] = os.path.splitext(filename)[0]
            ground_0 = []
            ground_1 = []
            ground_2 = []
            for row in reader:
                if row[0] == "Car":
                    bbox = list(float(i) for i in row[4:8])
                    ground_0.append(bbox)
                elif row[0] == "Pedestrian":
                    bbox = list(float(i) for i in row[4:8])
                    ground_1.append(bbox)
                elif row[0] == "Truck":
                    bbox = list(float(i) for i in row[4:8])
                    ground_2.append(bbox)
                else:
                    forgotten_labels.append(row[0])
            ground["Car"] = ground_0
            ground["Pedestrian"] = ground_1
            ground["Truck"] = ground_2
            format_json.append(ground)
    else:
        continue 

print("Finished")
print("Forgotten Labels: ", list(set(forgotten_labels)))

with open("./output/ground.json", 'w', encoding='utf-8') as ground_outfile:
    json.dump(format_json, ground_outfile, ensure_ascii=False, indent=4)