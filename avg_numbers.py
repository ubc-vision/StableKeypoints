#!/usr/bin/env python3

import sys
from glob import glob

perf_five = []
perf_ten = []

# for file in glob("/home/iamerich/burst/pfwillow_no_crop_ablation/*/pck_array_*.txt"):
for file in glob("/home/iamerich/burst/pfwillow_no_noise_ablation/*/pck_array_*.txt"):
    index = int(file.split("/")[-2])
    # if index < 250:
    #     continue
    # open file and read the 2 lines
    with open(file, 'r') as f:
        lines = f.readlines()
        perf_five.append(float(lines[0].strip()))
        perf_ten.append(float(lines[1].strip()))

print(len(perf_five))
print("Average performance for 5% PCK: ", sum(perf_five)/len(perf_five))
print("Average performance for 10% PCK: ", sum(perf_ten)/len(perf_ten))