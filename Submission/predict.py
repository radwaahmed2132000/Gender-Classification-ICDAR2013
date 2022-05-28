import sys                                          
import os
import time
import cv2
import numpy as np
import glob
#include any models or files here

if len(sys.argv) != 3: print('usage: predict.py <full path to the test set directory> <full path to the output directory>'); sys.exit(1)

# Paths should be in a similar format to: /home/usr/someone/somewhere/test/ i.e. /*/ or /*/*/ etc.
assert os.path.exists(sys.argv[1]), "Cannot find the test directory, "+str(sys.argv[1])
assert os.path.exists(sys.argv[2]), "Cannot find the output directory, "+str(sys.argv[2])

print("Hooray we found your directories!")

TEST_SET_PATH = sys.argv[1]
OUTPUT_PATH   = sys.argv[2]

results_file = open(OUTPUT_PATH + "results.txt", 'w')
times_file = open(OUTPUT_PATH + "times.txt", 'w')

print(TEST_SET_PATH)

test_data, test_results = [], []
#load test images
for filename in sorted(glob.glob(TEST_SET_PATH+'*.jpg')):
    try:
        img = cv2.imread(filename, 0)                           # 0 for gray-scale
        test_data.append(img)
    except:
        print("corrupted image detected.")

for img in test_data:

    result = -1             # In case the model can't predict.
    start_time = time.time()
    # Call our chosen model on the img here.
    # result = model(img)
    finish_time = time.time()
    total_time = finish_time - start_time

    test_results.append(result)
    results_file.write(str(result)+'\n')

    if total_time == 0:
        times_file.write('0.001\n') # Put 1 millisecond as tolerance.
    else:
        times_file.write(str(total_time)+'\n')

results_file.close()
times_file.close()