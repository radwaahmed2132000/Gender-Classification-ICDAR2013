import sys
import os
import time
import cv2
import numpy as np
import glob
import imutils
import pickle
from sklearn import svm

#Constants
#_____________________
N_ANGLE_BINS = 40
BIN_SIZE = 360 // N_ANGLE_BINS
LEG_LENGTH = 25
np.seterr(divide = 'ignore')
#_____________________

#Helpler Functions
#_____________________
def remove_shadow(img):
    rgb_planes = cv2.split(img)
    kernel = np.ones((20, 20), np.float32)/400
    result_planes = []
    for plane in rgb_planes:
        f = cv2.filter2D(plane, -1, kernel)
        result_planes.append(f)
    result_planes = np.array(result_planes, np.int32)
    result_planes = (rgb_planes-result_planes)
    result = cv2.merge(result_planes)
    result[result < 0] = 0
    result = result*9
    result = np.array(result, np.uint8)
    return result

def clip_writing_area(img):
    shadow_free_img = remove_shadow(img)
    binary = (shadow_free_img > 90).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=2)
    dilate = cv2.dilate(erosion, kernel, iterations=30)
    coords = cv2.findNonZero(dilate)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    return rect

def get_hinge_feature(image):
	image = cv2.medianBlur(image, 3)
	bw_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	contours, _= cv2.findContours(
				bw_image, cv2.RETR_TREE, 
				cv2.CHAIN_APPROX_NONE
				)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
	
	hist = np.zeros((N_ANGLE_BINS, N_ANGLE_BINS))
	for cnt in contours:
		n_pixels = len(cnt)
		if n_pixels <= LEG_LENGTH:
			continue
		
		points = np.array([point[0] for point in cnt])
		xs, ys = points[:, 0], points[:, 1]
		point_1s = np.array([cnt[(i + LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
		point_2s = np.array([cnt[(i - LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
		x1s, y1s = point_1s[:, 0], point_1s[:, 1]
		x2s, y2s = point_2s[:, 0], point_2s[:, 1]
		
		phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
		phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)
		
		indices = np.where(phi_2s > phi_1s)[0]
		
		for i in indices:
			phi1 = int(phi_1s[i] // BIN_SIZE) % N_ANGLE_BINS
			phi2 = int(phi_2s[i] // BIN_SIZE) % N_ANGLE_BINS
			hist[phi1, phi2] += 1
			
	normalised_hist = hist / np.sum(hist)
	feature_vector = normalised_hist[np.triu_indices_from(normalised_hist, k = 1)]
	
	return feature_vector
#_____________________
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
        img = cv2.imread(filename)                           # 0 for gray-scale
        test_data.append(img)
    except:
        print("corrupted image detected.")

for img in test_data:
    result = -1             # In case the model can't predict.
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    start_time = time.time()#!START
    #Preprocessing
    #________________________
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = imutils.resize(img, width=3000)
    img = img[:int(img.shape[0]*(1-0.1))]
    img = img[:, int(img.shape[1]*0.05):]
    img = img[:, :int(img.shape[1]*(1-0.05))]
    img = img[int(img.shape[0]*0.05):]
    img = clip_writing_area(img)
    #________________________

    #Feature Extraction
    #________________________
    X = get_hinge_feature(img)
    #________________________

    #Prediction
    #________________________
    result = model.predict([X])[0]
    finish_time = time.time()#!FINISH
    #_________________________
    result = 0 if result else 1 #?Because we weren't using that convention form the beginning.
    total_time = finish_time - start_time
    test_results.append(result)
    results_file.write(str(result)+'\n')

    if total_time == 0:
        times_file.write('0.001\n') # Put 1 millisecond as tolerance.
    else:
        times_file.write(str(total_time)+'\n')

results_file.close()
times_file.close()