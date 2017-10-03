import json
import os
import random
import numpy as np
from PIL import Image



SOURCE_DIR = "../target-generator/generated/"
IMG_SOURCE_DIR = SOURCE_DIR + "images/"
LABEL_SOURCE_DIR = SOURCE_DIR + "labels/"
ANNOTATIONS_FILE = SOURCE_DIR + "annotations.json"

DATA_OUTPUT_DIR = "data/"
if not os.path.exists(DATA_OUTPUT_DIR):
	os.mkdir(DATA_OUTPUT_DIR)
TRAIN_IMAGES_SAVEPATH = DATA_OUTPUT_DIR + "train_images.npy"
TRAIN_LABELS_SAVEPATH = DATA_OUTPUT_DIR + "train_labels.npy"
VALIDATION_IMAGES_SAVEPATH = DATA_OUTPUT_DIR + "validation_images.npy"
VALIDATION_LABELS_SAVEPATH = DATA_OUTPUT_DIR + "validation_labels.npy"
TEST_IMAGES_SAVEPATH = DATA_OUTPUT_DIR + "test_images.npy"
TEST_LABELS_SAVEPATH = DATA_OUTPUT_DIR + "test_labels.npy"

IMG_PIXEL_WIDTH = 30
IMG_PIXEL_HEIGHT = 30
IMG_CHANNELS = 3

TRAIN_PERCENTAGE = 60
VALIDATE_PERCENTAGE = 20
TEST_PERCENTAGE = 20


annotations = json.load(open(ANNOTATIONS_FILE, 'r'))
shape_class_dict = annotations["Shape"]
color_class_dict = annotations["Background Color"]


img_filenames = os.listdir(IMG_SOURCE_DIR)
num_images = len(img_filenames)
print("Found", num_images, "images.")
images = np.zeros(shape = (num_images,IMG_PIXEL_WIDTH,IMG_PIXEL_HEIGHT,IMG_CHANNELS), dtype=np.uint8)
labels = np.zeros(shape = (num_images,len(shape_class_dict) * len(color_class_dict)))

for i in range(num_images):

	print(i)
	index = random.randint(0, num_images-i-1)
	img_filename = img_filenames.pop(index)
	label_filename = img_filename.split('.')[0] + ".json"
	img_filepath = IMG_SOURCE_DIR + img_filename
	label_filepath = LABEL_SOURCE_DIR + label_filename
	
	img = np.asarray(Image.open(img_filepath))
	images[i, :, :, :] = img
	
	img_metadata = json.load(open(label_filepath, 'r'))
	shape = img_metadata['shape']
	color = img_metadata['shape_color']
	label_onehot_index = shape_class_dict[shape] * len(color_class_dict) + color_class_dict[color]
	labels[i, label_onehot_index] = 1

num_train_images = int(num_images * TRAIN_PERCENTAGE / 100)
num_validatation_images = int(num_images * VALIDATE_PERCENTAGE / 100)
num_test_images = num_images - num_train_images - num_validatation_images

print("Number of train images:", num_train_images)
print("Number of validation images:", num_validatation_images)
print("Number of test images:", num_test_images)

train_images = images[0:num_train_images]
train_labels = labels[0:num_train_images]
validation_images = images[num_train_images:num_validatation_images+num_train_images]
validation_labels = labels[num_train_images:num_validatation_images+num_train_images]
test_images = images[num_validatation_images+num_train_images:]
test_labels = labels[num_validatation_images+num_train_images:]

np.save(TRAIN_IMAGES_SAVEPATH, train_images)
np.save(TRAIN_LABELS_SAVEPATH, train_labels)
np.save(VALIDATION_IMAGES_SAVEPATH, validation_images)
np.save(VALIDATION_LABELS_SAVEPATH, validation_labels)
np.save(TEST_IMAGES_SAVEPATH, test_images)
np.save(TEST_LABELS_SAVEPATH, test_labels)