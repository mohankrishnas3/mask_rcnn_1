from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import threading
import logging
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import torchvision.models as models
maskrcnn_resnet50_fpn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# #app = Flask(__name__)
# #CORS(app, support_credentials=True)
# logging.basicConfig(level=logging.DEBUG,
#                     format='(%(threadName)-10s) %(message)s',
#                     )
#
# @app.route('/json', methods=["POST"])
# def json():
#     print(request.json['url'], 'hello')


# from selenium import webdriver
# #from selenium.webdriver import ActionChains
# #from selenium.webdriver.common.by import By
# from webdriver_manager.chrome import ChromeDriverManager
# #import time
# import os
# options = webdriver.ChromeOptions()
# options.headless = True
# driver = webdriver.Chrome(ChromeDriverManager().install())
# driver.implicitly_wait(10)
# driver.get(request.json['url'])
# #driver.get_screenshot_as_file('screen1.png');
# S = lambda X: driver.execute_script('return document.body.parentNode.scroll' + X)
# driver.set_window_size(S('Width'), S('Height'))
# driver.find_element_by_tag_name('body').screenshot('main.png')
#
# import cv2
#
# # Load .png image
# image = cv2.imread('main.png')
#
# # Save .jpg image
# cv2.imwrite('image.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
# img = cv2.imread('image.jpg', 1)
# sliimg = img[11:982, 12:959]
# path = r'C:\Users\Asus\PycharmProjects\tesseract\venv\Include\cig_butts\New Folder\cig_butts/real_testA'
# cv2.imwrite(os.path.join(path , 'waka.jpg'), sliimg)
# cv2.waitKey(0)
import cv2
import os
import sys
import json
import numpy as np
# import time
from PIL import Image, ImageDraw

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = r'C:\Users\Asus\PycharmProjects\tesseract/Mask_RCNN'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class CigButtsConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cig_butts"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 1 (cig_butt)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5

    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = maskrcnn_resnet50_fpn

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000


config = CigButtsConfig()
config.display()


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


dataset_train = CocoLikeDataset()
dataset_train.load_data(
    r'C:\Users\Asus\PycharmProjects\tesseract\venv\Include\cig_butts\New Folder\cig_butts\trainA/coco_instances.json',
    r'C:\Users\Asus\PycharmProjects\tesseract\venv\Include\cig_butts\New Folder\cig_butts\trainA/images')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data(
    r'C:\Users\Asus\PycharmProjects\tesseract\venv\Include\cig_butts\New Folder\cig_butts\valA/coco_instances.json',
    r'C:\Users\Asus\PycharmProjects\tesseract\venv\Include\cig_butts\New Folder\cig_butts\valA/images')
dataset_val.prepare()

dataset = dataset_train
image_ids = np.random.choice(dataset.image_ids, 4)
print(image_ids)
# for image_id in image_ids:
#   image = dataset.load_image(image_id)
#   mask, class_ids = dataset.load_mask(image_id)
#  visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
tensorboard = TensorBoard(log_dir='log{}'.format(time()))


# start_train = time.time()
# model.train(dataset_train, dataset_val,
#               learning_rate=config.LEARNING_RATE,
#                epochs=4,
#              layers='heads')
# #end_train = time.time()
# #minutes = round((end_train - start_train) / 60, 2)
# #print(f'Training took {minutes} minutes')
# #start_train = time.time()
# model.train(dataset_train, dataset_val,
#              learning_rate=config.LEARNING_RATE / 10,
#              epochs=8,
#              layers="all")
# end_train = time.time()
# minutes = round((end_train - start_train) / 60, 2)
# print(f'Training took {minutes} minutes')
#
# epochs = range(epochs[-1])


class InferenceConfig(CigButtsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.80


inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)
model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

import skimage
from skimage import io, color
from skimage.transform import resize

real_test_dir = r'C:\Users\Asus\PycharmProjects\tesseract\venv\Include\cig_butts\New Folder\cig_butts/real_testA'

image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))
import io
from PIL import Image
from PIL import ImageCms

for image_path in image_paths:
    originalImage = cv2.imread(image_path)
    img = skimage.io.imread(image_path)
    if img.shape[-1] == 4:
        image = img[:, :, :3]
    img_arr = np.array(image)

    # with graph.as_default():
    results = model.detect([img_arr], verbose=1)
    K.clear_session()
    r = results[0]
    a = r['rois'][0]
    print(a)
    data = r['scores'][0]
    print(data)
    # #b = r['rois'][1]
    #
    x1 = a[1]
    x2 = a[3]
    y1 = a[0]
    y2 = a[2]

    # a1 = b[1]
    # a2 = b[3]
    # b1 = b[0]
    # b2 = b[2]

visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], figsize=(5, 5))
#
# #   cropped = img[y1:y2, x1:x2]

slicedImage = originalImage[y1:y2, x1:x2]
# #    cv2.imshow("Original Image", originalImage)
cv2.imshow("Sliced Image", slicedImage)
cv2.imwrite('main.PNG', slicedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
# # slicedImage1 = originalImage[b1:b2, a1:a2]
# # #    cv2.imshow("Original Image", originalImage)
# # cv2.imshow("Sliced Image", slicedImage1)
# # cv2.imwrite('main1.PNG', slicedImage1)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#     #io.imshow(cropped)
#     #io.show()
# import cv2
# import pytesseract
# import re
#
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# def ocr_core(img):
#     text = pytesseract.image_to_string(img)
#     #text = pytesseract.image_to_string(img, lang='eng', config='--psm 1')
#     return text
#
# img = cv2.imread('main.PNG')
#
# def get_greyscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# def remove_noise(image):
#     return cv2.medianBlur(image,5)
#
# def thresholding(image):
#     return cv2.threshold(img,0,255,cv2.THRESH_BINARY)
#     #return cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     #return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
#
# img = get_greyscale(img)
# #img = remove_noise(img)
# #img = thresholding(img)
# img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# #img = cv2.blur(img,(5,5))
# #img = cv2.GaussianBlur(img,(11,11),0)
# #img = cv2.medianBlur(img,3)
# cv2.imshow("Sliced Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #img = remove_noise(img)
#
# s = ocr_core(img)
# print(s)


# return jsonify(About_item = s)

# def ocr_core(img):
#     text = pytesseract.image_to_string(img)
#     return text
#
# img = cv2.imread('main1.PNG')
#
# def get_greyscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # def remove_noise(image):
#     # return cv2.medianBlur(image,5)
#
# def thresholding(image):
#     return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# img = get_greyscale(img)
# img = thresholding(img)
#     # img = remove_noise(img)
#
# s1 = ocr_core(img)
# print(s1)
# c = s + s1

# return jsonify(About_item  = s, Table = s1)


# app.run(host='127.0.0.1', port=8090)


