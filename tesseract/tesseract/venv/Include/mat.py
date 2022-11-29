from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import threading
import logging
import cv2
import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import tensorflow as tf
from keras import backend as K

app = Flask(__name__)
CORS(app, support_credentials=True)
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

@app.route('/json', methods=["POST"])

def json():
    from selenium import webdriver
    # from selenium.webdriver import ActionChains
    # from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    # import time
    import os
    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(10)
    driver.get(request.json['url'])
    # driver.get_screenshot_as_file('screen1.png');
    S = lambda X: driver.execute_script('return document.body.parentNode.scroll' + X)
    driver.set_window_size(S('Width'), S('Height'))
    driver.find_element_by_tag_name('body').screenshot('main.png')

    import cv2

    # Load .png image
    image = cv2.imread('main.png')

    # Save .jpg image
    cv2.imwrite('image.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    img = cv2.imread('image.jpg', 1)
    sliimg = img[11:982, 12:959]
    path = r'C:\Users\Asus\PycharmProjects\tesseract\venv\Include\cig_butts\New Folder\cig_butts/real_testA'
    cv2.imwrite(os.path.join(path, 'waka.jpg'), sliimg)
    cv2.waitKey(0)
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
        BACKBONE = 'resnet50'

        # To be honest, I haven't taken the time to figure out what these do
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
        TRAIN_ROIS_PER_IMAGE = 32
        MAX_GT_INSTANCES = 50
        POST_NMS_ROIS_INFERENCE = 500
        POST_NMS_ROIS_TRAINING = 1000

    config = CigButtsConfig()
    config.display()

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

    #Load trained weights (fill in path to trained weights here)
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

    for image_path in image_paths:
        originalImage = cv2.imread(image_path)
        img = skimage.io.imread(image_path)
        img_arr = np.array(img)
                # with graph.as_default():
        results = model.detect([img_arr], verbose=1)
        K.clear_session()
        r = results[0]
        a = r['rois'][0]
        b = r['rois'][1]

        x1 = a[1]
        x2 = a[3]
        y1 = a[0]
        y2 = a[2]

        a1 = b[1]
        a2 = b[3]
        b1 = b[0]
        b2 = b[2]

        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                     r['scores'], figsize=(5, 5))

        #    cropped = img[y1:y2, x1:x2]
        slicedImage = originalImage[y1:y2, x1:x2]
        #    cv2.imshow("Original Image", originalImage)
        cv2.imshow("Sliced Image", slicedImage)
        cv2.imwrite('main.PNG', slicedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        slicedImage1 = originalImage[b1:b2, a1:a2]
        #    cv2.imshow("Original Image", originalImage)
        cv2.imshow("Sliced Image", slicedImage1)
        cv2.imwrite('main1.PNG', slicedImage1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # io.imshow(cropped)
        # io.show()
        import cv2
        import pytesseract
        import re

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        def ocr_core(img):
            text = pytesseract.image_to_string(img)
            return text

        img = cv2.imread('main.PNG')

        def get_greyscale(image):
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # def remove_noise(image):
        # return cv2.medianBlur(image,5)

        def thresholding(image):
            return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        img = get_greyscale(img)
        img = thresholding(img)
        # img = remove_noise(img)

        s = ocr_core(img)
        print(s)


    return jsonify(About_item=s, Table="table")



app.run(host='127.0.0.1', port=8090)


