from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import cv2
import pytesseract
import re
import numpy as np

app = Flask(__name__)
CORS(app, support_credentials=True)
#logging.basicConfig(level=logging.DEBUG,
 #            format='(%(threadName)-10s) %(message)s',
  #                   )

@app.route('/json', methods=["POST"])
def json():
    print('hello')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    def ocr_core(img):
        text = pytesseract.image_to_string(img)
        #text = pytesseract.image_to_string(img, lang='eng', config='--psm 1')
        return text

    img = cv2.imread('main.PNG')

    def get_greyscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(image):
        return cv2.medianBlur(image,5)

    def thresholding(image):
        return cv2.threshold(img,0,255,cv2.THRESH_BINARY)
        #return cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    img = get_greyscale(img)
    #img = remove_noise(img)
    #img = thresholding(img)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #img = cv2.blur(img,(5,5))
    #img = cv2.GaussianBlur(img,(11,11),0)
    #img = cv2.medianBlur(img,3)
    #cv2.imshow("Sliced Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #img = remove_noise(img)

    s = ocr_core(img)
    print(s)
    lines = s.split('\n')
    #print(lines)
    str_list = list(filter(None, lines))
    #str_list.remove(' ')
    #print(str_list)
    seq=str_list[2::3]
    print(seq)

    return jsonify(seq)

app.run(host='127.0.0.1', port=8090)