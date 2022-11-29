#import tensorflow
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
tensorboard = TensorBoard(log_dir='log{}'.format(time()))