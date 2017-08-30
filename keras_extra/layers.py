from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
import keras
import numpy as np

__author__ = 'Sergey Kiyan'
__email__ = 'sokian92@gmail.com'
__version__ = '1.0.0'


class LinearActivation(Layer):
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = float(alpha)
        self.beta = float(beta)
        super(LinearActivation, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return self.alpha * inputs + self.beta
