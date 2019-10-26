from model.activation_function.ActivationFunction import ActivationFunction
import numpy as np


class Sigmoid(ActivationFunction):

    @classmethod
    def get_value(cls, input_x):
        return 1 / (1 + np.exp(-input_x))

