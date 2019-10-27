import unittest

import numpy as np

from model.Layer import Layer
from model.Network import Network
from model.activation_function import Sigmoid
from model.exception.ArrayDimensionError import ArrayDimensionError


class TestNetworkForwardPropagation(unittest.TestCase):

    def setUp(self) -> None:
        self.network = Network(inputs_n=3)
        self.network.add_layer(Layer(neurons_n=2, activation_f=Sigmoid))
        self.network.add_layer(Layer(neurons_n=1, activation_f=Sigmoid))
        self.network.compile()

    def test_setting_inputs_from_list(self):
        # given
        inputs_x = [[1], [2], [3]] # 1 set of inputs

        # when
        self.network.set_inputs(inputs_x)

        # then
        self.assertEqual(self.network.inputs_x.shape, (3, 1))

    def test_setting_inputs_from_np_array(self):
        # given
        inputs_x = np.array([[1, 2, 3]]).T

        # when
        self.network.set_inputs(inputs_x)

        # then
        self.assertEqual(self.network.inputs_x.shape, (3, 1))

    def test_setting_inputs_from_np_array_ArrayDimensionError_1_dimensional_array_passed(self):
        # given
        inputs_x = np.array([1, 2, 3]).T

        # when
        self.assertRaises(ArrayDimensionError, lambda: self.network.set_inputs(inputs_x))

    def test_setting_inputs_from_np_array_ArrayDimensionError_wrong_number_of_inputs(self):
        # given
        inputs_x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]]).T

        # when
        self.assertRaises(ArrayDimensionError, lambda: self.network.set_inputs(inputs_x))

    def test_setting_inputs_from_int_TypeError(self):
        # given
        inputs_x = 10

        # when
        self.assertRaises(TypeError, lambda: self.network.set_inputs(inputs_x))