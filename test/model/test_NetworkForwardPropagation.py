import unittest
import numpy as np
from model.Layer import Layer
from model.Network import Network
from model.activation_function.Sigmoid import Sigmoid

class TestNetworkForwardPropagation(unittest.TestCase):

    def test_forward_propagate(self):
        # given
        inputs_x = [[1], [2]]  # 1 set of inputs
        self.network = Network(inputs_n=2)
        self.network.add_layer(Layer(neurons_n=3, activation_f=Sigmoid))
        self.network.add_layer(Layer(neurons_n=1, activation_f=Sigmoid))
        self.network.compile()
        self.network.set_inputs(inputs_x)

        # set weights to 1 - so calculations are easy to do by hand
        for layer_n in range(self.network.n_layers):
            neurons_in_layer = self.network.layers[layer_n].neurons_n
            inputs_to_layer = self.network.layers[layer_n].inputs_n
            self.network.weights[layer_n] = np.array(np.ones((inputs_to_layer, neurons_in_layer)))

        # when
        self.network.forward_propagate()

        # then
        #                [layer][example_nr][neuron_nr]
        self.assertEqual(self.network.z[0][0][0], 4.0)
        self.assertEqual(self.network.z[1][0][0], 13.0)
        self.assertEqual(self.network.z[1][0][0], self.network.outputs_y)
        self.assertEqual(np.round(self.network.a[0][0][0], decimals=7), 0.9820138)
        self.assertEqual(np.round(self.network.a[1][0][0], decimals=7), 0.9999977)
