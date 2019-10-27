import unittest

from model.Layer import Layer
from model.Network import Network
from model.activation_function import Sigmoid


class TestNetworkModelCreation(unittest.TestCase):

    def test_network_sizes_when_add_layer_method_used_without_calling_compile(self):
        # given
        network = Network(inputs_n=5)
        network.add_layer(Layer(3, Sigmoid))
        network.add_layer(Layer(1, Sigmoid))

        expected_n_layers = 2
        expected_n_neurons_in_layers = [3, 1]
        expected_first_layer_inputs_n = 5
        expected_second_layer_inputs_n = 3

        # then
        self.assertEqual(network.n_layers, expected_n_layers)
        self.assertEqual(network.n_neurons_in_layers, expected_n_neurons_in_layers)
        self.assertEqual(network.layers[0].inputs_n, expected_first_layer_inputs_n)
        self.assertEqual(network.layers[1].inputs_n, expected_second_layer_inputs_n)

        # because compile method not called
        self.assertRaises(KeyError, lambda: network.weights[0])
        self.assertRaises(KeyError, lambda: network.weights[1])
        self.assertRaises(KeyError, lambda: network.biases[0])
        self.assertRaises(KeyError, lambda: network.biases[0])

    def test_network_sizes_when_add_layer_method_used_and_compile_method_called(self):
        # given
        network = Network(inputs_n=5)
        network.add_layer(Layer(3, Sigmoid))
        network.add_layer(Layer(1, Sigmoid))
        network.compile()

        # then
        expected_n_layers = 2
        expected_n_neurons_in_layers = [3, 1]
        expected_first_layer_biases_n = 3
        expected_second_layer_biases_n = 1
        expected_first_layer_weights_n = 5
        expected_second_layer_weights_n = 3
        expected_first_layer_inputs_n = 5
        expected_second_layer_inputs_n = 3

        self.assertEqual(network.n_layers, expected_n_layers)
        self.assertEqual(network.n_neurons_in_layers, expected_n_neurons_in_layers)
        self.assertEqual(len(network.weights[0]), expected_first_layer_weights_n)
        self.assertEqual(len(network.weights[1]), expected_second_layer_weights_n)
        self.assertEqual(len(network.biases[0]), expected_first_layer_biases_n)
        self.assertEqual(len(network.biases[1]), expected_second_layer_biases_n)
        self.assertEqual(network.layers[0].inputs_n, expected_first_layer_inputs_n)
        self.assertEqual(network.layers[1].inputs_n, expected_second_layer_inputs_n)