import unittest
from model.Layer import Layer
from model.Network import Network
from model.activation_function.Linear import Linear
from model.activation_function.Sigmoid import Sigmoid

class TestNetworkForwardPropagation(unittest.TestCase):

    def test_forward_propagate_linear_activation(self):
        # given
        inputs_x = [[1], [2]]  # 1 set of inputs
        self.network = Network(inputs_n=2)
        self.network.add_layer(Layer(neurons_n=3, activation_f=Linear))
        self.network.add_layer(Layer(neurons_n=1, activation_f=Linear))
        self.network.compile()
        self.network.set_inputs(inputs_x)

        # set weights to 1 - so calculations are easy to do by hand
        self.network.set_all_weights_to_one()

        # when
        self.network.forward_propagate()

        # then
        #                [layer][example_nr][neuron_nr]
        self.assertEqual(self.network.z[0][0][0], 4.0)
        self.assertEqual(self.network.z[1][0][0], 13.0)
        self.assertEqual(self.network.actual_outputs_a, 13)

    def test_forward_propagate_linear_activation(self):
        # given
        inputs_x = [[1], [2]]  # 1 set of inputs
        self.network = Network(inputs_n=2)
        self.network.add_layer(Layer(neurons_n=3, activation_f=Sigmoid))
        self.network.add_layer(Layer(neurons_n=1, activation_f=Sigmoid))
        self.network.compile()
        self.network.set_inputs(inputs_x)

        # set weights to 1 - so calculations are easy to do by hand
        self.network.set_all_weights_to_one()

        # when
        self.network.forward_propagate()

        # then
        expected_output = Sigmoid.value(3 * Sigmoid.value(4) + 1)
        self.assertEqual(self.network.get_actual_outputs(), expected_output)