import unittest
import numpy as np
from model.Layer import Layer
from model.Network import Network
from model.activation_function.Sigmoid import Sigmoid
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class TestNetworkBackwardPropagation(unittest.TestCase):

    def test_backward_propagate(self):
        # given
        self.network = Network(inputs_n=4)
        self.network.add_layer(Layer(neurons_n=3, activation_f=Sigmoid))
        self.network.add_layer(Layer(neurons_n=1, activation_f=Sigmoid))
        self.network.compile()

        iris = load_iris()
        X = iris.data

        # Create a variable for the target data
        y = iris.target
        X_train, X_test, y_train, y_test = \
            train_test_split(X[:100], y[:100], test_size=0.2, shuffle=True)

        # when
        # for epoch in range(50):
        for epoch in range(112):
            for i in range(y_train.size):
                inputs_x = X_train[i].T  # 1 set of inputs
                desired_output = y_train[i]
                self.network.set_inputs(np.reshape(inputs_x, (4, 1)))
                self.network.set_desired_outputs(np.reshape(desired_output, (1, 1)))
                self.network.forward_propagate()
                self.network.backward_propagate()

        print('a')

        correct_predictions = 0
        for i in range(y_test.size):
            inputs_x = X_test[i].T  # 1 set of inputs
            desired_output = y_test[i]
            self.network.set_inputs(np.reshape(inputs_x, (4, 1)))
            self.network.set_desired_outputs(np.reshape(desired_output, (1, 1)))
            self.network.forward_propagate()

            predicted = convert_output_to_prediction(self.network.get_actual_outputs())
            if predicted == y_test[i]:
                correct_predictions += 1

            # print("inputs: ", self.network.inputs_x)
            print("output predicted: ", self.network.get_actual_outputs())
            print("predicted: ", predicted)
            print("actual: ", y_test[i], "\n")

        print("correct predictions: ", correct_predictions)

def convert_output_to_prediction(output):
    if output < 0.5:
        return 0
    else:
        return 1