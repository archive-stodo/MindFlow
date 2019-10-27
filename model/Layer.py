from model.activation_function import ActivationFunction


class Layer:

    def __init__(self, neurons_n: int, activation_f: ActivationFunction):
        self.neurons_n = neurons_n
        self.activation_f = activation_f
        # self.z = [] # incoming signal
        self.inputs_n = 0

    def set_inputs_n(self, inputs_n):
        self.inputs_n = inputs_n

