import numpy as np

from model import Layer


class Network:

    def __init__(self, inputs_n, layers):
        self.layers = layers  # tbd - layer objects from input to output
        self.weights = {} # all weights in NN
        self.biases = {}
        self.inputs_n = inputs_n
        self.n_layers = len(self.layers)
        self.n_neurons_in_layers = []  # 10, 10, 8, 1
        # self.input = []  # X
        # self.output = []  # Y

        self.initialise_weights_and_biases()

    # <Layer recalculations> -------------------------------------------------------
    def recalculate_layers(self):
        self._recalculate_n_layers()
        self._recalculate_n_neurons_in_layers()
        self._recalculate_layers_inputs_n()

    def _recalculate_n_neurons_in_layers(self):
        self.n_neurons_in_layers = []
        [self.n_neurons_in_layers.append(layer.neurons_n) for layer in self.layers]

    def _recalculate_n_layers(self):
        self.n_layers = len(self.layers)

    def _recalculate_layers_inputs_n(self):
        if self.layers: # if list is not empty
            self.layers[0].inputs_n = self.inputs_n

        for layer_n in range(1, len(self.layers)):
            processed_layer: Layer = self.layers[layer_n]
            previous_layer_neurons_n = self.layers[layer_n - 1].neurons_n
            processed_layer.set_inputs_n(previous_layer_neurons_n)

    # </Layer recalculations> #########################################################

    def initialise_weights_and_biases(self):
        self.recalculate_layers()
        for layer_n in range(self.n_layers):
            neurons_in_layer = self.layers[layer_n].neurons_n
            inputs_to_layer = self.layers[layer_n].inputs_n

            self.weights[layer_n] = np.random.randn(neurons_in_layer, inputs_to_layer)

        for layer_n in range(self.n_layers):
            self.biases[layer_n] = np.ones(self.layers[layer_n].neurons_n)