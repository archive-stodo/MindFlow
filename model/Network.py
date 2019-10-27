import numpy as np

from model import Layer
from model.exception.ArrayDimensionError import ArrayDimensionError


class Network:

    def __init__(self, inputs_n):
        self.layers = []
        self.weights = {}
        self.biases = {}
        self.inputs_n = inputs_n
        self.n_layers = len(self.layers)
        self.n_neurons_in_layers = []  # 10, 10, 8, 1

        self.inputs_x = []
        self.outputs_y = []
        self.z = {}
        self.a = {}
        self.model_compiled = False

    # </Model Creation> =================================================================
    # ===================================================================================
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
        if self.layers:  # if list is not empty
            self.layers[0].inputs_n = self.inputs_n

        for layer_n in range(1, len(self.layers)):
            processed_layer: Layer = self.layers[layer_n]
            previous_layer_neurons_n = self.layers[layer_n - 1].neurons_n
            processed_layer.set_inputs_n(previous_layer_neurons_n)

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        self.recalculate_layers()

    # </Layer recalculations> #########################################################

    def initialise_weights_and_biases(self):
        self.recalculate_layers()
        for layer_n in range(self.n_layers):
            neurons_in_layer = self.layers[layer_n].neurons_n
            inputs_to_layer = self.layers[layer_n].inputs_n

            self.weights[layer_n] = np.array(np.random.randn(inputs_to_layer, neurons_in_layer))

        for layer_n in range(self.n_layers):
            self.biases[layer_n] = np.array(np.ones((self.layers[layer_n].neurons_n, 1)))

        self.model_compiled = True

    def compile(self):
        self.initialise_weights_and_biases()
        self.model_compiled = True

    # </Model Creation> =================================================================
    # ===================================================================================

    # <Forward Propagation> -----------------------------------------------------------
    def set_inputs(self, inputs_x):
        if isinstance(inputs_x, np.ndarray):
            self._check_inputs_x_dimensions(inputs_x)
        elif isinstance(inputs_x, list):
            inputs_x = np.array(inputs_x)
            self._check_inputs_x_dimensions(inputs_x)
        else:
            raise TypeError('Input is not a list or numpy array.')

        self.inputs_x = inputs_x

    def _check_inputs_x_dimensions(self, inputs_x):
        dimensions_number = inputs_x.ndim
        if dimensions_number != 2:
            raise ArrayDimensionError("Wrong number of input dimensions:", dimensions_number,
                                      ". Input array should be of dimension: (inputs_n, examples_n)")

        rows = inputs_x.shape[0]
        if rows != self.inputs_n:
            raise ArrayDimensionError("Wrong number of inputs: ", rows,
                                      "Input array should be of dimension: (inputs_n, examples_n)")

    def forward_propagate(self):
        self.z = {0: np.dot(self.weights[0].T, self.inputs_x) + self.biases[0]}
        self.a = {0: self.layers[0].activation_f.value(self.z[0])}
        for layer_n in range(1, self.n_layers):
            self.z[layer_n] = np.dot(self.weights[layer_n].T, self.z[layer_n - 1]) + self.biases[layer_n]
            self.a[layer_n] = self.layers[layer_n].activation_f.value(self.z[layer_n])

        self.outputs_y = self.z[self.n_layers - 1]
        # could potentially set z[layer_n] to corresponding Layer here
