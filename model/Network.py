import numpy as np
from model import Layer
from model.activation_function.ActivationFunction import ActivationFunction
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
        self.desired_outputs_y = np.array([])
        self.z = {}
        self.a = {}

        self.error_term = {}
        self.dw = {}
        self.db = {}

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

            self.weights[layer_n] = np.array(np.random.randn(inputs_to_layer, neurons_in_layer)) / 10

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

    # TBD - checks like for input setting
    def set_desired_outputs(self, desired_outputs):
        self.desired_outputs_y = desired_outputs

    def _check_inputs_x_dimensions(self, inputs_x):
        dimensions_number = inputs_x.ndim
        if dimensions_number != 2:
            raise ArrayDimensionError("Wrong number of input dimensions:", dimensions_number,
                                      ". Input array should be of dimension: (inputs_n, examples_n)")

        rows = inputs_x.shape[0]
        if rows != self.inputs_n:
            raise ArrayDimensionError("Wrong number of inputs: ", rows,
                                      "Input array should be of dimension: (inputs_n, examples_n)")

    def set_all_weights_to_one(self):
        # set weights to 1 - so test calculations are easy to do by hand
        for layer_n in range(self.n_layers):
            neurons_in_layer = self.layers[layer_n].neurons_n
            inputs_to_layer = self.layers[layer_n].inputs_n
            self.weights[layer_n] = np.array(np.ones((inputs_to_layer, neurons_in_layer)))

    def get_actual_outputs(self):
        return self.a[self.n_layers - 1]

    def forward_propagate(self):
        self.a = {-1: self.inputs_x}
        for layer_n in range(0, self.n_layers):
            self.z[layer_n] = np.dot(self.weights[layer_n].T, self.a[layer_n - 1]) \
                              # + self.biases[layer_n]
            self.a[layer_n] = self.layers[layer_n].activation_f.value(self.z[layer_n])

    # not working properly - to be thoroughly checked
    def backward_propagate(self):
        last_layer_n = self.n_layers - 1
        f: ActivationFunction = self.layers[last_layer_n].activation_f
        self.error_term[last_layer_n] = [(self.desired_outputs_y - self.get_actual_outputs()) * f.derivative(self.get_actual_outputs())]

        for layer_n in range(self.n_layers - 2, -1, -1):
            f: ActivationFunction = self.layers[layer_n].activation_f

            self.error_term[layer_n] = np.zeros((self.layers[layer_n].neurons_n, 1))
            for neuron_n in range(self.layers[layer_n].neurons_n):
                for neuron_n_next in range(self.layers[layer_n + 1].neurons_n):
                    self.error_term[layer_n][neuron_n] = self.weights[layer_n + 1][neuron_n_next] * self.error_term[layer_n + 1][neuron_n_next] * f.derivative(self.a[layer_n][neuron_n])

        for layer_n in range(0, self.n_layers):
            for neuron_n in range(self.layers[layer_n].neurons_n):
                self.weights[layer_n][:, neuron_n] = self.weights[layer_n][:, neuron_n] + 0.8 * np.multiply(self.a[layer_n], self.error_term[layer_n])[neuron_n]
                # self.biases[layer_n].T[:, neuron_n] = self.biases[layer_n].T[:, neuron_n] + 0.1 * np.multiply(self.a[layer_n], self.error_term[layer_n])[neuron_n]



