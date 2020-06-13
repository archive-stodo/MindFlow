from model.activation_function.ActivationFunction import ActivationFunction


class Linear(ActivationFunction):

    @classmethod
    def value(cls, input_x):
        return input_x

    @classmethod
    def derivative(cls, input_x):
        return 1