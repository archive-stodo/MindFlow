class ActivationFunction:
    @classmethod
    def value(cls, input_x):
        raise NotImplementedError

    @classmethod
    def derivative(cls, input_x):
        raise NotImplementedError
