import unittest
from model.activation_function.Sigmoid import Sigmoid


class TestSigmoid(unittest.TestCase):

    def test_sigmoid_at_zero(self):
        # when
        sigmoid_at_zero = Sigmoid.value(0)

        # then
        self.assertEqual(sigmoid_at_zero, 0.5)

    def test_sigmoid_derivative_at_zero(self):
        # when
        sigmoid_derivative_at_zero = Sigmoid.derivative(0)

        # then
        self.assertEqual(sigmoid_derivative_at_zero, 0.25)

    def test_sigmoid_at_ten(self):
        # when
        sigmoid_at_ten = Sigmoid.value(10)

        # then
        self.assertAlmostEqual(sigmoid_at_ten, 0.9999546)

    def test_sigmoid_derivative_at_ten(self):
        # when
        sigmoid_derivative_at_ten = Sigmoid.derivative(10)

        # then
        self.assertAlmostEqual(sigmoid_derivative_at_ten, 0.00004539)