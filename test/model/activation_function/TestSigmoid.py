import unittest
from model.activation_function.Sigmoid import Sigmoid


class TestSigmoid(unittest.TestCase):

    def test_sigmoid_at_zero(self):
        # when
        sigmoid_at_zero = Sigmoid.get_value(0)

        # then
        self.assertEqual(sigmoid_at_zero, 0.5)

    def test_sigmoid_at_one_ten(self):
        # when
        sigmoid_at_ten = Sigmoid.get_value(10)

        # then
        self.assertAlmostEqual(sigmoid_at_ten, 0.9999546)
