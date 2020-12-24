import numpy

from network.algorithm import Algorithm


class Tahn(Algorithm):
    def calculate(self, value: float) -> float:
        return numpy.tanh(value)

    def calculate_partial_prime(self, value: float) -> float:
        return 1.0 - value ** 2
