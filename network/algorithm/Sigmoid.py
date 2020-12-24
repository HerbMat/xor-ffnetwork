import numpy

from network.algorithm import Algorithm


class Sigmoid(Algorithm):
    def calculate(self, value: float) -> float:
        return 1.0 / (1.0 + numpy.exp(-value))

    def calculate_partial_prime(self, value: float) -> float:
        return value * (1 - value)
