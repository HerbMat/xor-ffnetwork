import numpy

from network.algorithm import Algorithm


class Relu(Algorithm):
    def calculate(self, value: float) -> float:
        return numpy.maximum(0, value)

    def calculate_partial_prime(self, value: float) -> float:
        if numpy.greater(value, 0):
            return 1
        return 0
