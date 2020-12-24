import random
from typing import Iterable, Tuple, List

import numpy

from network import Algorithm

TrainInput = List[float]
TrainOutput = List[float]
TrainData = List[Tuple[TrainInput, TrainOutput]]


class NeuralNetwork(object):
    BIAS_VALUE = 1

    def __init__(self, layers_desc: List[int], algorithm: Algorithm, bias=True):
        self.layers = []
        self.algorithm = algorithm
        self.number_of_inputs = layers_desc[0]
        self.momentum_weights = []
        self.bias = bias
        number_of_weights = layers_desc[0]
        if self.bias:
            number_of_weights += 1
        for neuronCount in layers_desc[1:]:
            self.layers.append(numpy.random.rand(neuronCount, number_of_weights))
            self.momentum_weights.append(numpy.zeros(self.layers[-1].shape))
            number_of_weights = neuronCount
            if self.bias:
                number_of_weights += 1

    def train(self, train_data: TrainData, learning_rate=0.2, epochs=100000, momentum=0.3) -> List[float]:
        errors = []
        train_data_length = len(train_data)
        for iteration in range(epochs):
            train_data_pos = iteration % train_data_length
            if train_data_pos == 0:
                random.shuffle(train_data)
            single_train_data = train_data[train_data_pos]
            errors.append(self.train_single_epoch(single_train_data[0], single_train_data[1], learning_rate, momentum))
        return errors

    def train_single_epoch(self, inputs: TrainInput, outputs: TrainOutput, learning_rate: float,
                           momentum: float) -> float:
        layer_outputs = [inputs.copy()]
        if self.bias:
            layer_outputs[-1].append(self.BIAS_VALUE)
        for layer in self.layers:
            layer_outputs.append(self.calculate_layer(layer, layer_outputs[-1]))
        if self.bias:
            layer_outputs[-1].pop()
        iteration_error = self.calculate_error(outputs, layer_outputs[-1])
        layer_pos = len(self.layers) - 1
        layer_errors = [self.calculate_output_error_total(target, error) for target, error in
                        zip(outputs, layer_outputs[-1])]
        new_layers = []
        for layer_pos in range(layer_pos, -1, -1):
            new_layers.append(self.fit_layer(
                layer_outputs[layer_pos + 1], layer_outputs[layer_pos], layer_errors,
                learning_rate, momentum, layer_pos))
            previous_layer_error_derivative = [layer_error * self.algorithm.calculate_partial_prime(layer_output) for
                                               layer_output, layer_error in
                                               zip(layer_outputs[layer_pos + 1], layer_errors)]
            layer_errors = [self.calculate_layer_error(self.layers[layer_pos], previous_layer_error_derivative, pos) for
                            pos, neuron in
                            enumerate(self.layers[layer_pos - 1])]
        new_layers.reverse()
        self.layers = new_layers
        return float(numpy.sum(iteration_error).astype(float))

    def calculate_layer_error(self, neurons: numpy.ndarray, old_layer_errors: List[float], pos: int) -> float:
        layer_errors_per_neuron = [self.calculate_layer_errors(neuron, layer_error, pos) for neuron, layer_error in
                                   zip(neurons, old_layer_errors)]
        return sum(layer_errors_per_neuron)

    @staticmethod
    def calculate_layer_errors(neuron: numpy.ndarray, layer_error: float, pos: int) -> float:
        return neuron[pos] * layer_error

    @staticmethod
    def calculate_error(expected_outputs: List[float], result: List[float]) -> List[float]:
        return [0.5 * (target - output) ** 2 for target, output in zip(expected_outputs, result)]

    def delta(self, error_total: float, output: float, neuron_output: float):
        return error_total * self.algorithm.calculate_partial_prime(output) * neuron_output

    @staticmethod
    def calculate_output_error_total(target: float, output: float):
        return -(target - output)

    def fit_layer(self, outputs: List[float],
                  neuron_outputs: List[float], layer_errors: List[float], learning_rate: float,
                  momentum: float, layer_pos: int) -> numpy.ndarray:
        result = []
        new_momentums_for_layer = []
        for neuron, output, layer_error, momentum_weights_for_neuron in zip(self.layers[layer_pos], outputs,
                                                                            layer_errors,
                                                                            self.momentum_weights[layer_pos]):
            adjustment_result = self.fix_weights(neuron, output, neuron_outputs, layer_error, learning_rate,
                                                 momentum_weights_for_neuron,
                                                 momentum)
            result.append(adjustment_result[0])
            new_momentums_for_layer.append(adjustment_result[1])
        self.momentum_weights[layer_pos] = numpy.stack(new_momentums_for_layer).reshape(
            self.momentum_weights[layer_pos].shape)
        return numpy.stack(result).reshape(self.layers[layer_pos].shape)

    def fix_weights(self, neuron: numpy.ndarray, output: float,
                    neuron_outputs: List[float], layer_error: float, learning_rate: float,
                    momentum_weights_for_neuron: numpy.ndarray, momentum: float):
        error_deltas = [-self.delta(layer_error, output, neuron_output) * learning_rate for neuron_output in
                        neuron_outputs]
        new_neuron = numpy.sum([numpy.sum([neuron, error_deltas], axis=0), momentum_weights_for_neuron * momentum],
                               axis=0)
        return new_neuron, numpy.array(error_deltas).reshape(momentum_weights_for_neuron.shape)

    def predict(self, inputs: List[float]) -> List[float]:
        layer_result = inputs
        if self.bias:
            layer_result.append(self.BIAS_VALUE)
        for i in self.layers:
            layer_result = self.calculate_layer(i, layer_result)
        if self.bias:
            layer_result.pop()
        return layer_result

    def calculate_layer(self, layer: Iterable[Iterable[float]], inputs: List[float]) -> List[float]:
        layer_result = []
        for neuron in layer:
            layer_result.append(self.calculate_neuron(neuron, inputs))
        if self.bias:
            layer_result.append(self.BIAS_VALUE)
        return layer_result

    def calculate_neuron(self, weights: Iterable[float], inputs: Iterable[float]) -> float:
        return self.algorithm.calculate(float(numpy.dot(weights, inputs)))
