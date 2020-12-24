from network import NeuralNetwork, Sigmoid, Tahn, Relu
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_data = [
        ([1, 1], [0]),
        ([1, 0], [1]),
        ([0, 1], [1]),
        ([0, 0], [0])
    ]
    network = NeuralNetwork([2, 3, 1], Tahn())
    plt.plot(network.train(train_data))
    plt.ylabel('Error')
    plt.savefig('xor-tahn.png')
    # plt.show()
    print(network.predict([1, 1]))
    print(network.predict([0, 1]))
    print(network.predict([1, 0]))
    print(network.predict([0, 0]))
