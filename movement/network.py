import numpy as np
import config

class NeuralNetwork:
    def __init__(self, weights=None):
        self.w1_shape = (config.INPUT_SIZE, config.HIDDEN_SIZE)
        self.b1_shape = (config.HIDDEN_SIZE,)
        self.w2_shape = (config.HIDDEN_SIZE, config.OUTPUT_SIZE)
        self.b2_shape = (config.OUTPUT_SIZE,)
        
        self.param_count = (config.INPUT_SIZE * config.HIDDEN_SIZE + config.HIDDEN_SIZE + 
                            config.HIDDEN_SIZE * config.OUTPUT_SIZE + config.OUTPUT_SIZE)
        
        if weights is None:
            self.weights = np.random.uniform(-1.0, 1.0, self.param_count)
        else:
            self.weights = np.array(weights)
            
        self.w1, self.b1, self.w2, self.b2 = self._deserialize(self.weights)

    def _deserialize(self, flat_weights):
        idx = 0
        w1 = flat_weights[idx:idx + self.w1_shape[0] * self.w1_shape[1]].reshape(self.w1_shape)
        idx += self.w1_shape[0] * self.w1_shape[1]
        b1 = flat_weights[idx:idx + self.b1_shape[0]]
        idx += self.b1_shape[0]
        w2 = flat_weights[idx:idx + self.w2_shape[0] * self.w2_shape[1]].reshape(self.w2_shape)
        idx += self.w2_shape[0] * self.w2_shape[1]
        b2 = flat_weights[idx:idx + self.b2_shape[0]]
        return w1, b1, w2, b2

    def forward(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        out = np.tanh(np.dot(h, self.w2) + self.b2)
        return out
