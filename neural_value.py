from rl import ValueFunction
import numpy as np
import cgt
from cgt import nn
import numpy.random as nr

class NeuralValueFunction(ValueFunction):

    def __init__(self, num_features=None, num_hidden=100):
        stepsize = 0.01
        # with shape (batchsize, ncols)
        X = cgt.matrix("X", fixed_shape=(1, num_features))
        # y: a symbolic variable representing the rewards, which are integers
        y = cgt.scalar("y", dtype='float64')
        
        hid1 = nn.rectify(
            nn.Affine(num_features, num_hidden, weight_init=nn.IIDGaussian(std=.1), bias_init=nn.Constant(1))(X)
        )
        # One final fully-connected layer, and then a linear activation output for reward
        output = nn.Affine(num_hidden, 1, weight_init=nn.IIDGaussian(std=.1), bias_init=nn.Constant(1))(hid1)
        abs_deviation = cgt.abs(output - y).mean()
        params = nn.get_parameters(abs_deviation)
        gparams = cgt.grad(abs_deviation, params)
        
        updates = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
        self.predictor = cgt.function([X], output)
        self.updater = cgt.function([X, y], abs_deviation, updates=updates)

    def _features(self, path):
        raise NotImplementedError

    def fit(self, paths):
        # (pathlength * num_paths) x features
        featmat = np.concatenate([self._features(path) for path in paths])
        # (pathlength * num_paths) x 1
        returns = np.concatenate([path["returns"] for path in paths])
        n = len(featmat)
        for _ in range(1):
            order = np.arange(n)
            nr.shuffle(order)
            for idx in order:
                self.updater( featmat[idx, :].reshape((1,-1)), returns[idx] )    

    def predict(self, path):
        featmat = self._features(path)
        n = len(featmat)
        prediction = np.empty(n) 
        for i in range(n):
            prediction[i] = self.predictor( featmat[i, :].reshape((1,-1)) )
        return prediction
