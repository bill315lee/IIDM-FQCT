from sklearn import metrics
import numpy as np
import torch
import torch.nn.functional as F
from explain_method.model_acekl import explanation
from copy import deepcopy
import pdb
import math

def weighted_mse_loss(input, target, weights):
    out = (torch.squeeze(input)-target)**2
    # pdb.set_trace()
    loss = torch.mean(out * weights) # or sum over whatever dimensions
    return loss

def softplus(x): # Smooth Relu
    return np.log(1 + np.exp(x))

class AceRegression(torch.nn.Module):

    def __init__(self, input_dim, is_usekl=False, lmd = 1.0):
        super(AceRegression, self).__init__()
        # Variables
        self.lmd = lmd
        self.input_dim = input_dim
        self.is_usekl = is_usekl

        self.weight = torch.nn.Parameter(torch.zeros(1, input_dim))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_X):
        """
        linear part
        """
        linear_part = F.linear(input_X, self.weight, self.bias)

        return linear_part

    def cal_loss(self, input_X, input_y, sample_weights):

        linear_part = self.forward(input_X)
        # pdb.set_trace()
        mse = weighted_mse_loss(linear_part, input_y, sample_weights)

        regularzizer = self.lmd/len(input_y) * torch.sum(self.weight**2)

        # pdb.set_trace()
        if self.is_usekl:
            product = input_X * self.weight.expand_as(input_X)

            prob_mat = F.softplus(product)/torch.sum(F.softplus(product), 1).view(-1, 1).expand_as(product)
            kl = -50.0/self.input_dim * torch.mean(sample_weights * torch.sum(torch.log(prob_mat), 1))
            self.loss = mse+ regularzizer + (- kl)
        else:
            self.loss = mse+ regularzizer# + (- kl)

        return self.loss

    def predict(self, test_x):
        # print test_data.shape
        prediction_value = self.forward(test_x)

        return prediction_value.cpu().data.numpy()


    def get_paras(self):

        return (np.squeeze(self.weight.cpu().data.numpy()), np.squeeze(self.bias.cpu().data.numpy()))

class AceExplainer(object):

    def __init__(self,
                examined_data,
                input_dim,
                epochs,
                is_usekl=False,
                is_lime=False,
                feature_names=None,
                kernel_width=None,
                verbose=False):
        """
        Init function.
        """
        self.is_lime = is_lime
        self.is_usekl = is_usekl

        if kernel_width is None:
            kernel_width = np.sqrt(examined_data.shape[0]) * .75
        self.kernel_width = float(kernel_width)

        self.kernel_fn = lambda d: np.sqrt(np.exp(-(d ** 2) / self.kernel_width ** 2))

        self.feature_names = feature_names
        self.input_dim = input_dim
        self.rng = np.random.RandomState(12345)
        self.easy_model = AceRegression(self.input_dim, is_usekl)
        self.epochs = epochs


    def fit(self, train_X, train_y, train_sample_weights):

        assert len(train_X) == len(train_y)
        optimizer = torch.optim.SGD(self.easy_model.parameters(), lr=0.01)
        diff, prev_train_loss = np.inf, np.inf
        epoch = 0

        if torch.cuda.is_available():
            self.easy_model.cuda()

        # pdb.set_trace()
        while diff > 1e-4:

            """
            re-ordering
            """
            idxs = self.rng.permutation(len(train_y))
            train_X, train_y, train_sample_weights = train_X[idxs], train_y[idxs], train_sample_weights[idxs]


            # train_sample_weights = np.linspace(1.0, 0.6, 5000).astype(np.float32)
            X, y, sample_weights = torch.autograd.Variable(torch.from_numpy(train_X)), torch.autograd.Variable(torch.from_numpy(train_y)), torch.autograd.Variable(torch.from_numpy(train_sample_weights))

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                sample_weights = sample_weights.cuda()

            """
            parameter update
            """
            optimizer.zero_grad()
            train_loss = self.easy_model.cal_loss(X.float(), y, sample_weights)

            # print('loss {}'.format(train_loss.data[0]))
            train_loss.backward()
            optimizer.step()

            diff = np.abs(prev_train_loss - train_loss.cpu().data.numpy())
            prev_train_loss = train_loss.cpu().data.numpy()
            # print "diff:", diff
            '''
            if epoch % 1000 == 0:
                print 'epoch, loss, kl = {}: {}, {}'.format(epoch, train_loss, kl)
                # print "debug: {}".format(debugme)
                # print self.w.eval(), self.b.eval()
            '''
            epoch += 1

            if epoch == self.epochs:
                break

    def get_explanation(self,
                        neighborhood_X,
                        neighborhood_y,
                        distances,
                        verbose = True):
        """
        Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_X: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_y: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
        """

        sample_weights = self.kernel_fn(distances)

        original_row_data = neighborhood_X[0, :]

        self.fit(neighborhood_X[:, :], neighborhood_y, sample_weights)

        w, b = self.easy_model.get_paras()

        if verbose:
            self.easy_model.eval()
            x = torch.autograd.Variable(torch.from_numpy(original_row_data)).float()
            if torch.cuda.is_available():
                x = x.cuda()
            # local_pred = self.easy_model.predict(x)
            # print('Intercept', np.squeeze(b))
            # print('Prediction_local', np.squeeze(local_pred))
            # print('Truth:', np.squeeze(neighborhood_y[0]))

        if self.is_lime:
            contributions = softplus(w)/np.sum(softplus(w))
        else:
            # print w, b
            components = w*original_row_data
            # print components
            contributions = softplus(components)/np.sum(softplus(components))
            # print contributions

        return (b, sorted(zip(range(self.input_dim), contributions),
                        key=lambda x: x[1],
                        reverse=True))


    def explain_instance(self,
                        original_datum,
                        data_row_neighbours_X,
                        data_row_neighbours_y,
                        distance_metric='euclidean',
                        model_regressor=None):
        """
        Generates explanations for a prediction.
        """
        # sample the neibourhood of the data point

        # data_row_neighbours_X[0] is the original row data
        # pdb.set_trace()
        distances = metrics.pairwise_distances(data_row_neighbours_X,
                                                original_datum.reshape(1, -1),
                                                metric=distance_metric
                                                ).ravel()

        # print "inverse", inverse.shape
        # print "yss", yss.shape
        feature_names = deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row_neighbours_X.shape[1])]

        # values of each feature
        values = ['%.2f' % a for a in original_datum]

        ret_exp = explanation.Explanation(feature_names, values, original_datum)
        # print "yss", yss
        ret_exp.predict_score = data_row_neighbours_y[0]
        # print(labels)

        # print distances.shape
        # print yss.shape
        ret_exp.intercept, ret_exp.local_exp = self.get_explanation(
                                                        data_row_neighbours_X,
                                                        data_row_neighbours_y,
                                                        distances, verbose=False)
        return ret_exp


"""
to test
"""

def main():
    input_dim = 20
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(300, 20))
    w = np.linspace(-10, 10, 20)
    b = 30
    y = np.dot(X, w) + b + rng.normal(size = (300))
    sample_weights = np.ones(300)

    neighborhood_X = rng.normal(size=(40, 20)).astype(np.float32)
    neighborhood_y = rng.normal(size=(40)).astype(np.float32)
    # print scaled_examed_example.shape
    myacekl_explaniner = AceExplainer(neighborhood_X[0], 20)
    myexp = myacekl_explaniner.explain_instance(neighborhood_X,
                                        neighborhood_y)
    myexp.as_map()




if __name__ == '__main__':
    main()
