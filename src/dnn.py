import chainer
import chainer.functions as F
import chainer.links as L

class DNN(chainer.Chain):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(DNN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_input, n_hidden1)
            self.l2 = L.Linear(n_hidden1, n_hidden2)
            self.l3 = L.Linear(n_hidden2, n_hidden3)
            self.l4 = L.Linear(n_hidden3, n_output)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        return self.l4(h)

