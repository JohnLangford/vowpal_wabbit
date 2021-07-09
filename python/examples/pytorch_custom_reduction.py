import sys, os
import math
import numpy as np
import torch

from scipy.sparse import coo_matrix
from vowpalwabbit import pyvw

class PyTorchReduction(pyvw.Copperhead):
    def __init__(self):
        super(PyTorchReduction, self).__init__()

    def _save_load(self):
        print("saving/loading")

    def reduction_init(self, vw):
        config = vw.get_config()
        self.num_bits = config["general"][6][1][6].value
        self.num_features = 1 << self.num_bits
        self.model = torch.nn.Linear(self.num_features, 2)
        # self.model = torch.nn.SparseLinear(self.num_features, 2)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.5) 
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda n: 1 / (1 + np.sqrt(1 + n)))
        self.lossfn = torch.nn.CrossEntropyLoss()

    def vw_ex_to_pytorch(self, ec):
        sparsej = []
        sparsev = []

        for ns_id in range(ec.num_namespaces()):
            # 128 ord_ns is constant namespace
            ord_ns = ec.namespace(ns_id)
            names = pyvw.namespace_id(ec, ns_id)
            for i in range(ec.num_features_in(names.ord_ns)):
                f = ec.feature(names.ord_ns, i)
                f = f & ((1 << self.num_bits) - 1)
                w = ec.feature_weight(names.ord_ns, i)
                sparsej.append(f)
                sparsev.append(w)

        X = coo_matrix((sparsev, ([0]*len(sparsej), sparsej)), 
                       shape=(1, self.num_features),
                       dtype=np.float32)

        values = X.data
        indices = np.vstack((X.row, X.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = X.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

        # return X

    def _predict(self, ec, learner, no_grad_enabled=True):
        if no_grad_enabled:
            with torch.no_grad():
                pred = self.model.forward(self.vw_ex_to_pytorch(ec))
                logits = self.logsoftmax(pred)
        else:
            pred = self.model.forward(self.vw_ex_to_pytorch(ec))
            logits = self.logsoftmax(pred)

        ec.set_partial_prediction(logits[0, -1].item() - np.log(0.5))
        ec.set_simplelabel_prediction(logits[0, -1].item() - np.log(0.5))

        return pred

    def _learn(self, ec, learner):
        self.optimizer.zero_grad()
        pred = self._predict(ec, learner, False)
        target = torch.LongTensor([ 1 if ec.get_simplelabel_label() > 0 else 0 ])
        # loss = ec.get_simplelabel_weight() * self.lossfn(pred, target)
        # missing requires_grad see line 57
        loss = self.lossfn(pred, target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    # see https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def _save_load(self, read, text):
        if read:
            # model.load_state_dict(torch.load(PATH))
            # model.eval()
            print(f'loading {read} {text}')
        else:
            # torch.save(model.state_dict(), PATH)
            print(f'saving {read} {text}')

def sanity_check():
    print(os.getpid())

    print(torch.__version__)

    if sys.version_info > (3, 0):
        print("good, python3")
    else:
        raise Exception("you are not on python 3")

def print_config(config):
    cmd_str = []

    for name, config_group in config.items():
        print(f'Reduction name: {name}')
        for (group_name, options) in config_group:
            print(f'\tOption group name: {group_name}')

            #if name == "general":
            #    continue

            for option in options:
                print(f'\t\tOption name: {option.name}, keep {option.keep}, help: {option.help_str}')
                temp_str = str(option)
                if temp_str:
                    cmd_str.append(temp_str)

    print(cmd_str)

# this should match cpp_binary() output
# doesn't do anything, runs in python see class impl NoopPythonicReductions
def noop_example():
    vw = pyvw.vw(python_reduction=PyTorchReduction, arg_str="--save_resume --loss_function logistic --binary  -d /root/vowpal_wabbit/test/train-sets/rcv1_small.dat")
    #print(vw.get_stride())
    vw.run_parser()

    # print_config(vw.get_config())
    vw.finish()
    #prediction = vw.predict("-1 |f 9:6.2699720e-02 14:3.3754818e-02")
    #vw.learn("-1 |f 9:6.2699720e-02 14:3.3754818e-02")

sanity_check()
print("noop")
noop_example()