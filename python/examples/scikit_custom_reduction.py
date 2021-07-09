import sys, os
import math
import numpy as np

from scipy.sparse import coo_matrix
from vowpalwabbit import pyvw

class ScikitGDReduction(pyvw.Copperhead):
    def __init__(self):
        super(ScikitGDReduction, self).__init__()

    def reduction_init(self, vw):
        from sklearn.linear_model import SGDClassifier

        # set classes (0 and 1 for binary -> labels)
        self.classes = np.array([0,1])

        config = vw.get_config()
        # not ideal, WIP config api
        self.num_bits = config["general"][6][1][6].value

        self.num_features = 1 << self.num_bits

        self.classifier = SGDClassifier(loss='log')

        fakeX = coo_matrix(([], ([], [])), shape=(1, self.num_features))
        fakeY = np.array([0])
        self.classifier.partial_fit(fakeX, fakeY, classes=self.classes)

    def vw_ex_to_scikit(self, ec):
        sparsej = []
        sparsev = []

        for ns_id in range(ec.num_namespaces()):
            names = pyvw.namespace_id(ec, ns_id)
            for i in range(ec.num_features_in(names.ord_ns)):
                f = ec.feature(names.ord_ns, i)
                f = f & ((1 << self.num_bits) - 1)

                # sanity check
                assert(f < (self.num_features))

                w = ec.feature_weight(names.ord_ns, i)
                sparsej.append(f)
                sparsev.append(w)

        X = coo_matrix((sparsev, ([0]*len(sparsej), sparsej)), 
                       shape=(1, self.num_features),
                       dtype=np.float32)
        
        return X

    def _predict(self, ec, learner):
        pred = self.classifier.predict_log_proba(X = self.vw_ex_to_scikit(ec))

        prediction_f = pred[0, -1] - np.log(0.5)
        ec.set_partial_prediction(prediction_f)
        ec.set_simplelabel_prediction(prediction_f)

    def _learn(self, ec, learner):
        self._predict(ec, learner)
        self.classifier.partial_fit(
            X = self.vw_ex_to_scikit(ec),
            y = [ 1 if ec.get_simplelabel_label() > 0 else 0 ],
            sample_weight = [ ec.get_simplelabel_weight() ])

    # see https://scikit-learn.org/stable/modules/model_persistence.html
    def _save_load(self, read, text, modelIO):
        modelIO.read_write()
        import pickle
        if read:
            print(f'loading {read} {text}')
            # pickle.loads(s)
        else:
            s = pickle.dumps(self.classifier)
            print(f'saving {read} {text}')
            # print(s)

    # todo saving and loading
    #  vw vs on python (give me away of loading this thing)

def sanity_check():
    print(os.getpid())

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


def run_example():
    # vw = pyvw.vw(python_reduction=ScikitGDReduction, arg_str="--loss_function logistic --binary -d /root/vw/test/train-sets/rcv1_small.dat")
    vw = pyvw.vw(python_reduction=ScikitGDReduction, arg_str="-f hola.model --save_resume --loss_function logistic -d /root/vowpal_wabbit/test/train-sets/rcv1_small.dat")

    vw.run_parser()
    config = vw.get_config()

    #vw.learn("-1 |f 9:6.2699720e-02 14:3.3754818e-02")
    #prediction = vw.predict("-1 |f 9:6.2699720e-02 14:3.3754818e-02")

    vw.finish()
    #print_config(config)

sanity_check()
print("Running example...")
run_example()