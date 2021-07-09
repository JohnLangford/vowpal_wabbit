import sys
import math

from vowpalwabbit import pyvw

class NoopPythonReduction(pyvw.Copperhead):
    def __init__(self):
        super(NoopPythonReduction, self).__init__()

    def reduction_init(self, vw):
        pass

    def _predict(self, ec, learner):
        learner.predict(ec)

    def _learn(self, ec, learner):
        learner.learn(ec)

# this is a recreation of the impl in vw
# see https://github.com/VowpalWabbit/vowpal_wabbit/blob/ac3a2c21a9760b68ce49368b11a35bf95faeb8b8/vowpalwabbit/binary.cc
class BinaryPythonReduction(pyvw.Copperhead):
    def reduction_init(self, vw):
        pass

    def _predict(self, ec, learner):
        learner.predict(ec)

    def _learn(self, ec, learner):
        learner.learn(ec)

        if ec.get_simplelabel_prediction() > 0:
            ec.set_simplelabel_prediction(1)
        else:
            ec.set_simplelabel_prediction(-1)

        temp_label = ec.get_simplelabel_label()

        # going to have to expose FLT_MAX?
        if temp_label != sys.float_info.max:
            if math.fabs(temp_label) != 1.0:
                print("You are using label " + temp_label + " not -1 or 1 as loss function expects!")
            elif temp_label == ec.get_scalar():
                ec.set_loss(0)
            else:
                ec.set_loss(ec.get_simplelabel_weight())

# this should match cpp_binary() output
# doesn't do anything, runs in python see class impl NoopPythonicReductions
def noop_example():
    vw = pyvw.vw(enable_logging=True, python_reduction=NoopPythonReduction, arg_str="--loss_function logistic --binary -d test/train-sets/rcv1_small.dat")
    vw.run_parser()
    vw.finish()
    return vw.get_log()

# this should match cpp_binary() output
# mirror implementation of the cpp, runs in python see class impl BinaryPythonReductions
def python_binary():
    vw = pyvw.vw(enable_logging=True, python_reduction=BinaryPythonReduction, arg_str="--loss_function logistic -d test/train-sets/rcv1_small.dat")
    vw.run_parser()
    vw.finish()
    return vw.get_log()

# this should be the baseline
def cpp_binary():
    vw = pyvw.vw("--loss_function logistic --binary -d test/train-sets/rcv1_small.dat", enable_logging=True)
    vw.run_parser()
    vw.finish()
    return vw.get_log()

import pytest

def helper_compare_binary_output(a, b):
    assert len(a) == len(b)
    line_number = 0
    for j, i in zip(a, b):
        if line_number == 7:
            assert "Enabled reductions" in j
            assert "Enabled reductions" in i
            line_number += 1
            continue

        assert i == j, "line mismatch should be: " + j + " output: " + i
        line_number += 1

def test_python_binary_reduction():
    noop_binary_log = noop_example()
    python_binary_log = python_binary()
    native_binary_log = cpp_binary()

    helper_compare_binary_output(python_binary_log, native_binary_log)
    helper_compare_binary_output(noop_binary_log, native_binary_log)
