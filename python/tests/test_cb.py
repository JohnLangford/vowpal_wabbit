import pandas as pd
import sklearn as sk
import numpy as np
from os import path
from vowpalwabbit import pyvw

import pytest

def helper_get_test_dir():
    curr_path = path.dirname(path.realpath(__file__))
    return path.join(path.dirname(path.dirname(curr_path)), "test")

def helper_get_data():
    train_data = [{'action': 1, 'cost': 2, 'probability': 0.4, 'feature1': 'a', 'feature2': 'c', 'feature3': ''},
                {'action': 3, 'cost': 0, 'probability': 0.2, 'feature1': 'b', 'feature2': 'd', 'feature3': ''},
                {'action': 4, 'cost': 1, 'probability': 0.5, 'feature1': 'a', 'feature2': 'b', 'feature3': ''},
                {'action': 2, 'cost': 1, 'probability': 0.3, 'feature1': 'a', 'feature2': 'b', 'feature3': 'c'},
                {'action': 3, 'cost': 1, 'probability': 0.7, 'feature1': 'a', 'feature2': 'd', 'feature3': ''}]

    train_df = pd.DataFrame(train_data)

    train_df['index'] = range(1, len(train_df) + 1)
    train_df = train_df.set_index("index")

    test_data = [{'feature1': 'b', 'feature2': 'c', 'feature3': ''},
                {'feature1': 'a', 'feature2': '', 'feature3': 'b'},
                {'feature1': 'b', 'feature2': 'b', 'feature3': ''},
                {'feature1': 'a', 'feature2': '', 'feature3': 'b'}]

    test_df = pd.DataFrame(test_data)

    # Add index to data frame
    test_df['index'] = range(1, len(test_df) + 1)
    test_df = test_df.set_index("index")

    return train_df, test_df

def test_getting_started_example_cb():
    return helper_getting_started_example("--cb")

def test_getting_started_example_legacy_cb():
    return helper_getting_started_example("--cb_force_legacy --cb")

def helper_getting_started_example(which_cb):
    train_df, test_df = helper_get_data()

    vw = pyvw.vw(which_cb + " 4", enable_logging=True)

    for i in train_df.index:
        action = train_df.loc[i, "action"]
        cost = train_df.loc[i, "cost"]
        probability = train_df.loc[i, "probability"]
        feature1 = train_df.loc[i, "feature1"]
        feature2 = train_df.loc[i, "feature2"]
        feature3 = train_df.loc[i, "feature3"]
        
        learn_example = str(action) + ":" + str(cost) + ":" + str(probability) + " | " + str(feature1) + " " + str(feature2) + " " + str(feature3)
        vw.learn(learn_example)

    assert vw.get_prediction_type() == vw.pMULTICLASS, "prediction_type should be multiclass"

    for j in test_df.index:
        feature1 = test_df.loc[j, "feature1"]
        feature2 = test_df.loc[j, "feature2"]
        feature3 = test_df.loc[j, "feature3"]
        choice = vw.predict("| "+str(feature1)+" "+str(feature2)+" "+str(feature3))
        assert isinstance(choice, int), "choice should be int"
        assert choice == 3, "predicted action should be 3 instead of " + str(choice)

    # test that metrics is empty since "--extra_metrics filename" was not supplied
    assert(len(vw.get_learner_metrics()) == 0)

    vw.finish()

    output = vw.get_log()

    if which_cb.find("legacy") != -1:
        test_file = "test-sets/ref/python_test_cb_legacy.stderr"
    else:
        test_file = "test-sets/ref/python_test_cb.stderr"
    
    with open(path.join(helper_get_test_dir(), test_file), 'r') as file:
        actual = file.readlines()
        assert len(actual) == len(output)
        for j, i in zip(actual, output):
            assert i == j, "line mismatch should be: " + j + " output: " + i

def test_getting_started_example_with():
    train_df, test_df = helper_get_data()

    # with syntax calls into vw.finish() automatically.
    # you actually want to use 'with pyvw.vw("--cb 4") as vw:'
    # but we need to assert on vw.finished for test purposes
    vw = pyvw.vw("--cb 4")
    with vw as vw:
        for i in train_df.index:
            action = train_df.loc[i, "action"]
            cost = train_df.loc[i, "cost"]
            probability = train_df.loc[i, "probability"]
            feature1 = train_df.loc[i, "feature1"]
            feature2 = train_df.loc[i, "feature2"]
            feature3 = train_df.loc[i, "feature3"]
            
            learn_example = str(action) + ":" + str(cost) + ":" + str(probability) + " | " + str(feature1) + " " + str(feature2) + " " + str(feature3)
            vw.learn(learn_example)

        assert vw.get_prediction_type() == vw.pMULTICLASS, "prediction_type should be multiclass"

        for j in test_df.index:
            feature1 = test_df.loc[j, "feature1"]
            feature2 = test_df.loc[j, "feature2"]
            feature3 = test_df.loc[j, "feature3"]
            choice = vw.predict("| "+str(feature1)+" "+str(feature2)+" "+str(feature3))
            assert isinstance(choice, int), "choice should be int"
            assert choice == 3, "predicted action should be 3"
    
    assert vw.finished == True, "with syntax should finish() vw instance"
