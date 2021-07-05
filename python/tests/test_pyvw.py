import os

from vowpalwabbit import pyvw
from vowpalwabbit.pyvw import vw
import pytest

BIT_SIZE = 18

# Since these tests still run with Python 2, this is required.
# Otherwise we could use math.isclose
def isclose(a, b, rel_tol=1e-05, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class TestVW:

    model = vw(quiet=True, b=BIT_SIZE)

    def test_constructor(self):
        assert isinstance(self.model, vw)

    def test_learn_predict(self):
        ex = self.model.example("1 | a b c")
        init = self.model.predict(ex)
        assert init == 0
        self.model.learn(ex)
        assert self.model.predict(ex) > init
        ex = ["| a", "| b"]
        check_error_raises(TypeError, lambda: self.model.predict(ex))
        check_error_raises(TypeError, lambda: self.model.learn(ex))

    def test_get_tag(self):
        ex = self.model.example("1 foo| a b c")
        assert ex.get_tag() == "foo"
        ex = self.model.example("1 1.0 bar| a b c")
        assert ex.get_tag() == "bar"
        ex = self.model.example("1 'baz | a b c")
        assert ex.get_tag() == "baz"

    def test_num_weights(self):
        assert self.model.num_weights() == 2 ** BIT_SIZE

    def test_get_weight(self):
        assert self.model.get_weight(0, 0) == 0

    def test_finish(self):
        assert not self.model.finished
        self.model.finish()
        assert self.model.finished


def test_delete():
    model = vw(quiet=True, b=BIT_SIZE)
    assert "model" in locals()
    del model
    assert "model" not in locals()


# Test prediction types


def test_scalar_prediction_type():
    model = vw(quiet=True)
    model.learn("1 | a b c")
    assert model.get_prediction_type() == model.pSCALAR
    prediction = model.predict(" | a b c")
    assert isinstance(prediction, float)
    del model


def test_scalars_prediction_type():
    n = 3
    model = vw(loss_function="logistic", oaa=n, probabilities=True, quiet=True)
    model.learn("1 | a b c")
    assert model.get_prediction_type() == model.pSCALARS
    prediction = model.predict(" | a b c")
    assert isinstance(prediction, list)
    assert len(prediction) == n
    del model


def test_multiclass_prediction_type():
    n = 3
    model = vw(loss_function="logistic", oaa=n, quiet=True)
    model.learn("1 | a b c")
    assert model.get_prediction_type() == model.pMULTICLASS
    prediction = model.predict(" | a b c")
    assert isinstance(prediction, int)
    del model


def test_prob_prediction_type():
    model = vw(
        loss_function="logistic",
        csoaa_ldf="mc",
        probabilities=True,
        quiet=True,
    )
    multi_ex = [
        model.example("1:0.2 | a b c"),
        model.example("2:0.8  | a b c"),
    ]
    model.learn(multi_ex)
    assert model.get_prediction_type() == model.pPROB
    multi_ex = [model.example("1 | a b c"), model.example("2 | a b c")]
    prediction = model.predict(multi_ex)
    assert isinstance(prediction, float)
    del model


def test_action_scores_prediction_type():
    model = vw(loss_function="logistic", csoaa_ldf="m", quiet=True)
    multi_ex = [model.example("1:1 | a b c"), model.example("2:-1  | a b c")]
    model.learn(multi_ex)
    assert model.get_prediction_type() == model.pMULTICLASS
    multi_ex = [model.example("1 | a b c"), model.example("2 | a b c")]
    prediction = model.predict(multi_ex)
    assert isinstance(prediction, int)
    del model


def test_action_probs_prediction_type():
    model = vw(cb_explore=2, ngram=2, quiet=True)
    model.learn("1 | a b c")
    assert model.get_prediction_type() == model.pACTION_PROBS
    prediction = model.predict(" | a b c")
    assert isinstance(prediction, list)
    del model


def test_multilabel_prediction_type():
    model = vw(multilabel_oaa=4, quiet=True)
    model.learn("1 | a b c")
    assert model.get_prediction_type() == model.pMULTILABELS
    prediction = model.predict(" | a b c")
    assert isinstance(prediction, list)
    del model


def test_cbandits_label():
    model = vw(cb=4, quiet=True)
    cbl = pyvw.cbandits_label(model.example("1:10:0.5 |"))
    assert cbl.costs[0].action == 1
    assert cbl.costs[0].probability == 0.5
    assert cbl.costs[0].partial_prediction == 0
    assert cbl.costs[0].cost == 10.0
    assert str(cbl) == "1:10.0:0.5"
    del model


def test_cost_sensitive_label():
    model = vw(csoaa=4, quiet=True)
    csl = pyvw.cost_sensitive_label(model.example("2:5 |"))
    assert csl.costs[0].label == 2
    assert csl.costs[0].wap_value == 0.0
    assert csl.costs[0].partial_prediction == 0.0
    assert csl.costs[0].cost == 5.0
    assert str(csl) == "2:5.0"
    del model


def test_multiclass_probabilities_label():
    n = 4
    model = pyvw.vw(
        loss_function="logistic", oaa=n, probabilities=True, quiet=True
    )
    ex = model.example("1 | a b c d", 2)
    model.learn(ex)
    mpl = pyvw.multiclass_probabilities_label(ex)
    assert str(mpl) == "1:0.25 2:0.25 3:0.25 4:0.25"
    mpl = pyvw.multiclass_probabilities_label([1, 2, 3], [0.4, 0.3, 0.3])
    assert str(mpl) == "1:0.4 2:0.3 3:0.3"


def test_regressor_args():
    # load and parse external data file
    data_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "resources", "train.dat"
    )
    model = vw(oaa=3, data=data_file, passes=30, c=True, k=True)
    assert model.predict("| feature1:2.5") == 1

    # update model in memory
    for _ in range(10):
        model.learn("3 | feature1:2.5")
    assert model.predict("| feature1:2.5") == 3

    # save model
    model.save("tmp.model")
    del model

    # load initial regressor and confirm updated prediction
    new_model = vw(i="tmp.model", quiet=True)
    assert new_model.predict("| feature1:2.5") == 3
    del new_model

    # clean up
    os.remove("{}.cache".format(data_file))
    os.remove("tmp.model")


def test_keys_with_list_of_values():
    # No exception in creating and executing model with a key/list pair
    model = vw(quiet=True, q=["fa", "fb"])
    model.learn("1 | a b c")
    prediction = model.predict(" | a b c")
    assert isinstance(prediction, float)
    del model


def helper_parse(examples):
    model = vw(quiet=True, cb_adf=True)
    ex = model.parse(examples)
    assert len(ex) == 2
    model.learn(ex)
    model.finish_example(ex)
    model.finish()


def test_parse():
    helper_parse("| a:1 b:0.5\n0:0.1:0.75 | a:0.5 b:1 c:2")

    helper_parse(
        """| a:1 b:0.5
    0:0.1:0.75 | a:0.5 b:1 c:2"""
    )

    helper_parse(
        """
    | a:1 b:0.5
    0:0.1:0.75 | a:0.5 b:1 c:2
    """
    )

    helper_parse(["| a:1 b:0.5", "0:0.1:0.75 | a:0.5 b:1 c:2"])


def test_parse_2():
    model = vw(quiet=True, cb_adf=True)
    ex = model.parse("| a:1 b:0.5\n0:0.1:0.75 | a:0.5 b:1 c:2")
    assert len(ex) == 2
    model.learn(ex)
    model.finish_example(ex)
    model.finish()

    model = vw(quiet=True, cb_adf=True)
    ex = model.parse(["| a:1 b:0.5", "0:0.1:0.75 | a:0.5 b:1 c:2"])
    assert len(ex) == 2
    model.learn(ex)
    model.finish_example(ex)
    model.finish()


def test_learn_predict_multiline():
    model = vw(quiet=True, cb_adf=True)
    ex = model.parse(["| a:1 b:0.5", "0:0.1:0.75 | a:0.5 b:1 c:2"])
    assert model.predict(ex) == [0.0, 0.0]
    model.finish_example(ex)
    ex = ["| a", "| b"]
    model.learn(ex)
    assert model.predict(ex) == [0.0, 0.0]


def test_namespace_id():
    vw_ex = vw(quiet=True)
    ex = vw_ex.example("1 |a two features |b more features here")
    nm1 = pyvw.namespace_id(ex, 0)
    nm2 = pyvw.namespace_id(ex, 1)
    nm3 = pyvw.namespace_id(ex, 2)
    assert nm1.id == 0
    assert nm1.ord_ns == 97
    assert nm1.ns == "a"
    assert nm2.id == 1
    assert nm2.ord_ns == 98
    assert nm2.ns == "b"
    assert nm3.id == 2
    assert nm3.ord_ns == 128
    assert nm3.ns == "\x80"  # Represents string of ord_ns


def test_example_namespace():
    vw_ex = vw(quiet=True)
    ex = vw_ex.example("1 |a two features |b more features here")
    ns_id = pyvw.namespace_id(ex, 1)
    ex_nm = pyvw.example_namespace(
        ex, ns_id, ns_hash=vw_ex.hash_space(ns_id.ns)
    )
    assert isinstance(ex_nm.ex, pyvw.example)
    assert isinstance(ex_nm.ns, pyvw.namespace_id)
    assert ex_nm.ns_hash == 2514386435
    assert ex_nm.num_features_in() == 3
    assert ex_nm[2] == (11617, 1.0)  # represents (feature, value)
    iter_obj = ex_nm.iter_features()
    for i in range(ex_nm.num_features_in()):
        assert ex_nm[i] == next(iter_obj)
    assert ex_nm.pop_feature()
    ex_nm.push_features(ns_id, ["c", "d"])
    assert ex_nm.num_features_in() == 4


def test_simple_label():
    sl = pyvw.simple_label(2.0, weight=0.5)
    assert sl.label == 2.0
    assert sl.weight == 0.5
    assert sl.prediction == 0.0
    assert sl.initial == 0.0
    assert str(sl) == "2.0:0.5"


def test_simple_label_example():
    vw_ex = vw(quiet=True)
    ex = vw_ex.example("1 |a two features |b more features here")
    sl2 = pyvw.simple_label(ex)
    assert sl2.label == 1.0
    assert sl2.weight == 1.0
    assert sl2.prediction == 0.0
    assert sl2.initial == 0.0
    assert str(sl2) == "1.0"


def test_multiclass_label():
    ml = pyvw.multiclass_label(2, weight=0.2)
    assert ml.label == 2
    assert ml.weight == 0.2
    assert ml.prediction == 1
    assert str(ml) == "2:0.2"


def test_multiclass_label_example():
    n = 4
    model = pyvw.vw(loss_function="logistic", oaa=n, quiet=True)
    ex = model.example("1 | a b c d", 2)
    ml2 = pyvw.multiclass_label(ex)
    assert ml2.label == 1
    assert ml2.weight == 1.0
    assert ml2.prediction == 0
    assert str(ml2) == "1"


def test_example_namespace_id():
    vw_ex = vw(quiet=True)
    ex = vw_ex.example("1 |a two features |b more features here")
    ns = pyvw.namespace_id(ex, 1)
    assert isinstance(ex.get_ns(1), pyvw.namespace_id)
    assert isinstance(ex[2], pyvw.example_namespace)
    assert ex.setup_done is True
    assert ex.num_features_in(ns) == 3


def test_example_learn():
    vw_ex = vw(quiet=True)
    ex = vw_ex.example("1 |a two features |b more features here")
    ex.learn()
    assert ex.setup_done is True
    ex.unsetup_example()  # unsetup an example as it is already setup
    assert ex.setup_done is False


def test_example_label():
    vw_ex = vw(quiet=True)
    ex = vw_ex.example("1 |a two features |b more features here")
    ex.set_label_string("1.0")
    assert isinstance(ex.get_label(), pyvw.simple_label)


def test_example_features():
    vw_ex = vw(quiet=True)
    ex = vw_ex.example("1 |a two features |b more features here")
    ns = pyvw.namespace_id(ex, 1)
    assert ex.get_feature_id(ns, "a") == 127530
    ex.push_hashed_feature(ns, 1122)
    ex.push_features("x", [("c", 1.0), "d"])
    ex.push_feature(ns, 11000)
    assert ex.num_features_in("x") == 2
    assert ex.sum_feat_sq(ns) == 5.0
    ns2 = pyvw.namespace_id(ex, 2)
    ex.push_namespace(ns2)
    assert ex.pop_namespace()

def test_num_namespaces_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.num_namespaces() == 6

def test_namespace_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.namespace(0) == ord('d')
    assert ex.namespace(1) == ord('d')
    assert ex.namespace(2) == ord('d')
    assert ex.namespace(3) == ord('a')
    assert ex.namespace(4) == ord('a')
    assert ex.namespace(5) == 128

def test_num_features_in_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.num_features_in('da') == 3
    assert ex.num_features_in('db') == 2
    assert ex.num_features_in('d') == 2
    assert ex.num_features_in('a') == 4
    assert ex.num_features_in('app') == 2

def test_feature_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.feature('a', 0) == ex.feature('apple', 0)
    assert ex.feature('a', 1) == ex.feature('apple', 1)
    assert ex.feature('d', 0) != ex.feature('da', 0)
    assert ex.feature('d', 1) != ex.feature('da', 1)

def test_feature_weight_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.feature_weight('d', 0) == 1
    assert ex.feature_weight('da', 0) == 1
    assert ex.feature_weight('db', 0) == 2
    assert ex.feature_weight('db', 1) == 3
    assert ex.feature_weight('a', 1) == 13
    assert ex.feature_weight('app', 1) == 1

def test_sum_feat_sq_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.sum_feat_sq('da') == 3
    assert ex.sum_feat_sq('db') == 13
    assert ex.sum_feat_sq('d') == 2
    assert ex.sum_feat_sq('a') == 292
    assert ex.sum_feat_sq('apple') == 290

def test_push_feature_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    ex.push_feature('d', 100, 2)
    assert ex.sum_feat_sq('d') == 6
    assert ex.num_features_in('d') == 3
    assert ex.num_features_in('da') == 3
    ex.push_feature('a', 101, 2)
    assert ex.sum_feat_sq('apple') == 294
    assert ex.num_features_in('apple') == 3
    assert ex.sum_feat_sq('app') == 2
    assert ex.num_features_in('app') == 2

def test_push_feature_list_namespace():
    model = vw(quiet=True)
    ex = model.example('1 |a two features |b more features here')
    ex.push_features('xa', ['a', 'b'])
    ex.push_features('xb', [('c', 2.), 'd'])
    space_hash = model.hash_space('xa')
    feat_hash  = model.hash_feature('a', space_hash)
    ex.push_features('xa', [feat_hash])
    assert ex.num_features_in('xa') == 3
    assert ex.num_features_in('xb') == 2
    assert ex.num_features_in('x') == 5
    assert ex.sum_feat_sq('xa') == 3
    assert ex.sum_feat_sq('xb') == 5
    assert ex.sum_feat_sq('x') == 8

def test_push_namespace_full_namespace():
    # This function will call get_or_create_feature_group which will overwrite
    # the constant namespace on the first run by default. This is why the
    # number of namespaces will remain the same on the first run.
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.num_namespaces() == 6
    ex.push_namespace('new')
    ex.push_feature('new',1,3)
    assert ex.sum_feat_sq('new') == 9
    assert ex.num_namespaces() == 6
    ex.push_namespace('new2')
    assert ex.num_namespaces() == 7
    ex.push_namespace('new2')
    assert ex.num_namespaces() == 7

def test_ensure_namespace_exists_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.num_namespaces() == 6
    ex.ensure_namespace_exists('new')
    ex.push_feature('new',1,3)
    assert ex.sum_feat_sq('new') == 9
    assert ex.num_namespaces() == 6
    ex.ensure_namespace_exists('new2')
    assert ex.num_namespaces() == 7
    ex.push_namespace('new2')
    assert ex.num_namespaces() == 7

def test_push_feature_dict_full_namespace():
    model = vw(quiet=True)
    ex = model.example({'da':[1, 3, 2], 'd':['more','feats'], 'db':[(3,2), (4,3)], 'apple':[(10, 11), (12,13)], 'app':[1, 2]})
    assert ex.num_features_in('da') == 3 
    assert ex.num_features_in('d') == 2
    assert ex.sum_feat_sq('a') == 292

def test_pop_feature_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    ex.pop_feature('d')
    assert ex.sum_feat_sq('da') == 3
    assert ex.sum_feat_sq('d') == 1
    ex.pop_feature('da')
    assert ex.sum_feat_sq('da') == 2
    ex.pop_feature('a')
    assert ex.sum_feat_sq('apple') == 121

def test_erase_namespace_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    ex.erase_namespace('da')
    assert ex.num_namespaces() == 5
    ex.erase_namespace('d')
    assert ex.num_namespaces() == 4
    ex.erase_namespace('a')
    assert ex.num_namespaces() == 2
    ex.erase_namespace('a')
    assert ex.num_namespaces() == 2

def test_pop_namespace_full_namespace():
    model = vw(quiet=True)
    ex = model.example('|da 1 3 2 |d more feats |db 3:2 4:3 |apple 10:11 12:13 |app 1 2')
    assert ex.num_namespaces() == 6
    assert ex.sum_feat_sq('a') == 292
    # Note that this will call unsetup_example and remove the constant namespace
    ex.pop_namespace()
    assert ex.sum_feat_sq('a') == 290
    assert ex.num_namespaces() == 4
    ex.pop_namespace()
    assert ex.num_namespaces() == 3
    ex.pop_namespace()
    assert ex.num_namespaces() == 2
    ex.pop_namespace()
    assert ex.num_namespaces() == 1
    ex.pop_namespace()
    assert ex.num_namespaces() == 0
    ex.pop_namespace()
    assert ex.num_namespaces() == 0

def test_get_weight_name():
    model = vw(quiet=True)
    model.learn("1 | a a b c |ns x")
    assert model.get_weight_from_name("a") != 0.
    assert model.get_weight_from_name("b") != 0.
    assert model.get_weight_from_name("b") == model.get_weight_from_name("c")
    assert model.get_weight_from_name("a") != model.get_weight_from_name("b")
    assert model.get_weight_from_name("x") == 0.
    assert model.get_weight_from_name("x", "ns") != 0.
    assert model.get_weight_from_name("x", "ns") == model.get_weight_from_name("b")


def test_runparser_cmd_string():
    vw = pyvw.vw("--data ./test/train-sets/rcv1_small.dat")
    assert vw.parser_ran == True, "vw should set parser_ran to true if --data present"
    vw.finish()

def test_runparser_cmd_string_short():
    vw = pyvw.vw("-d ./test/train-sets/rcv1_small.dat")
    assert vw.parser_ran == True, "vw should set parser_ran to true if --data present"
    vw.finish()

def test_not_runparser_cmd_string():
    vw = pyvw.vw("")
    assert vw.parser_ran == False, "vw should set parser_ran to false"
    vw.finish()

def check_error_raises(type, argument):
    """
    This function is used to check whether the exception is raised or not.

    Parameter
    ---------

    type: Type of Error raised
    argument: lambda function with no parameters.

    Example:
    >>> ex = ["|a", "|b"]
    >>> vw = pyvw.vw(quiet=True)
    >>> check_error_raises(TypeError, lambda: vw.learn(ex))

    """
    with pytest.raises(type) as error:
        argument()

def test_dsjson():
    vw = pyvw.vw('--cb_explore_adf --epsilon 0.2 --dsjson')

    ex_l_str='{"_label_cost":-1.0,"_label_probability":0.5,"_label_Action":1,"_labelIndex":0,"o":[{"v":1.0,"EventId":"38cbf24f-70b2-4c76-aa0c-970d0c8d388e","ActionTaken":false}],"Timestamp":"2020-11-15T17:09:31.8350000Z","Version":"1","EventId":"38cbf24f-70b2-4c76-aa0c-970d0c8d388e","a":[1,2],"c":{ "GUser":{"id":"person5","major":"engineering","hobby":"hiking","favorite_character":"spock"}, "_multi": [ { "TAction":{"topic":"SkiConditions-VT"} }, { "TAction":{"topic":"HerbGarden"} } ] },"p":[0.5,0.5],"VWState":{"m":"N/A"}}\n'
    ex_l = vw.parse(ex_l_str)
    vw.learn(ex_l)
    pred = ex_l[0].get_action_scores()
    expected = [0.5, 0.5]
    assert len(pred) == len(expected)
    for a,b in zip(pred, expected):
        assert isclose(a, b)
    vw.finish_example(ex_l)

    ex_p='{"_label_cost":-1.0,"_label_probability":0.5,"_label_Action":1,"_labelIndex":0,"o":[{"v":1.0,"EventId":"38cbf24f-70b2-4c76-aa0c-970d0c8d388e","ActionTaken":false}],"Timestamp":"2020-11-15T17:09:31.8350000Z","Version":"1","EventId":"38cbf24f-70b2-4c76-aa0c-970d0c8d388e","a":[1,2],"c":{ "GUser":{"id":"person5","major":"engineering","hobby":"hiking","favorite_character":"spock"}, "_multi": [ { "TAction":{"topic":"SkiConditions-VT"} }, { "TAction":{"topic":"HerbGarden"} } ] },"p":[0.5,0.5],"VWState":{"m":"N/A"}}\n'
    pred = vw.predict(ex_p)
    expected = [0.9, 0.1]
    assert len(pred) == len(expected)
    for a,b in zip(pred, expected):
        assert isclose(a, b)

def test_dsjson_with_metrics():
    vw = pyvw.vw('--extra_metrics metrics.json --cb_explore_adf --epsilon 0.2 --dsjson')

    ex_l_str='{"_label_cost":-0.9,"_label_probability":0.5,"_label_Action":1,"_labelIndex":0,"o":[{"v":1.0,"EventId":"38cbf24f-70b2-4c76-aa0c-970d0c8d388e","ActionTaken":false}],"Timestamp":"2020-11-15T17:09:31.8350000Z","Version":"1","EventId":"38cbf24f-70b2-4c76-aa0c-970d0c8d388e","a":[1,2],"c":{ "GUser":{"id":"person5","major":"engineering","hobby":"hiking","favorite_character":"spock"}, "_multi": [ { "TAction":{"topic":"SkiConditions-VT"} }, { "TAction":{"topic":"HerbGarden"} } ] },"p":[0.5,0.5],"VWState":{"m":"N/A"}}\n'
    ex_l = vw.parse(ex_l_str)
    vw.learn(ex_l)
    pred = ex_l[0].get_action_scores()
    expected = [0.5, 0.5]
    assert len(pred) == len(expected)
    for a,b in zip(pred, expected):
        assert isclose(a, b)
    vw.finish_example(ex_l)

    ex_p='{"_label_cost":-1.0,"_label_probability":0.5,"_label_Action":1,"_labelIndex":0,"o":[{"v":1.0,"EventId":"38cbf24f-70b2-4c76-aa0c-970d0c8d388e","ActionTaken":false}],"Timestamp":"2020-11-15T17:09:31.8350000Z","Version":"1","EventId":"38cbf24f-70b2-4c76-aa0c-970d0c8d388e","a":[1,2],"c":{ "GUser":{"id":"person5","major":"engineering","hobby":"hiking","favorite_character":"spock"}, "_multi": [ { "TAction":{"topic":"SkiConditions-VT"} }, { "TAction":{"topic":"HerbGarden"} } ] },"p":[0.5,0.5],"VWState":{"m":"N/A"}}\n'
    pred = vw.predict(ex_p)
    expected = [0.9, 0.1]
    assert len(pred) == len(expected)
    for a,b in zip(pred, expected):
        assert isclose(a, b)

    learner_metric_dict = vw.get_learner_metrics()
    assert(len(vw.get_learner_metrics()) == 17)

    assert(learner_metric_dict["total_predict_calls"] == 2)
    assert(learner_metric_dict["total_learn_calls"] == 1)
    assert(learner_metric_dict["cbea_labeled_ex"] == 1)
    assert(learner_metric_dict["cbea_predict_in_learn"] == 0)
    assert(learner_metric_dict["cbea_label_first_action"] == 1)
    assert(learner_metric_dict["cbea_label_not_first"] == 0)
    assert(pytest.approx(learner_metric_dict["cbea_sum_cost"]) == -0.9)
    assert(pytest.approx(learner_metric_dict["cbea_sum_cost_baseline"]) == -0.9)
    assert(learner_metric_dict["cbea_non_zero_cost"] == 1)
    assert(pytest.approx(learner_metric_dict["cbea_avg_feat_per_event"]) == 18)
    assert(pytest.approx(learner_metric_dict["cbea_avg_actions_per_event"]) == 2)
    assert(pytest.approx(learner_metric_dict["cbea_avg_ns_per_event"]) == 12)
    assert(pytest.approx(learner_metric_dict["cbea_avg_feat_per_action"]) == 9)
    assert(pytest.approx(learner_metric_dict["cbea_avg_ns_per_action"]) == 6)
    assert(learner_metric_dict["cbea_min_actions"] == 2)
    assert(learner_metric_dict["cbea_max_actions"] == 2)
    assert(learner_metric_dict["sfm_count_learn_example_with_shared"] == 1)

def test_constructor_exception_is_safe():
    try:
        vw = pyvw.vw("--invalid_option")
    except:
        pass
