# -*- coding: utf-8 -*-
"""Python binding for pylibvw class"""

from __future__ import division
import pylibvw
import warnings
from vowpalwabbit.DFtoVWtoTensorboard import VWtoTensorboard


# baked in con py boost https://wiki.python.org/moin/boost.python/FAQ#The_constructors_of_some_classes_I_am_trying_to_wrap_are_private_because_instances_must_be_created_by_using_a_factory._Is_it_possible_to_wrap_such_classes.3F
class VWOption:
    def __init__(self, name, help_str, short_name, keep, necessary, allow_override, value, value_supplied, default_value, default_value_supplied):
        self._name = name
        self._help_str = help_str
        self._short_name = short_name
        self._keep = keep
        self._necessary = necessary
        self._allow_override = allow_override
        self._value = value
        self._value_supplied = value_supplied
        self._default_value = default_value
        self._default_value_supplied = default_value_supplied

    @property
    def name(self):
        return self._name

    @property
    def help_str(self):
        return self._help_str

    @property
    def short_name(self):
        return self._short_name

    @property
    def keep(self):
        return self._keep

    @property
    def necessary(self):
        return self._necessary

    @property
    def allow_override(self):
        return self._allow_override

    @property
    def value_supplied(self):
        return self._value_supplied

    @property
    def default_value(self):
        return self._default_value

    @property
    def default_value_supplied(self):
        return self._default_value_supplied

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value_supplied = True
        self._value = val

    def is_flag(self):
        return type(self._default_value) == bool or (self.value_supplied and type(self.value) == bool)

    def __str__(self):
        if self.value_supplied:
            if self.is_flag():
                return "--{}".format(self.name)
            else:
                # missing list case
                if isinstance(self.value, list):
                    return "**NOT_IMPL**"
                else:
                    return "--{} {}".format(self.name, self.value)
        else:
            return ''

class SearchTask:
    """Search task class"""

    def __init__(self, vw, sch, num_actions):
        """
        Parameters
        ----------

        vw : vw object
        sch : search object
        num_actions : integer
            The number of actions with the object can be initialized with

        Returns
        -------

        self : SearchTask

        See Also
        --------

        pyvw.vw

        """

        self.vw = vw
        self.sch = sch
        self.bogus_example = [self.vw.example("1 | x")]

    def __del__(self):
        self.vw.finish_example(self.bogus_example)

    def _run(self, your_own_input_example):
        pass

    def _call_vw(
        self, my_example, isTest, useOracle=False
    ):  # run_fn, setup_fn, takedown_fn, isTest):
        self._output = None
        self.bogus_example[0].set_test_only(isTest)

        def run():
            self._output = self._run(my_example)

        setup = None
        takedown = None
        if callable(getattr(self, "_setup", None)):
            setup = lambda: self._setup(my_example)
        if callable(getattr(self, "_takedown", None)):
            takedown = lambda: self._takedown(my_example)
        self.sch.set_structured_predict_hook(run, setup, takedown)
        self.sch.set_force_oracle(useOracle)
        self.vw.learn(
            self.bogus_example
        )  # this will cause our ._run hook to get called

    def learn(self, data_iterator):
        """Train search task by providing an iterator of examples.

        Parameters
        ----------

        data_iterator: iterable objects
            Consists of examples to be learned

        Returns
        -------

        self : SearchTask

        """
        for my_example in data_iterator.__iter__():
            self._call_vw(my_example, isTest=False)

    def example(self, initStringOrDict=None, labelType=pylibvw.vw.lDefault):
        """Create an example initStringOrDict can specify example as VW
        formatted string, or a dictionary labelType can specify the desire
        label type

        Parameters
        ----------

        initStringOrDict : str/dict
            Example in either string or dictionary form
        labelType : integer
            - 0 : lDEFAULT
            - 1 : lBINARY
            - 2 : lMULTICLASS
            - 3 : lCOST_SENSITIVE
            - 4 : lCONTEXTUAL_BANDIT
            - 5 : lMAX
            - 6 : lCONDITIONAL_CONTEXTUAL_BANDIT
            - 7 : lSLATES
            - 8 : lCONTINUOUS
            The integer is used to map the corresponding labelType using the
            above available options

        Returns
        -------

        out : Example

        """
        if self.sch.predict_needs_example():
            return self.vw.example(initStringOrDict, labelType)
        else:
            return self.vw.example(None, labelType)

    def predict(self, my_example, useOracle=False):
        """Predict on the example

        Parameters
        ----------

        my_example : Example
            example used for prediction
        useOracle : bool

        Returns
        -------

        out : integer
            Prediction on the example
        """
        self._call_vw(my_example, isTest=True, useOracle=useOracle)
        return self._output


def get_label(example, label_type):
    """Similar to pyvw.get_prediction implementation this method gets the label for current example

    Parameters
    ----------

    example    : example object
                    Methods of this example object are used to get the appropiate label
    label_type : integer
                    This is the integer for representing a specific type of label in vw
                    - 0: lDEFAULT
                    - 1: lBINARY
                    - 2: lMULTICLASS
                    - 3: lCOST_SENSITIVE
                    - 4: lCONTEXTUAL_BANDIT
                    - 5: lMAX
                    - 6: lCONDITIONAL_CONTEXTUAL_BANDIT
                    - 7: lSLATES
                    - 8: lCONTINUOUS

    Returns
    -------

    label  : integer
                Actual label of the current example object according to label type
    """
    switch_label_type = {
        pylibvw.vw.lDefault: None,
        pylibvw.vw.lBinary: example.get_simplelabel_label,
        pylibvw.vw.lMulticlass: example.get_multiclass_label,
        pylibvw.vw.lCostSensitive: example.get_costsensitive_class,
        pylibvw.vw.lContextualBandit: example.get_cbandits_class,
#         pylibvw.vw.lConditionalContextualBandit: None,
#         pylibvw.vw.lSlates: None,
#         pylibvw.vw.lContinuous: None
    }
    return switch_label_type[label_type]()  


def get_prediction(ec, prediction_type):
    """Get specified type of prediction from example

    Parameters
    ----------

    ec : Example
    prediction_type : integer
        - 0: pSCALAR
        - 1: pSCALARS
        - 2: pACTION_SCORES
        - 3: pACTION_PROBS
        - 4: pMULTICLASS
        - 5: pMULTILABELS
        - 6: pPROB
        - 7: pMULTICLASSPROBS
        - 8: pDECISION_SCORES
        - 9: pACTION_PDF_VALUE
        - 10: pPDF

    Examples
    --------

    >>> from vowpalwabbit import pyvw
    >>> import pylibvw
    >>> vw = pyvw.vw(quiet=True)
    >>> ex = vw.example('1 |a two features |b more features here')
    >>> pyvw.get_prediction(ex, pylibvw.vw.pSCALAR)
    0.0

    Returns
    -------

    out : integer/list
        Prediction according to parameter prediction_type
    """
    switch_prediction_type = {
        pylibvw.vw.pSCALAR: ec.get_simplelabel_prediction,
        pylibvw.vw.pSCALARS: ec.get_scalars,
        pylibvw.vw.pACTION_SCORES: ec.get_action_scores,
        pylibvw.vw.pACTION_PROBS: ec.get_action_scores,
        pylibvw.vw.pMULTICLASS: ec.get_multiclass_prediction,
        pylibvw.vw.pMULTILABELS: ec.get_multilabel_predictions,
        pylibvw.vw.pPROB: ec.get_prob,
        pylibvw.vw.pMULTICLASSPROBS: ec.get_scalars,
        pylibvw.vw.pDECISION_SCORES: ec.get_decision_scores,
        pylibvw.vw.pACTION_PDF_VALUE: ec.get_action_pdf_value,
        pylibvw.vw.pPDF: ec.get_pdf,
        pylibvw.vw.pACTIVE_MULTICLASS: ec.get_active_multiclass,
    }
    return switch_prediction_type[prediction_type]()

def get_all_vw_options():
    temp = vw("--dry_run")
    config = temp.get_config(filtered_enabled_reductions_only=False)
    temp.finish()
    return config

class log_forward:
    def __init__(self):
        self.current_message = ""
        self.messages = []

    def log(self, msg):
        self.current_message += msg
        if "\n" in self.current_message:
            self.messages.append(self.current_message)
            self.current_message = ""


class vw(pylibvw.vw):
    """The pyvw.vw object is a (trivial) wrapper around the pylibvw.vw
    object; you're probably best off using this directly and ignoring
    the pylibvw.vw structure entirely."""
    log_wrapper = None
    parser_ran = False
    init = False
    finished = False
    log_fwd = None

    def __init__(self, arg_str=None, enable_logging=False, **kw):
        """Initialize the vw object.

        Parameters
        ----------

        arg_str : str
            The command line arguments to initialize VW with,
            for example "--audit". By default is None.

        **kw : Using key/value pairs for different options available

        Examples
        --------

        >>> from vowpalwabbit import pyvw
        >>> vw1 = pyvw.vw('--audit')
        >>> vw2 = pyvw.vw(audit=True, b=24, k=True, c=True, l2=0.001)
        >>> vw3 = pyvw.vw("--audit", b=26)
        >>> vw4 = pyvw.vw(q=["ab", "ac"])

        Returns
        -------

        self : vw
        """

        def format_input_pair(key, val):
            if type(val) is bool and not val:
                s = ""
            else:
                prefix = "-" if len(key) == 1 else "--"
                value = "" if type(val) is bool else " {}".format(val)
                s = "{p}{k}{v}".format(p=prefix, k=key, v=value)
            return s

        def format_input(key, val):
            if isinstance(val, list):
                # if a list is passed as a parameter value - create a key for
                # each list element
                return " ".join(
                    [format_input_pair(key, value) for value in val]
                )
            else:
                return format_input_pair(key, val)

        l = [format_input(k, v) for k, v in kw.items()]
        if arg_str is not None:
            l = [arg_str] + l

        if enable_logging:
            self.log_fwd = log_forward()
            self.log_wrapper = pylibvw.vw_log(self.log_fwd)

        if self.log_wrapper:
            super(vw, self).__init__(" ".join(l), self.log_wrapper)
        else:
            super(vw, self).__init__(" ".join(l))
        self.init = True

        # check to see if native parser needs to run
        ext_file_args = ["d", "data", "passes"]
        if any(x in kw for x in ext_file_args):
            pylibvw.vw.run_parser(self)
            self.parser_ran = True
        elif arg_str:
            # space after -d to avoid matching with other substrings
            ext_file_cmd_str = ["-d ", "--data", "--passes"]
            if [cmd for cmd in ext_file_cmd_str if(cmd in arg_str)]:
                pylibvw.vw.run_parser(self)
                self.parser_ran = True

    def get_config(self, filtered_enabled_reductions_only=True):
        return self.get_options(VWOption, filtered_enabled_reductions_only)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()

    def parse(self, str_ex, labelType=pylibvw.vw.lDefault):
        """Returns a collection of examples for a multiline example learner or
        a single example for a single example learner.

        Parameters
        ----------

        str_ex : str/list of str
            string representing examples. If the string is multiline then each
            line is considered as an example. In case of list, each string
            element is considered as an example
        labelType : integer
            - 0 : lDEFAULT
            - 1 : lBINARY
            - 2 : lMULTICLASS
            - 3 : lCOST_SENSITIVE
            - 4 : lCONTEXTUAL_BANDIT
            - 5 : lMAX
            - 6 : lCONDITIONAL_CONTEXTUAL_BANDIT
            - 7 : lSLATES
            - 8 : lCONTINUOUS
            The integer is used to map the corresponding labelType using the
            above available options

        Examples
        --------

        >>> from vowpalwabbit import pyvw
        >>> model = pyvw.vw(quiet=True)
        >>> ex = model.parse("0:0.1:0.75 | a:0.5 b:1 c:2")
        >>> type(ex)
        <class 'vowpalwabbit.pyvw.example'>
        >>> model = pyvw.vw(quiet=True, cb_adf=True)
        >>> ex = model.parse(["| a:1 b:0.5", "0:0.1:0.75 | a:0.5 b:1 c:2"])
        >>> type(ex)
        <class 'list'>
        >>> len(ex) # Shows the multiline example is parsed
        2

        Returns
        -------

        ec : list
            list of examples parsed
        """

        # check if already parsed
        if isinstance(str_ex, example) and getattr(str_ex, "setup_done", None):
            return str_ex

        elif isinstance(str_ex, list):
            if all([getattr(ex, "setup_done", None) for ex in str_ex]):
                return str_ex

        if not isinstance(str_ex, (list, str)):
            raise TypeError(
                "Unsupported type. List or string object must be passed."
            )

        if isinstance(str_ex, list):
            str_ex = "\n".join(str_ex)
        str_ex = str_ex.replace("\r", "")
        str_ex = str_ex.strip()
        ec = self._parse(str_ex)
        ec = [example(self, x, labelType) for x in ec]
        for ex in ec:
            ex.setup_done = True
        if not self._is_multiline():
            if len(ec) == 1:
                ec = ec[0]
            else:
                raise TypeError(
                    "expecting single line example, got multi_ex of len %i"
                    % len(ec)
                )
        return ec

    def finish_example(self, ex):
        """Should only be used in conjunction with the parse method

        Parameters
        ----------

        ex : Example
            example to be finished
        """

        if isinstance(ex, example):
            if self._is_multiline():
                raise ValueError(
                    "Learner is multiline but single example was passed to "
                    "finish_example. Use the list of examples instead?"
                )
            if not ex.finished:
                pylibvw.vw._finish_example(self, ex)
                ex.finished = True
        elif isinstance(ex, list):
            if not self._is_multiline():
                raise ValueError(
                    "Learner is singleline but multi example was passed to "
                    "finish_example. Use a single example instead?"
                )
            if all(x.finished == False for x in ex):
                pylibvw.vw._finish_example_multi_ex(self, ex)
                for x in ex:
                    x.finished = True

    def num_weights(self):
        """Get length of weight vector."""
        return pylibvw.vw.num_weights(self)

    def get_weight(self, index, offset=0):
        """Get the weight at a particular position in the (learned) weight
        vector.

        Parameters
        ----------

        index : integer
            position in the learned  weight vector
        offset : integer
            By default is 0

        Returns
        -------

        weight : float
            Weight at the given index

        """
        return pylibvw.vw.get_weight(self, index, offset)

    def get_weight_from_name(self, feature_name, namespace_name=" "):
        """Get the weight based on the feature name and the namespace name.

        Parameters
        ----------
        feature_name : str
            The name of the feature
        namespace_name : str, by default is ""
            The name of the namespace where the feature lives

        Returns
        -------
        weight : float
            Weight for the given feature and namespace name
        """
        space_hash = self.hash_space(namespace_name)
        feat_hash = self.hash_feature(feature_name, space_hash)
        return self.get_weight(feat_hash)

    def learn(self, ec, vw_to_tensorboard=None):
        """Perform an online update

        Parameters
        ----------

        ec : example/str/list
            examples on which the model gets updated

        vw_to_tensorboard : VWtoTensorboard object
                           This object is used to calculate metrics and then log them for Tensorboard visualization
        """
        if isinstance(vw_to_tensorboard, VWtoTensorboard):
            sum_loss = super().get_sum_loss()
            weighted_examples = super().get_weighted_examples()

        # If a string was given, parse it before passing to learner.
        new_example = False
        if isinstance(ec, str):
            ec = self.parse(ec)
            new_example = True

        elif isinstance(ec, list):
            if not self._is_multiline():
                raise TypeError("Expecting a mutiline Learner.")
            ec = self.parse(ec)
            new_example = True

        if isinstance(ec, example):
            if not getattr(ec, "setup_done", None):
                ec.setup_example()
            pylibvw.vw.learn(self, ec)

        elif isinstance(ec, list):
            pylibvw.vw.learn_multi(self, ec)

        else:
            raise TypeError(
                "expecting string or example object as ec argument for learn,"
                " got %s" % type(ec)
            )

        if new_example:
            self.finish_example(ec)

        if isinstance(vw_to_tensorboard, VWtoTensorboard):
            if new_example is False:  # As according to this method's logic self.finish_example(ec) is only called  if new_example==True  thats why  if new_example==False then self.finish_example(ec) has not been called
                raise RuntimeError("finish_example on current example has not been called, metrics may not be computed correctly")

            sum_loss_since_last = super().get_sum_loss() - sum_loss  # .get_sum_loss() return current sum loss, sum_loss variable right now holds sum loss of previous iteration
            weighted_examples_since_last = super().get_weighted_examples() - weighted_examples  # .get_weighted_examples() return current weighted examples(sum),  weighted_examples variable right now holds weighted examples of previous iteration             
            sum_loss = super().get_sum_loss()  # Now sum_loss no longer hold previous iteration's sum_loss
            weighted_examples = super().get_weighted_examples()  # Now weighted_examples no longer hold previous iteration's weighted examples

            average_loss = (sum_loss / weighted_examples) if weighted_examples != 0  else 0.0
            since_last = (sum_loss_since_last / weighted_examples_since_last) if weighted_examples_since_last != 0  else 0.0

            vw_to_tensorboard.emit_learning_metrics(average_loss, since_last)

    def predict(self, ec, prediction_type=None):
        """Just make a prediction on the example

        Parameters
        ----------

        ec : Example/list/str
            examples to be predicted
        prediction_type : optional, by default is None
            if provided then the matching return type is
            used otherwise the the learner's prediction type will determine the
            output.

        Returns
        -------

        prediction : Prediction made on each examples
        """

        new_example = False
        if isinstance(ec, dict):
            ec = self.example(ec)
            ec.setup_done = True
            new_example = True

        # If a string was given, parse it before passing to learner.
        elif isinstance(ec, str):
            ec = self.parse(ec)
            new_example = True

        elif isinstance(ec, list):
            if not self._is_multiline():
                raise TypeError("Expecting a multiline Learner.")
            ec = self.parse(ec)
            new_example = True

        if not isinstance(ec, example) and not isinstance(ec, list):
            raise TypeError(
                "expecting string, example object, or list of example objects"
                " as ec argument for predict, got %s" % type(ec)
            )

        if isinstance(ec, example) and not getattr(ec, "setup_done", True):
            ec.setup_example()

        if isinstance(ec, example):
            pylibvw.vw.predict(self, ec)
        else:
            pylibvw.vw.predict_multi(self, ec)

        if prediction_type is None:
            prediction_type = pylibvw.vw.get_prediction_type(self)

        if isinstance(ec, example):
            prediction = get_prediction(ec, prediction_type)
        else:
            prediction = get_prediction(ec[0], prediction_type)

        if new_example:
            self.finish_example(ec)

        return prediction

    def save(self, filename):
        """save model to disk"""
        pylibvw.vw.save(self, filename)

    def finish(self):
        """stop VW by calling finish (and, eg, write weights to disk)"""
        if not self.finished and self.init:
            pylibvw.vw.finish(self)
            self.init = False
            self.finished = True

    # returns the latest vw log
    # call after vw.finish() for complete log
    # useful for debugging
    def get_log(self):
        if self.log_fwd:
            return self.log_fwd.messages
        else:
            raise Exception("enable_logging set to false")

    def example(self, stringOrDict=None, labelType=pylibvw.vw.lDefault):
        """Create an example initStringOrDict can specify example as VW
        formatted string, or a dictionary labelType can specify the desire
        label type

        Parameters
        ----------

        initStringOrDict : str/dict
            Example in either string or dictionary form
        labelType : integer
            - 0 : lDEFAULT
            - 1 : lBINARY
            - 2 : lMULTICLASS
            - 3 : lCOST_SENSITIVE
            - 4 : lCONTEXTUAL_BANDIT
            - 5 : lMAX
            - 6 : lCONDITIONAL_CONTEXTUAL_BANDIT
            - 7 : lSLATES
            - 8 : lCONTINUOUS
            The integer is used to map the corresponding labelType using the
            above available options

        Returns
        -------

        out : Example

        """
        return example(self, stringOrDict, labelType)

    def __del__(self):
        self.finish()

    def init_search_task(self, search_task, task_data=None):
        sch = self.get_search_ptr()

        def predict(
            examples,
            my_tag,
            oracle,
            condition=None,
            allowed=None,
            learner_id=0,
        ):
            """The basic (via-reduction) prediction mechanism

            Parameters
            ----------

            examples : Example/list
                can be a single example (interpreted as
                non-LDF mode) or a list of examples (interpreted as
                LDF mode).  it can also be a lambda function that
                returns a single example or list of examples, and in
                that list, each element can also be a lambda function
                that returns an example. this is done for lazy
                example construction (aka speed).

            my_tag : integer
                should be an integer id, specifying this prediction

            oracle : label/list
                can be a single label (or in LDF mode a single
                array index in 'examples') or a list of such labels if
                the oracle policy is indecisive; if it is None, then
                the oracle doesn't care

            condition : Optional, by default is None
                should be either: (1) a (tag,char) pair, indicating
                to condition on the given tag with identifier from the char;
                or (2) a (tag,len,char) triple, indicating to condition on
                tag, tag-1, tag-2, ..., tag-len with identifiers char,
                char+1, char+2, ..., char+len. or it can be a (heterogenous)
                list of such things.

            allowed : optional, by default id None
                can be None, in which case all actions are allowed;
                or it can be list of valid actions (in LDF mode, this should
                be None and you should encode the valid actions in 'examples')

            learner_id : integer
                specifies the underlying learner id

            Returns
            -------

            out : integer
                a single prediction.

            """
            P = sch.get_predictor(my_tag)
            if sch.is_ldf():
                # we need to know how many actions there are, even if we don't
                # know their identities
                while hasattr(examples, "__call__"):
                    examples = examples()
                if not isinstance(examples, list):
                    raise TypeError(
                        "expected example _list_ in LDF mode for SearchTask.predict()"
                    )
                P.set_input_length(len(examples))
                if sch.predict_needs_example():
                    for n in range(len(examples)):
                        ec = examples[n]
                        while hasattr(ec, "__call__"):
                            ec = ec()  # unfold the lambdas
                        if not isinstance(ec, example) and not isinstance(
                            ec, pylibvw.example
                        ):
                            raise TypeError(
                                "non-example in LDF example list in SearchTask.predict()"
                            )
                        if hasattr(ec, "setup_done") and not ec.setup_done:
                            ec.setup_example()
                        P.set_input_at(n, ec)
                else:
                    pass  # TODO: do we need to set the examples even though they're not used?
            else:
                if sch.predict_needs_example():
                    while hasattr(examples, "__call__"):
                        examples = examples()
                    if (
                        hasattr(examples, "setup_done")
                        and not examples.setup_done
                    ):
                        examples.setup_example()
                    P.set_input(examples)
                else:
                    pass  # TODO: do we need to set the examples even though they're not used?

            # if (isinstance(examples, list) and all([isinstance(ex, example) or
            #       isinstance(ex, pylibvw.example) for ex in examples])) or \
            #    isinstance(examples, example) or isinstance(examples, pylibvw.example):
            #     if isinstance(examples, list): # LDF
            #         P.set_input_length(len(examples))
            #         for n in range(len(examples)):
            #             P.set_input_at(n, examples[n])
            #     else: # non-LDF
            #         P.set_input(examples)
            if oracle is None:
                pass
            elif isinstance(oracle, list):
                assert 0 not in oracle, "multiclass labels are from 1..., "
                "please do not use zero or bad things will happen!"
                if len(oracle) > 0:
                    P.set_oracles(oracle)
            elif isinstance(oracle, int):
                assert oracle > 0, "multiclass labels are from 1...,"
                " please do not use zero or bad things will happen!"
                P.set_oracle(oracle)
            else:
                raise TypeError("expecting oracle to be a list or an integer")

            if condition is not None:
                if not isinstance(condition, list):
                    condition = [condition]
                for c in condition:
                    if not isinstance(c, tuple):
                        raise TypeError(
                            "item "
                            + str(c)
                            + " in condition list is malformed"
                        )
                    if (
                        len(c) == 2
                        and isinstance(c[0], int)
                        and isinstance(c[1], str)
                        and len(c[1]) == 1
                    ):
                        P.add_condition(max(0, c[0]), c[1])
                    elif (
                        len(c) == 3
                        and isinstance(c[0], int)
                        and isinstance(c[1], int)
                        and isinstance(c[2], str)
                        and len(c[2]) == 1
                    ):
                        P.add_condition_range(max(0, c[0]), max(0, c[1]), c[2])
                    else:
                        raise TypeError(
                            "item " + str(c) + " in condition list malformed"
                        )

            if allowed is None:
                pass
            elif isinstance(allowed, list):
                assert 0 not in allowed, "multiclass labels are from 1...,"
                "please do not use zero or bad things will happen!"
                P.set_alloweds(allowed)
            else:
                raise TypeError("allowed argument wrong type")

            if learner_id != 0:
                P.set_learner_id(learner_id)

            p = P.predict()
            return p

        sch.predict = predict
        num_actions = sch.get_num_actions()
        return (
            search_task(self, sch, num_actions)
            if task_data is None
            else search_task(self, sch, num_actions, task_data)
        )


class namespace_id:
    """The namespace_id class is simply a wrapper to convert between
    hash spaces referred to by character (eg 'x') versus their index
    in a particular example. Mostly used internally, you shouldn't
    really need to touch this."""

    def __init__(self, ex, id):
        """Given an example and an id, construct a namespace_id.

        Parameters
        ----------

        ex : Example
            example used to create a namespace id
        id : integer/str
            The id can either be an integer (in which case we take it to be an
            index into ex.indices[]) or a string (in which case we take
            the first character as the namespace id).

        Returns
        -------

        self : namespace_id
        """
        if isinstance(id, int):  # you've specified a namespace by index
            if id < 0 or id >= ex.num_namespaces():
                raise Exception("namespace " + str(id) + " out of bounds")
            self.id = id
            self.ord_ns = ex.namespace(id)
            self.ns = chr(self.ord_ns)
        elif isinstance(id, str):  # you've specified a namespace by string
            if len(id) == 0:
                id = " "
            self.id = None  # we don't know and we don't want to do the linear search required to find it
            self.ns = id[0]
            self.ord_ns = ord(self.ns)
        else:
            raise Exception(
                "ns_to_characterord failed because id type is unknown: "
                + str(type(id))
            )


class example_namespace:
    """The example_namespace class is a helper class that allows you
    to extract namespaces from examples and operate at a namespace
    level rather than an example level. Mainly this is done to enable
    indexing like ex['x'][0] to get the 0th feature in namespace 'x'
    in example ex."""

    def __init__(self, ex, ns, ns_hash=None):
        """Construct an example_namespace

        Parameters
        ----------

        ex : Example
            examples from which namespace is to be extracted
        ns : namespace_id
            Target namespace
        ns_hash : Optional, by default is None
            The hash of the namespace

        Returns
        -------

        self : example_namespace
        """
        if not isinstance(ns, namespace_id):
            raise TypeError("ns should an instance of namespace_id.")
        self.ex = ex
        self.ns = ns
        self.ns_hash = ns_hash

    def num_features_in(self):
        """Return the total number of features in this namespace."""
        return self.ex.num_features_in(self.ns)

    def __getitem__(self, i):
        """Get the feature/value pair for the ith feature in this
        namespace."""
        f = self.ex.feature(self.ns, i)
        v = self.ex.feature_weight(self.ns, i)
        return (f, v)

    def iter_features(self):
        """iterate over all feature/value pairs in this namespace."""
        for i in range(self.num_features_in()):
            yield self[i]

    def push_feature(self, feature, v=1.0):
        """Add an unhashed feature to the current namespace (fails if
        setup has already run on this example).

        Parameters
        ----------

        feature : integer/str
            Feature to be pushed to current namespace
        v : float
            Feature value, by default is 1.0

        """
        if self.ns_hash is None:
            self.ns_hash = self.ex.vw.hash_space(self.ns)
        self.ex.push_feature(self.ns, feature, v, self.ns_hash)

    def pop_feature(self):
        """Remove the top feature from the current namespace; returns True
        if a feature was removed, returns False if there were no
        features to pop."""
        return self.ex.pop_feature(self.ns)

    def push_features(self, ns, featureList):
        """Push a list of features to a given namespace.

        Parameters
        ----------

        ns : namespace
            namespace to which feature list is to be pushed
        featureList : list
            Each feature in the list can either be an integer (already hashed)
            or a string (to be hashed) and may be paired with a value or not
            (if not, the value is assumed to be 1.0).
        See example.push_features for examples.
        """
        self.ex.push_features(self.ns, featureList)


class abstract_label:
    """An abstract class for a VW label."""

    def __init__(self):
        pass

    def from_example(self, ex):
        """grab a label from a given VW example"""
        raise Exception("from_example not yet implemented")


class simple_label(abstract_label):
    """Class for simple VW label"""

    def __init__(self, label=0.0, weight=1.0, initial=0.0, prediction=0.0):
        if not isinstance(label, (example, float)):
            raise TypeError("Label should be float or an example.")
        abstract_label.__init__(self)
        if isinstance(label, example):
            self.from_example(label)
        else:
            self.label = label
            self.weight = weight
            self.initial = initial
            self.prediction = prediction

    def from_example(self, ex):
        self.label = ex.get_simplelabel_label()
        self.weight = ex.get_simplelabel_weight()
        self.initial = ex.get_simplelabel_initial()
        self.prediction = ex.get_simplelabel_prediction()

    def __str__(self):
        s = str(self.label)
        if self.weight != 1.0:
            s += ":" + str(self.weight)
        return s


class multiclass_label(abstract_label):
    """Class for multiclass VW label with prediction"""

    def __init__(self, label=1, weight=1.0, prediction=1):
        if not isinstance(label, (example, int)):
            raise TypeError("Label should be integer or an example.")
        abstract_label.__init__(self)
        if isinstance(label, example):
            self.from_example(label)
        else:
            self.label = label
            self.weight = weight
            self.prediction = prediction

    def from_example(self, ex):
        self.label = ex.get_multiclass_label()
        self.weight = ex.get_multiclass_weight()
        self.prediction = ex.get_multiclass_prediction()

    def __str__(self):
        s = str(self.label)
        if self.weight != 1.0:
            s += ":" + str(self.weight)
        return s


class multiclass_probabilities_label(abstract_label):
    """Class for multiclass VW label with probabilities"""

    def __init__(self, label, prediction=None):
        abstract_label.__init__(self)
        if isinstance(label, example):
            self.from_example(label)
        else:
            self.prediction = prediction

    def from_example(self, ex):
        self.prediction = get_prediction(ex, pylibvw.vw.pMULTICLASSPROBS)

    def __str__(self):
        s = []
        for label, prediction in enumerate(self.prediction):
            s.append("{l}:{p}".format(l=label + 1, p=prediction))
        return " ".join(s)


class cost_sensitive_label(abstract_label):
    """Class for cost sensative VW label"""

    def __init__(self, costs=[], prediction=0):
        abstract_label.__init__(self)
        if isinstance(costs, example):
            self.from_example(costs)
        else:
            self.costs = costs
            self.prediction = prediction

    def from_example(self, ex):
        class wclass:
            def __init__(
                self, label, cost=0.0, partial_prediction=0.0, wap_value=0.0
            ):
                self.label = label
                self.cost = cost
                self.partial_prediction = partial_prediction
                self.wap_value = wap_value

        self.prediction = ex.get_costsensitive_prediction()
        self.costs = []
        for i in range(ex.get_costsensitive_num_costs()):
            wc = wclass(
                ex.get_costsensitive_class(i),
                ex.get_costsensitive_cost(i),
                ex.get_costsensitive_partial_prediction(i),
                ex.get_costsensitive_wap_value(i),
            )
            self.costs.append(wc)

    def __str__(self):
        return " ".join(["{}:{}".format(c.label, c.cost) for c in self.costs])


class cbandits_label(abstract_label):
    """Class for contextual bandits VW label"""

    def __init__(self, costs=[], prediction=0):
        abstract_label.__init__(self)
        if isinstance(costs, example):
            self.from_example(costs)
        else:
            self.costs = costs
            self.prediction = prediction

    def from_example(self, ex):
        class wclass:
            def __init__(
                self,
                action=None,
                cost=0.0,
                partial_prediction=0.0,
                probability=0.0,
                **kwargs
            ):
                if kwargs.get("label", False):
                    action = kwargs["label"]
                    warnings.warn(
                        "label has been deprecated. Please use 'action' instead.",
                        DeprecationWarning,
                    )
                self.label = action
                self.action = action
                self.cost = cost
                self.partial_prediction = partial_prediction
                self.probability = probability

        self.prediction = ex.get_cbandits_prediction()
        self.costs = []
        for i in range(ex.get_cbandits_num_costs()):
            wc = wclass(
                ex.get_cbandits_class(i),
                ex.get_cbandits_cost(i),
                ex.get_cbandits_partial_prediction(i),
                ex.get_cbandits_probability(i),
            )
            self.costs.append(wc)

    def __str__(self):
        return " ".join(
            [
                "{}:{}:{}".format(c.action, c.cost, c.probability)
                for c in self.costs
            ]
        )


class example(pylibvw.example):
    """The example class is a (non-trivial) wrapper around
    pylibvw.example. Most of the wrapping is to make the interface
    easier to use (by making the types safer via namespace_id) and
    also with added python-specific functionality."""

    def __init__(
        self,
        vw,
        initStringOrDictOrRawExample=None,
        labelType=pylibvw.vw.lDefault,
    ):
        """Construct a new example from vw.

        Parameters
        ----------

        vw : vw
            vw model
        initStringOrDictOrRawExample : dict/string/None
            If initString is None, you get an "empty" example which you
            can construct by hand (see, eg, example.push_features).
            If initString is a string, then this string is parsed as
            it would be from a VW data file into an example (and
            "setup_example" is run). if it is a dict, then we add all
            features in that dictionary. finally, if it's a function,
            we (repeatedly) execute it fn() until it's not a function
            any more(for lazy feature computation). By default is None
        labelType : integer
            - 0 : lDEFAULT
            - 1 : lBINARY
            - 2 : lMULTICLASS
            - 3 : lCOST_SENSITIVE
            - 4 : lCONTEXTUAL_BANDIT
            - 5 : lMAX
            - 6 : lCONDITIONAL_CONTEXTUAL_BANDIT
            - 7 : lSLATES
            - 8 : lCONTINUOUS
            The integer is used to map the corresponding labelType using the
            above available options

        Returns
        -------

        self : Example

        See Also
        --------

        pyvw.vw

        """

        while hasattr(initStringOrDictOrRawExample, "__call__"):
            initStringOrDictOrRawExample = initStringOrDictOrRawExample()

        if initStringOrDictOrRawExample is None:
            pylibvw.example.__init__(self, vw, labelType)
            self.setup_done = False
        elif isinstance(initStringOrDictOrRawExample, str):
            pylibvw.example.__init__(
                self, vw, labelType, initStringOrDictOrRawExample
            )
            self.setup_done = True
        elif isinstance(initStringOrDictOrRawExample, pylibvw.example):
            pylibvw.example.__init__(
                self, vw, labelType, initStringOrDictOrRawExample
            )
        elif isinstance(initStringOrDictOrRawExample, dict):
            pylibvw.example.__init__(self, vw, labelType)
            self.vw = vw
            self.stride = vw.get_stride()
            self.finished = False
            self.push_feature_dict(vw, initStringOrDictOrRawExample)
            self.setup_done = False
        else:
            raise TypeError(
                "expecting string or dict as argument for example construction"
            )

        self.vw = vw
        self.stride = vw.get_stride()
        self.finished = False
        self.labelType = labelType

    def get_ns(self, id):
        """Construct a namespace_id

        Parameters
        ----------

        id : namespace_id/str/integer
            id used to create namespace

        Returns
        -------

        out : namespace_id
            namespace_id created using parameter passed(if id was namespace_id,
            just return it directly)
        """
        if isinstance(id, namespace_id):
            return id
        else:
            return namespace_id(self, id)

    def __getitem__(self, id):
        """Get an example_namespace object associated with the given
        namespace id."""
        return example_namespace(self, self.get_ns(id))

    def feature(self, ns, i):
        """Get the i-th hashed feature id in a given namespace

        Parameters
        ----------

        ns : namespace
            namespace used to get the feature
        i : integer
            to get i-th hashed feature id in a given ns. It must range from
            0 to self.num_features_in(ns)-1

        Returns
        -------

        f : integer
            i-th hashed feature-id in a given ns
        """
        ns = self.get_ns(ns)  # guaranteed to be a single character
        f = pylibvw.example.feature(self, ns.ord_ns, i)
        if self.setup_done:
            f = (f - self.get_ft_offset()) // self.stride
        return f

    def feature_weight(self, ns, i):
        """Get the value(weight) associated with a given feature id

        Parameters
        ----------

        ns : namespace
            namespace used to get the feature id
        i : integer
            to get the weight of i-th feature in the given ns. It must range
            from 0 to self.num_features_in(ns)-1

        Returns
        -------

        out : float
            weight(value) of the i-th feature of given ns
        """
        return pylibvw.example.feature_weight(self, self.get_ns(ns).ord_ns, i)

    def set_label_string(self, string):
        """Give this example a new label

        Parameters
        ----------

        string : str
            a new label to this example, formatted as a string (ala the VW data
            file format)
        """
        pylibvw.example.set_label_string(self, self.vw, string, self.labelType)

    def setup_example(self):
        """If this example hasn't already been setup (ie, quadratic
        features constructed, etc.), do so."""
        if self.setup_done:
            raise Exception(
                "Trying to setup_example on an example that is already setup"
            )
        self.vw.setup_example(self)
        self.setup_done = True

    def unsetup_example(self):
        """If this example has been setup, reverse that process so you can
         continue editing the examples."""
        if not self.setup_done:
            raise Exception(
                "Trying to unsetup_example that has not yet been setup"
            )
        self.vw.unsetup_example(self)
        self.setup_done = False

    def learn(self):
        """Learn on this example (and before learning, automatically
        call setup_example if the example hasn't yet been setup)."""
        if not self.setup_done:
            self.setup_example()
        self.vw.learn(self)

    def sum_feat_sq(self, ns):
        """Get the total sum feature-value squared for a given
        namespace

        Parameters
        ----------

        ns : namespace
            Get the total sum feature-value squared of this namespace

        Returns
        -------

        sum_sq : float
            Total sum feature-value squared of the given ns
        """
        return pylibvw.example.sum_feat_sq(self, self.get_ns(ns).ord_ns)

    def num_features_in(self, ns):
        """Get the total number of features in a given namespace

        Parameters
        ----------

        ns : namespace
            Get the total features of this namespace

        Returns
        -------

        num_features : integer
            Total number of features in the given ns
        """
        return pylibvw.example.num_features_in(self, self.get_ns(ns).ord_ns)

    def get_feature_id(self, ns, feature, ns_hash=None):
        """Get the hashed feature id for a given feature in a given
        namespace. feature can either be an integer (already a feature
        id) or a string, in which case it is hashed.

        Parameters
        ----------

        ns : namespace
            namespace used to get the feature
        feature : integer/string
            If integer the already a feature else will be hashed
        ns_hash : Optional, by default is None
            The hash of the namespace

        Returns
        -------

        out : integer
            Hashed feature id

        .. note::
        If --hash all is on, then get_feature_id(ns,"5") !=
        get_feature_id(ns, 5). If you've already hashed the namespace,
        you can optionally provide that value to avoid re-hashing it.
        """
        if isinstance(feature, int):
            return feature
        if isinstance(feature, str):
            if ns_hash is None:
                ns_hash = self.vw.hash_space(self.get_ns(ns).ns)
            return self.vw.hash_feature(feature, ns_hash)
        raise Exception(
            "cannot extract feature of type: " + str(type(feature))
        )

    def push_hashed_feature(self, ns, f, v=1.0):
        """Add a hashed feature to a given namespace.

        Parameters
        ----------

        ns : namespace
            namespace in which the feature is to be pushed
        f : integer
            feature
        v : float
            The value of the feature, be default is 1.0
        """
        if self.setup_done:
            self.unsetup_example()
        pylibvw.example.push_hashed_feature(self, self.get_ns(ns).ord_ns, f, v)

    def push_feature(self, ns, feature, v=1.0, ns_hash=None):
        """Add an unhashed feature to a given namespace

        Parameters
        ----------

        ns : namespace
            namespace in which the feature is to be pushed
        f : integer
            feature
        v : float
            The value of the feature, be default is 1.0
        ns_hash : Optional, by default is None
            The hash of the namespace
        """
        f = self.get_feature_id(ns, feature, ns_hash)
        self.push_hashed_feature(ns, f, v)

    def pop_feature(self, ns):
        """Remove the top feature from a given namespace

        Parameters
        ----------

        ns : namespace
            namespace from which feature is popped

        Returns
        -------

        out : bool
            True if feature was removed else False as no feature was there to
            pop
        """
        if self.setup_done:
            self.unsetup_example()
        return pylibvw.example.pop_feature(self, self.get_ns(ns).ord_ns)

    def push_namespace(self, ns):
        """Push a new namespace onto this example.
        You should only do this if you're sure that this example doesn't
        already have the given namespace

        Parameters
        ----------

        ns : namespace
            namespace which is to be pushed onto example

        """
        if self.setup_done:
            self.unsetup_example()
        pylibvw.example.push_namespace(self, self.get_ns(ns).ord_ns)

    def pop_namespace(self):
        """Remove the top namespace from an example

        Returns
        -------

        out : bool
            True if namespace was removed else False as no namespace was there
            to pop
        """
        if self.setup_done:
            self.unsetup_example()
        return pylibvw.example.pop_namespace(self)

    def ensure_namespace_exists(self, ns):
        """Check to see if a namespace already exists.

        Parameters
        ----------

        ns : namespace
            If namespace exists does, do nothing. If it doesn't, add it.
        """
        if self.setup_done:
            self.unsetup_example()
        return pylibvw.example.ensure_namespace_exists(
            self, self.get_ns(ns).ord_ns
        )

    def push_features(self, ns, featureList):
        """Push a list of features to a given namespace.

        Parameters
        ----------

        ns :  namespace
            namespace in which the features are pushed
        featureList : list
            Each feature in the list can either be an integer
            (already hashed) or a string (to be hashed) and may be
            paired with a value or not (if not, the value is assumed to be 1.0

        Examples
        --------

        >>> from vowpalwabbit import pyvw
        >>> vw = pyvw.vw(quiet=True)
        >>> ex = vw.example('1 |a two features |b more features here')
        >>> ex.push_features('x', ['a', 'b'])
        >>> ex.push_features('y', [('c', 1.), 'd'])
        >>> space_hash = vw.hash_space('x')
        >>> feat_hash  = vw.hash_feature('a', space_hash)
        >>> ex.push_features('x', [feat_hash]) #'x' should match the space_hash!
        >>> ex.num_features_in('x')
        3
        >>> ex.num_features_in('y')
        2
        """
        ns = self.get_ns(ns)
        self.ensure_namespace_exists(ns)
        self.push_feature_list(
            self.vw, ns.ord_ns, featureList
        )  # much faster just to do it in C++
        # ns_hash = self.vw.hash_space( ns.ns )
        # for feature in featureList:
        #     if isinstance(feature, int) or isinstance(feature, str):
        #         f = feature
        #         v = 1.
        #     elif isinstance(feature, tuple) and len(feature) == 2 and
        #       (isinstance(feature[0], int) or isinstance(feature[0], str))
        #    and (isinstance(feature[1], int) or isinstance(feature[1], float)):
        #         f = feature[0]
        #         v = feature[1]
        #     else:
        #         raise Exception('malformed feature to push of type: '
        #                        + str(type(feature)))
        #     self.push_feature(ns, f, v, ns_hash)

    def iter_features(self):
        """Iterate over all feature/value pairs in this example (all
        namespace included)."""
        for ns_id in range(
            self.num_namespaces()
        ):  # iterate over every namespace
            ns = self.get_ns(ns_id)
            for i in range(self.num_features_in(ns)):
                f = self.feature(ns, i)
                v = self.feature_weight(ns, i)
                yield f, v

    def get_label(self, label_class=simple_label):
        """Given a known label class (default is simple_label), get
        the corresponding label structure for this example.

        Parameters
        ----------

        label_class : label classes
            Get the label of the example of label_class type, by default is
            simple_label
        """
        return label_class(self)

