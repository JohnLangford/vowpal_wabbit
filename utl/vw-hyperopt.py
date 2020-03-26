#!/usr/bin/env python
# coding: utf-8

"""
Github version of hyperparameter optimization for Vowpal Wabbit via hyperopt
"""


"""
TODO: Add option: cb, cb_explore and cb_explore_adf
TODO: Add option for cb_type
TODO: Training and testing for multiple seeds
"""

__author__ = 'kurtosis'

from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK
from sklearn.metrics import roc_curve, auc, log_loss, average_precision_score, hinge_loss, \
    mean_squared_error
import numpy as np
from datetime import datetime as dt
import subprocess, shlex
from math import exp, log
import argparse
import re
import logging
import json
from matplotlib import pyplot as plt
try:
    import seaborn as sns
except ImportError:
    print ("Warning: seaborn is not installed. "
           "Without seaborn, standard matplotlib plots will not look very charming. "
           "It's recommended to install it via pip install seaborn")


def read_arguments():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='subcommand')
    parser_cb = subparsers.add_parser('is_cb', help="It is contextual bandit problelms")
    parser_not_cb = subparsers.add_parser('not_cb', help="Not a contextual bandit problelms")
    parser_cb.set_defaults(is_cb=True)
    parser_not_cb.set_defaults(is_cb=False)

    parser.add_argument('--searcher', type=str, default='tpe', choices=['tpe', 'rand'])
    parser.add_argument('--additional_cmd', type=str, help="Additional arguments to be passed to vw while tuning hyper params. E.g.: '--keep a --keep b'", default='')
    parser.add_argument('--max_evals', type=int, default=100)
    
    parser.add_argument('--train', type=str, required=True, help="training set")
    parser.add_argument('--holdout', type=str, required=True, help="holdout set")
  #  parser_cb.add_argument('--iterations', type=str, default='1000', help="number of iterations")

    # TODO: Number for seeds on which training should be done for a particular hyperparameter setting
    parser_cb.add_argument('--train_seed', type=int, default='3', help="number of seeds for training") 
    # TODO: Number for seeds on which testing should be done for a particular hyperparameter setting 
    parser_cb.add_argument('--test_seeed', type=int, default='5', help="number of seeds for testing")

    parser.add_argument('--vw_space', type=str, required=True, help="hyperparameter search space (must be 'quoted')")
    parser_not_cb.add_argument('--outer_loss_function', default='logistic',
                        choices=['logistic', 'roc-auc', 'pr-auc', 'hinge', 'squared', 'quantile'])  
    parser_not_cb.add_argument('--regression', action='store_true', default=False, help="""regression (continuous class labels)
                                                                        or classification (-1 or 1, default value).""")
    parser.add_argument('--plot', action='store_true', default=False, help=("Plot the results in the end. "
                                                                            "Requires matplotlib and "
                                                                            "(optionally) seaborn to be installed."))
    args = parser.parse_args()
    return args


class HyperoptSpaceConstructor(object):
    """
    Takes command-line input and transforms it into hyperopt search space
    An example of command-line input:
    --algorithms=ftrl,sgd --l2=1e-8..1e-4~LO -l=0.01..10~L --ftrl_beta=0.01..1 --passes=1..10~I -q=SE+SZ+DR,SE~O
    """

    def __init__(self, command):
        self.command = command
        self.space = None  

        self.range_pattern = re.compile("[^~]+")  # re.compile("(?<=\[).+(?=\])")
        self.distr_pattern = re.compile("(?<=~)[IOL]*")  # re.compile("(?<=\])[IOL]*")
        self.only_continuous = re.compile("(?<=~)[IL]*")  # re.compile("(?<=\])[IL]*")

    def _process_vw_argument(self, arg, value, algorithm):
        try:
            distr_part = self.distr_pattern.findall(value)[0]
        except IndexError:
            distr_part = ''
        range_part = self.range_pattern.findall(value)[0]
        is_continuous = '..' in range_part

        ocd = self.only_continuous.findall(value)
        if not is_continuous and len(ocd)> 0 and ocd[0] != '':
            raise ValueError(("Need a range instead of a list of discrete values to define "
                              "uniform or log-uniform distribution. "
                              "Please, use [min..max]%s form") % (distr_part))

        if is_continuous and arg == '-q':
            raise ValueError(("You must directly specify namespaces for quadratic features "
                              "as a list of values, not as a parametric distribution"))

        hp_choice_name = "_".join([algorithm, arg.replace('-', '')])

        try_omit_zero = 'O' in distr_part
        distr_part = distr_part.replace('O', '')

        if is_continuous:
            vmin, vmax = [float(i) for i in range_part.split('..')]

            if distr_part == 'L':
                distrib = hp.loguniform(hp_choice_name, log(vmin), log(vmax))
            elif distr_part == '':
                distrib = hp.uniform(hp_choice_name, vmin, vmax)
            elif distr_part == 'I':
                distrib = hp.quniform(hp_choice_name, vmin, vmax, 1)
            elif distr_part in {'LI', 'IL'}:
                distrib = hp.qloguniform(hp_choice_name, log(vmin), log(vmax), 1)
            else:
                raise ValueError("Cannot recognize distribution: %s" % (distr_part))
        else:
            possible_values = range_part.split(',')
            if arg == '-q':
                possible_values = [v.replace('+', ' -q ') for v in possible_values]
            distrib = hp.choice(hp_choice_name, possible_values)

        if try_omit_zero:
            hp_choice_name_outer = hp_choice_name + '_outer'
            distrib = hp.choice(hp_choice_name_outer, ['omit', distrib])

        return distrib

class HyperoptSpaceConstructorNotCB(HyperoptSpaceConstructor):
    """
    Takes command-line input and transforms it into hyperopt search space
    An example of command-line input:
    --algorithms=ftrl,sgd --l2=1e-8..1e-4~LO -l=0.01..10~L --ftrl_beta=0.01..1 --passes=1..10~I -q=SE+SZ+DR,SE~O
    """

    def __init__(self, command):  
        self.algorithm_metadata = {
            'ftrl': {'arg': '--ftrl', 'prohibited_flags': set()},
            'sgd': {'arg': '', 'prohibited_flags': {'--ftrl_alpha', '--ftrl_beta'}}
        }

        HyperoptSpaceConstructor.__init__(self, command)

    def string_to_pyll(self):
        line = shlex.split(self.command)

        algorithms = ['sgd']
        for arg in line:
            arg, value = arg.split('=')
            if arg == '--algorithms':
                algorithms = set(self.range_pattern.findall(value)[0].split(','))
                if tuple(self.distr_pattern.findall(value)) not in {(), ('O',)}:
                    raise ValueError(("Distribution options are prohibited for --algorithms flag. "
                                      "Simply list the algorithms instead (like --algorithms=ftrl,sgd)"))
                elif self.distr_pattern.findall(value) == ['O']:
                    algorithms.add('sgd')

                for algo in algorithms:
                    if algo not in self.algorithm_metadata:
                        raise NotImplementedError(("%s algorithm is not found. "
                                                   "Supported algorithms by now are %s")
                                                  % (algo, str(self.algorithm_metadata.keys())))
                break

        self.space = {algo: {'type': algo, 'argument': self.algorithm_metadata[algo]['arg']} for algo in algorithms}
        for algo in algorithms:
            for arg in line:
                arg, value = arg.split('=')
                if arg == '--algorithms':
                    continue
                if arg not in self.algorithm_metadata[algo]['prohibited_flags']:
                    distrib = self._process_vw_argument(arg, value, algo)
                    self.space[algo][arg] = distrib
                else:
                    pass
        self.space = hp.choice('algorithm', self.space.values())

class HyperoptSpaceConstructorCB(HyperoptSpaceConstructor):
    """
    Takes command-line input and transforms it into hyperopt search space
    An example of command-line input:
    --algorithms=first,greedy,softmax --l2=1e-8..1e-4~LO -l=0.01..10~L --ftrl_beta=0.01..1 --passes=1..10~I -q=SE+SZ+DR,SE~O
    """

    def __init__(self, command):  
        self.cb_algorithm_metadata = {
            'first': {'arg': '--first', 'prohibited_flags': {'--mellowness', '--lambda'}},
            'greedy': {'arg': '', 'prohibited_flags': {'--mellowness', '--lambda'}},
            'cover': {'arg': '--cover', 'prohibited_flags': {'--mellowness', '--lambda'}},
            'softmax': {'arg': '--softmax', 'prohibited_flags': {'--mellowness'}},
            'bag': {'arg': '--bag', 'prohibited_flags': {'--mellowness', '--lambda'}},
            'regcb': {'arg': '--regcb', 'prohibited_flags': {'--lambda'}}
        }

        HyperoptSpaceConstructor.__init__(self, command)

    def string_to_pyll(self):
        line = shlex.split(self.command)

        cb_algorithms = ['greedy']
        for arg in line:
            arg, value = arg.split('=')
            if arg == '--cb_algorithms':
                cb_algorithms = set(self.range_pattern.findall(value)[0].split(','))
                if tuple(self.distr_pattern.findall(value)) not in {(), ('O',)}:
                    raise ValueError(("Distribution options are prohibited for --cb_algorithms flag. "
                                      "Simply list the cb_algorithms instead (like --cb_algorithms=ftrl,sgd)"))
                elif self.distr_pattern.findall(value) == ['O']:
                    cb_algorithms.add('greedy')

                for algo in cb_algorithms:
                    if algo not in self.cb_algorithm_metadata:
                        raise NotImplementedError(("%s cb_algorithm is not found. "
                                                   "Supported cb_algorithms by now are %s")
                                                  % (algo, str(self.cb_algorithm_metadata.keys())))
                break

        self.space = {algo: {'type': algo, 'argument': self.cb_algorithm_metadata[algo]['arg']} for algo in cb_algorithms}
        for algo in cb_algorithms:
            for arg in line:
                arg, value = arg.split('=')
                if arg == '--cb_algorithms':
                    continue
                if arg not in self.cb_algorithm_metadata[algo]['prohibited_flags']:
                    distrib = self._process_vw_argument(arg, value, algo)
                    self.space[algo][arg] = distrib
                else:
                    pass
        self.space = hp.choice('cb_algorithm', self.space.values())

class HyperOptimizer(object):
    def __init__(self, train_set, holdout_set, command, max_evals=100,
                 outer_loss_function='logistic',
                 searcher='tpe', is_regression=False, additional_cmd=''):
        self.train_set = train_set
        self.holdout_set = holdout_set

        self.train_model = './current.model'
        self.holdout_pred = './holdout.pred'
        self.trials_output = './trials.json'
        self.hyperopt_progress_plot = './hyperopt_progress.png'
        self.log = './log.log'

        self.logger = self._configure_logger()

        # hyperopt parameter sample, converted into a string with flags
        self.param_suffix = None
        self.train_command = None
        self.validate_command = None
        
        self.max_evals = max_evals
        self.searcher = searcher
        self.additional_cmd = additional_cmd   
        

        self.trials = Trials()
        self.current_trial = 0


    def _configure_logger(self):
        LOGGER_FORMAT = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(name)s/%(module)s:%(lineno)d]: %(message)s"
        LOGGER_DATEFMT = "%Y-%m-%d %H:%M:%S"
        LOGFILE = self.log

        logging.basicConfig(format=LOGGER_FORMAT,
                            datefmt=LOGGER_DATEFMT,
                            level=logging.DEBUG)
        formatter = logging.Formatter(LOGGER_FORMAT, datefmt=LOGGER_DATEFMT)

        file_handler = logging.FileHandler(LOGFILE)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(file_handler)
        return logger

    def get_hyperparam_string(self, **kwargs):
        for arg in ['--passes']: #, '--rank', '--lrq']:
            if arg in kwargs:
                kwargs[arg] = int(kwargs[arg])

        #print 'KWARGS: ', kwargs
        flags = [key for key in kwargs if key.startswith('-')]
        for flag in flags:
            if kwargs[flag] == 'omit':
                del kwargs[flag]

        self.param_suffix = ' '.join(['%s %s' % (key, kwargs[key]) for key in kwargs if key.startswith('-')])
        self.param_suffix += ' %s' % (kwargs['argument'])


    


class HyperOptimizerNotCB(HyperOptimizer):
    def __init__(self, train_set, holdout_set, command, max_evals=100,
                 outer_loss_function='logistic',
                 searcher='tpe', is_regression=False, additional_cmd=''):

        self.y_true_holdout = []
        self.outer_loss_function = outer_loss_function
        self.is_regression = is_regression
        self.labels_clf_count = 0

        self.space = self._get_space(command)

        HyperOptimizer.__init__(self, train_set, holdout_set, command, max_evals=max_evals, searcher=searcher, additional_cmd=additional_cmd)

    def _get_space(self, command):
        hs = HyperoptSpaceConstructorNotCB(command)
        hs.string_to_pyll()
        return hs.space

    def compose_vw_train_command(self):
        data_part = ('vw -d %s -f %s --holdout_off -c %s'
                     % (self.train_set, self.train_model, self.additional_cmd))
        if self.labels_clf_count > 2: # multiclass, should take probabilities
            data_part += ('--oaa %s --loss_function=logistic --probabilities '
                          % (self.labels_clf_count))
        self.train_command = ' '.join([data_part, self.param_suffix])

    def compose_vw_validate_command(self):
        data_part = 'vw -t -d %s -i %s -p %s --holdout_off -c %s' \
                    % (self.holdout_set, self.train_model, self.holdout_pred, self.additional_cmd)
        if self.labels_clf_count > 2: # multiclass
            data_part += ' --loss_function=logistic --probabilities'
        self.validate_command = data_part

    #I will put this function base class
    def fit_vw(self):
        self.compose_vw_train_command()
        self.logger.info("executing the following command (training): %s" % self.train_command)
        subprocess.call(shlex.split(self.train_command))

    #I will put this function base class
    def validate_vw(self):
        self.compose_vw_validate_command()
        self.logger.info("executing the following command (validation): %s" % self.validate_command)
        subprocess.call(shlex.split(self.validate_command))

    def get_y_true_holdout(self):
        self.logger.info("loading true holdout class labels...")
        yh = open(self.holdout_set, 'r')
        self.y_true_holdout = []
        for line in yh:
            self.y_true_holdout.append(float(line.split()[0]))
        if not self.is_regression:
            self.labels_clf_count = len(set(self.y_true_holdout))
            if self.labels_clf_count > 2 and self.outer_loss_function != 'logistic':
                raise KeyError('Only logistic loss function is available for multiclass clf')
            if self.labels_clf_count <= 2:
                self.y_true_holdout = [int((i + 1.) / 2) for i in self.y_true_holdout]
        self.logger.info("holdout length: %d" % len(self.y_true_holdout))

    def get_y_pred_holdout(self):
        y_pred_holdout = []
        with open('%s' % self.holdout_pred, 'r') as v:
            for line in v:
                if self.labels_clf_count > 2:
                    y_pred_holdout.append(list(map(lambda x: float(x.split(':')[1]), line.split())))
                else:
                    y_pred_holdout.append(float(line.split()[0].strip()))
        return y_pred_holdout

    def validation_metric_vw(self):
        y_pred_holdout = self.get_y_pred_holdout()

        if self.outer_loss_function == 'logistic':
            if self.labels_clf_count > 2:
                y_pred_holdout_proba = y_pred_holdout
            else:
                y_pred_holdout_proba = [1. / (1 + exp(-i)) for i in y_pred_holdout]

            loss = log_loss(self.y_true_holdout, y_pred_holdout_proba)

        elif self.outer_loss_function == 'squared':
            loss = mean_squared_error(self.y_true_holdout, y_pred_holdout)

        elif self.outer_loss_function == 'hinge':
            loss = hinge_loss(self.y_true_holdout, y_pred_holdout)

        elif self.outer_loss_function == 'pr-auc':
            loss = -average_precision_score(self.y_true_holdout, y_pred_holdout)

        elif self.outer_loss_function == 'roc-auc':
            y_pred_holdout_proba = [1. / (1 + exp(-i)) for i in y_pred_holdout]
            fpr, tpr, _ = roc_curve(self.y_true_holdout, y_pred_holdout_proba)
            loss = -auc(fpr, tpr)

        elif self.outer_loss_function == 'quantile': # Minimum at Median
            tau = 0.5         
            loss = np.mean([max(tau * (true - pred), (tau - 1) * (true - pred)) \
                          for true, pred in zip(self.y_true_holdout, y_pred_holdout)])

        else:
            raise KeyError('Invalide outer loss function')

        self.logger.info('parameter suffix: %s' % self.param_suffix)
        self.logger.info('loss value: %.6f' % loss)

        return loss

    def hyperopt_search(self, parallel=False):  # TODO: implement parallel search with MongoTrials
        def objective(kwargs):
            start = dt.now()

            self.current_trial += 1
            self.logger.info('\n\nStarting trial no.%d' % self.current_trial)
            self.get_hyperparam_string(**kwargs)
            self.fit_vw()
            self.validate_vw()
            loss = self.validation_metric_vw()

            finish = dt.now()
            elapsed = finish - start
            self.logger.info("evaluation time for this step: %s" % str(elapsed))

            # clean up
            subprocess.call(shlex.split('rm %s %s' % (self.train_model, self.holdout_pred)))

            to_return = {'status': STATUS_OK,
                         'loss': loss,  # TODO: include also train loss tracking in order to prevent overfitting
                         'eval_time': elapsed.seconds,
                         'train_command': self.train_command,
                         'current_trial': self.current_trial
            }
            return to_return

        self.trials = Trials()
        if self.searcher == 'tpe':
            algo = tpe.suggest
        elif self.searcher == 'rand':
            algo = rand.suggest
        else:
            raise KeyError('Invalid searcher')

        logging.debug("starting hypersearch...")
        best_params = fmin(objective, space=self.space, trials=self.trials, algo=algo, max_evals=self.max_evals)
        self.logger.debug("the best hyperopt parameters: %s" % str(best_params))

        json.dump(self.trials.results, open(self.trials_output, 'w'))
        self.logger.info('All the trials results are saved at %s' % self.trials_output)

        best_configuration = self.trials.results[np.argmin(self.trials.losses())]['train_command']
        best_loss = self.trials.results[np.argmin(self.trials.losses())]['loss']
        self.logger.info("\n\nA full training command with the best hyperparameters: \n%s\n\n" % best_configuration)
        self.logger.info("\n\nThe best holdout loss value: \n%s\n\n" % best_loss)

        return best_configuration, best_loss

    def plot_progress(self):
        try:
            sns.set_palette('Set2')
            sns.set_style("darkgrid", {"axes.facecolor": ".95"})
        except:
            pass

        self.logger.debug('plotting...')
        plt.figure(figsize=(15,10))
        plt.subplot(211)
        plt.plot(self.trials.losses(), '.', markersize=12)
        plt.title('Per-Iteration Outer Loss', fontsize=16)
        plt.ylabel('Outer loss function value')
        if self.outer_loss_function in ['logloss']:
            plt.yscale('log')
        xticks = [int(i) for i in np.linspace(plt.xlim()[0], plt.xlim()[1], min(len(self.trials.losses()), 11))]
        plt.xticks(xticks, xticks)


        plt.subplot(212)
        plt.plot(np.minimum.accumulate(self.trials.losses()), '.', markersize=12)
        plt.title('Cumulative Minimum Outer Loss', fontsize=16)
        plt.xlabel('Iteration number')
        plt.ylabel('Outer loss function value')
        xticks = [int(i) for i in np.linspace(plt.xlim()[0], plt.xlim()[1], min(len(self.trials.losses()), 11))]
        plt.xticks(xticks, xticks)

        plt.tight_layout()
        plt.savefig(self.hyperopt_progress_plot)
        self.logger.info('The diagnostic hyperopt progress plot is saved: %s' % self.hyperopt_progress_plot)

class HyperOptimizerCB(HyperOptimizer):
    def __init__(self, train_set, holdout_set, command, max_evals=100,
                 searcher='tpe', additional_cmd=''):

    	self.space = self._get_space(command)

    	HyperOptimizer.__init__(self, train_set, holdout_set, command, max_evals=max_evals, searcher=searcher, additional_cmd=additional_cmd)

    def _get_space(self, command):
        hs = HyperoptSpaceConstructorNotCB(command)
        hs.string_to_pyll()
        return hs.space

    '''
    Run validation and train commamd for multiple seeds so that we could get more generalise setting
    def get_seed(self):
        return  np.random.randint(100)
    '''

    def compose_vw_train_command(self):
        data_part = ('vw -d %s --cb_explore_adf -f %s --random_seed %d -c %s' \
                     % (self.train_set, self.train_model, 13, self.additional_cmd))  # Use get_seed function
        self.train_command = ' '.join([data_part, self.param_suffix])

    def compose_vw_validate_command(self):
        data_part = 'vw -t -d %s -i %s -p %s --random_seed %d -c %s' \
                    % (self.holdout_set, self.train_model, self.holdout_pred, 33, self.additional_cmd) # Use get_seed function
        self.validate_command = data_part

    #I will put this function base class
    def fit_vw(self):
        self.compose_vw_train_command()
        self.logger.info("executing the following command (training): %s" % self.train_command)
        subprocess.call(shlex.split(self.train_command))

    #I will put this function base class
    def validate_vw(self):
        self.compose_vw_validate_command()
        self.logger.info("executing the following command (validation): %s" % self.validate_command)
        subprocess.call(shlex.split(self.validate_command))
    

    def validation_metric_vw(self):
        '''
        Loss obtained from validations?
        TODO
        '''
        pass

    def hyperopt_search(self, parallel=False):  # TODO: implement parallel search with MongoTrials
        def objective(kwargs):
            start = dt.now()

            self.current_trial += 1
            self.logger.info('\n\nStarting trial no.%d' % self.current_trial)
            self.get_hyperparam_string(**kwargs)
            self.fit_vw()
            self.validate_vw()
            loss = self.validation_metric_vw()  #TODO

            finish = dt.now()
            elapsed = finish - start
            self.logger.info("evaluation time for this step: %s" % str(elapsed))

            # clean up
            subprocess.call(shlex.split('rm %s %s' % (self.train_model, self.holdout_pred)))

            to_return = {'status': STATUS_OK,
                         'loss': loss,  # TODO: include also train loss tracking in order to prevent overfitting
                         'eval_time': elapsed.seconds,
                         'train_command': self.train_command,
                         'current_trial': self.current_trial
            }
            return to_return

        self.trials = Trials()
        if self.searcher == 'tpe':
            algo = tpe.suggest
        elif self.searcher == 'rand':
            algo = rand.suggest
        else:
            raise KeyError('Invalid searcher')

        logging.debug("starting hypersearch...")
        best_params = fmin(objective, space=self.space, trials=self.trials, algo=algo, max_evals=self.max_evals)
        self.logger.debug("the best hyperopt parameters: %s" % str(best_params))

        json.dump(self.trials.results, open(self.trials_output, 'w'))
        self.logger.info('All the trials results are saved at %s' % self.trials_output)

        best_configuration = self.trials.results[np.argmin(self.trials.losses())]['train_command']
        best_loss = self.trials.results[np.argmin(self.trials.losses())]['loss']
        self.logger.info("\n\nA full training command with the best hyperparameters: \n%s\n\n" % best_configuration)
        self.logger.info("\n\nThe best holdout loss value: \n%s\n\n" % best_loss)

        return best_configuration, best_loss

    def plot_progress(self):

        """
        Same as for non CB algorithms, additionally it will show variations
        for test_seeds. 
        
        """
        pass     
    


def main():
    args = read_arguments()
    if args.is_cb:
        h = HyperOptimizerCB(train_set=args.train, holdout_set=args.holdout, command=args.vw_space,
                        max_evals=args.max_evals,
                        additional_cmd=args.additional_cmd,
                        searcher=args.searcher)
    else:
        h = HyperOptimizerNotCB(train_set=args.train, holdout_set=args.holdout, command=args.vw_space,
                        max_evals=args.max_evals,
                        outer_loss_function=args.outer_loss_function,
                        additional_cmd=args.additional_cmd,
                        searcher=args.searcher, is_regression=args.regression)
        h.get_y_true_holdout()

    h.hyperopt_search()
    if args.plot:
        h.plot_progress()


if __name__ == '__main__':
    main()

