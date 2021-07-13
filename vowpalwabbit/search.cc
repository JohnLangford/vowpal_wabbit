// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <float.h>
#include <string.h>
#include <math.h>
#include <memory>
#include <algorithm>
#include "vw.h"
#include "rand48.h"
#include "reductions.h"
#include "gd.h"  // for GD::foreach_feature
#include "parse_primitives.h"
#include "search_sequencetask.h"
#include "search_multiclasstask.h"
#include "search_dep_parser.h"
#include "search_entityrelationtask.h"
#include "search_hooktask.h"
#include "search_graph.h"
#include "search_meta.h"
#include "csoaa.h"
#include "active.h"
#include "label_dictionary.h"
#include "vw_exception.h"
#include "shared_data.h"

#include "io/logger.h"
// needed for printing ranges of objects (eg: all elements of a vector)
#include <fmt/ranges.h>

using namespace VW::LEARNER;
using namespace VW::config;
namespace CS = COST_SENSITIVE;
namespace MC = MULTICLASS;

namespace logger = VW::io::logger;

using std::endl;

namespace Search
{
using byte_array = std::unique_ptr<uint8_t[]>;

search_task* all_tasks[] = {&SequenceTask::task, &SequenceSpanTask::task, &SequenceTaskCostToGo::task,
    &ArgmaxTask::task, &SequenceTask_DemoLDF::task, &MulticlassTask::task, &DepParserTask::task,
    &EntityRelationTask::task, &HookTask::task, &GraphTask::task, nullptr};  // must nullptr terminate!

search_metatask* all_metatasks[] = {
    &DebugMT::metatask, &SelectiveBranchingMT::metatask, nullptr};  // must nullptr terminate!

constexpr bool PRINT_UPDATE_EVERY_EXAMPLE = false;
constexpr bool PRINT_UPDATE_EVERY_PASS = false;
constexpr bool PRINT_CLOCK_TIME = false;
constexpr uint64_t SEARCH_HASH_SEED = 3419;

std::string neighbor_feature_space("neighbor");
std::string condition_feature_space("search_condition");

uint32_t AUTO_CONDITION_FEATURES = 1, AUTO_HAMMING_LOSS = 2, EXAMPLES_DONT_CHANGE = 4, IS_LDF = 8, NO_CACHING = 16,
         ACTION_COSTS = 32;
enum SearchState
{
  INITIALIZE,
  INIT_TEST,
  INIT_TRAIN,
  LEARN,
  GET_TRUTH_STRING
};
enum RollMethod
{
  POLICY,
  ORACLE,
  MIX_PER_STATE,
  MIX_PER_ROLL,
  NO_ROLLOUT
};

// a data structure to hold conditioning information
struct prediction
{
  ptag me;        // the id of the current prediction (the one being memoized)
  size_t cnt;     // how many variables are we conditioning on?
  ptag* tags;     // which variables are they?
  action* acts;   // and which actions were taken at each?
  uint32_t hash;  // a hash of the above
};

// parameters for auto-conditioning
struct auto_condition_settings
{
  size_t max_bias_ngram_length;  // add a "bias" feature for each ngram up to and including this length. eg., if it's 1,
                                 // then you get a single feature for each conditional
  size_t max_quad_ngram_length;  // add bias *times* input features for each ngram up to and including this length
  float feature_value;           // how much weight should the conditional features get?
  bool use_passthrough_repr;     // should we ask lower-level reductions for their internal state?
};

struct scored_action
{
  action a;  // the action
  float s;   // the predicted cost of this action
  // v_array<feature> repr;
  scored_action(action _a = static_cast<action>(-1), float _s = 0) : a(_a), s(_s) {}
  // scored_action(action _a, float _s, v_array<feature>& _repr) : a(_a), s(_s), repr(_repr) {}
  // scored_action() { a = (action)-1; s = 0.; }
};
std::ostream& operator<<(std::ostream& os, const scored_action& x)
{
  os << x.a << ':' << x.s;
  return os;
}

struct action_repr
{
  action a = 0;
  features* repr = nullptr;
  action_repr() = default;
  action_repr(action _a, features* _repr) : a(_a)
  {
    if (_repr != nullptr) { repr = new features(*_repr); }
  }
  action_repr(action _a) : a(_a), repr(nullptr) {}
};

struct action_cache
{
  float min_cost;
  action k;
  bool is_opt;
  float cost;
  action_cache(float _min_cost, action _k, bool _is_opt, float _cost)
      : min_cost(_min_cost), k(_k), is_opt(_is_opt), cost(_cost)
  {
  }
};
std::ostream& operator<<(std::ostream& os, const action_cache& x)
{
  os << x.k << ':' << x.cost;
  if (x.is_opt) os << '*';
  return os;
}

void clear_memo_foreach_action(search_private& priv);

struct search_private
{
private:
  struct cached_item_equivalent
  {
    bool operator()(const byte_array& A, const byte_array& B) const
    {
      size_t sz_A = *A.get();
      size_t sz_B = *B.get();
      if (sz_A != sz_B) return false;
      return memcmp(A.get(), B.get(), sz_A) == 0;
    }
  };
  struct cached_item_hash
  {
    size_t operator()(const byte_array& key) const
    {
      size_t sz = *key.get();
      return uniform_hash(key.get(), sz, SEARCH_HASH_SEED);
    }
  };

public:
  using cache_map = std::unordered_map<byte_array, scored_action, cached_item_hash, cached_item_equivalent>;

  vw* all;
  std::shared_ptr<rand_state> _random_state;

  uint64_t offset;
  bool auto_condition_features;  // do you want us to automatically add conditioning features?
  bool auto_hamming_loss;        // if you're just optimizing hamming loss, we can do it for you!
  bool examples_dont_change;     // set to true if you don't do any internal example munging
  bool is_ldf;                   // user declared ldf
  bool use_action_costs;         // task promises to define per-action rollout-by-ref costs

  v_array<int32_t> neighbor_features;  // ugly encoding of neighbor feature requirements
  auto_condition_settings acset;       // settings for auto-conditioning
  size_t history_length;               // value of --search_history_length, used by some tasks, default 1

  size_t A;                 // total number of actions, [1..A]; 0 means ldf
  size_t num_learners;      // total number of learners;
  bool cb_learner;          // do contextual bandit learning on action (was "! rollout_all_actions" which was confusing)
  SearchState state;        // current state of learning
  size_t learn_learner_id;  // we allow user to use different learners for different states
  int mix_per_roll_policy;  // for MIX_PER_ROLL, we need to choose a policy to use; this is where it's stored (-2 means
                            // "not selected yet")
  bool no_caching;          // turn off caching
  size_t rollout_num_steps;  // how many calls of "loss" before we stop really predicting on rollouts and switch to
                             // oracle (0 means "infinite")
  bool linear_ordering;      // insist that examples are generated in linear order (rather that the default hoopla
                             // permutation)
  bool (*label_is_test)(polylabel&);  // tell me if the label data from an example is test

  size_t t;                                     // current search step
  size_t T;                                     // length of root trajectory
  std::vector<example> learn_ec_copy;           // copy of example(s) at learn_t
  example* learn_ec_ref;                        // reference to example at learn_t, when there's no example munging
  size_t learn_ec_ref_cnt;                      // how many are there (for LDF mode only; otherwise 1)
  v_array<ptag> learn_condition_on;             // a copy of the tags used for conditioning at the training position
  v_array<action_repr> learn_condition_on_act;  // the actions taken
  v_array<char> learn_condition_on_names;       // the names of the actions
  v_array<action> learn_allowed_actions;        // which actions were allowed at training time?
  v_array<action_repr> ptag_to_action;          // tag to action mapping for conditioning
  std::vector<action> test_action_sequence;  // if test-mode was run, what was the corresponding action sequence; it's a
                                             // vector cuz we might expose it to the library
  action learn_oracle_action;                // store an oracle action for debugging purposes
  features last_action_repr;

  polylabel allowed_actions_cache;

  size_t loss_declared_cnt;                 // how many times did run declare any loss (implicitly or explicitly)?
  v_array<scored_action> train_trajectory;  // the training trajectory
  size_t learn_t;                           // what time step are we learning on?
  size_t learn_a_idx;                       // what action index are we trying?
  bool done_with_all_actions;               // set to true when there are no more learn_a_idx to go

  float test_loss;   // loss incurred when run INIT_TEST
  float learn_loss;  // loss incurred when run LEARN
  float train_loss;  // loss incurred when run INIT_TRAIN

  bool hit_new_pass;     // have we hit a new pass?
  bool force_oracle;     // insist on using the oracle to make predictions
  float perturb_oracle;  // with this probability, choose a random action instead of oracle action

  size_t num_calls_to_run, num_calls_to_run_previous, save_every_k_runs;

  // if we're printing to stderr we need to remember if we've printed the header yet
  // (i.e., we do this if we're driving)
  bool printed_output_header;

  // various strings for different search states
  bool should_produce_string;
  std::unique_ptr<std::stringstream> pred_string;
  std::unique_ptr<std::stringstream> truth_string;
  std::unique_ptr<std::stringstream> bad_string_stream;

  // parameters controlling interpolation
  float beta;   // interpolation rate
  float alpha;  // parameter used to adapt beta for dagger (see above comment), should be in (0,1)

  RollMethod rollout_method;
  RollMethod rollin_method;
  float subsample_timesteps;  // train at every time step or just a (random) subset?
  bool xv;  // train three separate policies -- two for providing examples to the other and a third training on the
            // union (which will be used at test time -- TODO)

  bool allow_current_policy;  // should the current policy be used for training? true for dagger
  bool adaptive_beta;  // used to implement dagger-like algorithms. if true, beta = 1-(1-alpha)^n after n updates, and
                       // policy is mixed with oracle as \pi' = (1-beta)\pi^* + beta \pi
  size_t passes_per_policy;  // if we're not in dagger-mode, then we need to know how many passes to train a policy

  uint32_t current_policy;  // what policy are we training right now?

  // various statistics for reporting
  size_t num_features;
  uint32_t total_number_of_policies;
  size_t read_example_last_id;
  size_t passes_since_new_policy;
  size_t read_example_last_pass;
  size_t total_examples_generated;
  size_t total_predictions_made;
  size_t total_cache_hits;

  cache_map cache_hash_map;

  // for foreach_feature temporary storage for conditioning
  uint64_t dat_new_feature_idx;
  example* dat_new_feature_ec;
  std::stringstream dat_new_feature_audit_ss;
  features dat_new_feature_namespace;
  std::string* dat_new_feature_feature_space;
  float dat_new_feature_value;

  // to reduce memory allocation
  std::unique_ptr<std::stringstream> rawOutputStringStream;
  CS::label ldf_test_label;
  v_array<action_repr> condition_on_actions;
  v_array<size_t> timesteps;
  polylabel learn_losses;
  polylabel gte_label;
  std::vector<std::pair<float, size_t>> active_uncertainty;
  std::vector<std::vector<std::pair<CS::wclass&, bool>>> active_known;
  bool force_setup_ec_ref;
  bool active_csoaa;
  float active_csoaa_verify;

  VW::LEARNER::base_learner* base_learner;
  clock_t start_clock_time;

  CS::label empty_cs_label;

  search_task* task;          // your task!
  search_metatask* metatask;  // your (optional) metatask
  BaseTask* metaoverride;
  size_t meta_t;  // the metatask has it's own notion of time. meta_t+t, during a single run, is the way to think about
                  // the "real" decision step but this really only matters for caching purposes
  v_array<v_array<action_cache>*>
      memo_foreach_action;  // when foreach_action is on, we need to cache TRAIN trajectory actions for LEARN

  ~search_private()
  {
    if (all)
    {
      for (auto& ar : ptag_to_action) delete ar.repr;
      clear_memo_foreach_action(*this);
    }
  }
};

void clear_memo_foreach_action(search_private& priv)
{
  for (size_t i = 0; i < priv.memo_foreach_action.size(); i++)
    if (priv.memo_foreach_action[i])
    {
      delete priv.memo_foreach_action[i];
    }
  priv.memo_foreach_action.clear();
}

search::search()
{
  priv = &calloc_or_throw<search_private>();
  new (priv) search_private();
}

search::~search()
{
  if (this->priv)
  {
    this->priv->~search_private();
    free(this->priv);
  }
}

std::string audit_feature_space("conditional");
uint64_t conditional_constant = 8290743;

inline bool need_memo_foreach_action(search_private& priv)
{
  return (priv.state == INIT_TRAIN) && (priv.metatask) && (priv.metaoverride);  // &&
  //        (priv.metaoverride->_foreach_action || priv.metaoverride->_post_prediction);
}

int random_policy(search_private& priv, bool allow_current, bool allow_optimal, bool advance_prng = true)
{
  if (priv.beta >= 1)
  {
    if (allow_current) return static_cast<int>(priv.current_policy);
    if (priv.current_policy > 0) return ((static_cast<int>(priv.current_policy)) - 1);
    if (allow_optimal) return -1;
    logger::errlog_error("internal error (bug): no valid policies to choose from!  defaulting to current");
    return static_cast<int>(priv.current_policy);
  }

  int num_valid_policies = static_cast<int>(priv.current_policy) + allow_optimal + allow_current;
  int pid = -1;

  if (num_valid_policies == 0)
  {
    logger::errlog_error("internal error (bug): no valid policies to choose from!  defaulting to current");
    return static_cast<int>(priv.current_policy);
  }
  else if (num_valid_policies == 1)
    pid = 0;
  else if (num_valid_policies == 2)
    pid = (advance_prng ? priv._random_state->get_and_update_random() : priv._random_state->get_random()) >= priv.beta;
  else
  {
    // SPEEDUP this up in the case that beta is small!
    float r = (advance_prng ? priv._random_state->get_and_update_random() : priv._random_state->get_random());
    pid = 0;

    if (r > priv.beta)
    {
      r -= priv.beta;
      while ((r > 0) && (pid < num_valid_policies - 1))
      {
        pid++;
        r -= priv.beta * powf(1.f - priv.beta, static_cast<float>(pid));
      }
    }
  }
  // figure out which policy pid refers to
  if (allow_optimal && (pid == num_valid_policies - 1)) return -1;  // this is the optimal policy

  pid = static_cast<int>(priv.current_policy) - pid;
  if (!allow_current) pid--;

  return pid;
}

// for two-fold cross validation, we double the number of learners
// and send examples to one or the other depending on the xor of
// (is_training) and (example_id % 2)
int select_learner(search_private& priv, int policy, size_t learner_id, bool is_training, bool is_local)
{
  if (policy < 0)
    return policy;  // optimal policy
  else
  {
    if (priv.xv)
    {
      learner_id *= 3;
      if (!is_local) learner_id += 1 + static_cast<size_t>(is_training ^ (priv.all->sd->example_number % 2 == 1));
    }
    int p = static_cast<int>(policy * priv.num_learners + learner_id);
    return p;
  }
}

bool should_print_update(vw& all, bool hit_new_pass = false)
{
  // uncomment to print out final loss after all examples processed
  // commented for now so that outputs matches make test

  if (PRINT_UPDATE_EVERY_EXAMPLE) return true;
  if (PRINT_UPDATE_EVERY_PASS)
    if (hit_new_pass) return true;
  return (all.sd->weighted_examples() >= all.sd->dump_interval) && !all.logger.quiet && !all.bfgs;
}

bool might_print_update(vw& all)
{
  // basically do should_print_update but check me and the next
  // example because of off-by-ones

  if (PRINT_UPDATE_EVERY_EXAMPLE) return true;
  if (PRINT_UPDATE_EVERY_PASS) return true;  // SPEEDUP: make this better
  return (all.sd->weighted_examples() + 1. >= all.sd->dump_interval) && !all.logger.quiet && !all.bfgs;
}

bool must_run_test(vw& all, multi_ex& ec, bool is_test_ex)
{
  return (all.final_prediction_sink.size() > 0) ||  // if we have to produce output, we need to run this
      might_print_update(all) ||                    // if we have to print and update to stderr
      (all.raw_prediction != nullptr) ||            // we need raw predictions
      ((!all.vw_is_main) && (is_test_ex)) ||        // library needs predictions
      // or:
      //   it's not quiet AND
      //     current_pass == 0
      //     OR holdout is off
      //     OR it's a test example
      ((!all.logger.quiet || !all.vw_is_main) &&  // had to disable this because of library mode!
          (!is_test_ex) &&
          (all.holdout_set_off ||                          // no holdout
              ec[0]->test_only || (all.current_pass == 0)  // we need error rates for progressive cost
              ));
}

float safediv(float a, float b)
{
  if (b == 0.f)
    return 0.f;
  else
    return (a / b);
}

void to_short_string(std::string in, size_t max_len, char* out)
{
  for (size_t i = 0; i < max_len; i++)
    out[i] = ((i >= in.length()) || (in[i] == '\n') || (in[i] == '\t')) ? ' ' : in[i];

  if (in.length() > max_len)
  {
    out[max_len - 2] = '.';
    out[max_len - 1] = '.';
  }
  out[max_len] = 0;
}

std::string number_to_natural(size_t big)
{
  std::stringstream ss;
  if (big > 9999999999)
    ss << big / 1000000000 << "g";
  else if (big > 9999999)
    ss << big / 1000000 << "m";
  else if (big > 9999)
    ss << big / 1000 << "k";
  else
    ss << big;

  return ss.str();
}

void print_update(search_private& priv)
{
  // TODO: This function should be outputting to trace_message(?), but is mixing ostream and printf formats
  //       Currently there is no way to convert an ostream to FILE*, so the lines will need to be converted
  //       to ostream format
  vw& all = *priv.all;
  if (!priv.printed_output_header && !all.logger.quiet)
  {
    const char* header_fmt = "%-10s %-10s %8s%24s %22s %5s %5s  %7s  %7s  %7s  %-8s\n";
    fprintf(stderr, header_fmt, "average", "since", "instance", "current true", "current predicted", "cur", "cur",
        "predic", "cache", "examples", "");
    if (priv.active_csoaa)
      fprintf(stderr, header_fmt, "loss", "last", "counter", "output prefix", "output prefix", "pass", "pol", "made",
          "hits", "gener", "#run");
    else
      fprintf(stderr, header_fmt, "loss", "last", "counter", "output prefix", "output prefix", "pass", "pol", "made",
          "hits", "gener", "beta");
    std::cerr.precision(5);
    priv.printed_output_header = true;
  }

  if (!should_print_update(all, priv.hit_new_pass)) return;

  char true_label[21];
  char pred_label[21];
  to_short_string(priv.truth_string->str(), 20, true_label);
  to_short_string(priv.pred_string->str(), 20, pred_label);

  float avg_loss = 0.;
  float avg_loss_since = 0.;
  bool use_heldout_loss = (!all.holdout_set_off && all.current_pass >= 1) && (all.sd->weighted_holdout_examples > 0);
  if (use_heldout_loss)
  {
    avg_loss =
        safediv(static_cast<float>(all.sd->holdout_sum_loss), static_cast<float>(all.sd->weighted_holdout_examples));
    avg_loss_since = safediv(static_cast<float>(all.sd->holdout_sum_loss_since_last_dump),
        static_cast<float>(all.sd->weighted_holdout_examples_since_last_dump));

    all.sd->weighted_holdout_examples_since_last_dump = 0;
    all.sd->holdout_sum_loss_since_last_dump = 0.0;
  }
  else
  {
    avg_loss = safediv(static_cast<float>(all.sd->sum_loss), static_cast<float>(all.sd->weighted_labeled_examples));
    avg_loss_since = safediv(static_cast<float>(all.sd->sum_loss_since_last_dump),
        static_cast<float>(all.sd->weighted_labeled_examples - all.sd->old_weighted_labeled_examples));
  }

  auto const& inst_cntr = number_to_natural(static_cast<size_t>(all.sd->example_number));
  auto const& total_pred = number_to_natural(priv.total_predictions_made);
  auto const& total_cach = number_to_natural(priv.total_cache_hits);
  auto const& total_exge = number_to_natural(priv.total_examples_generated);

  fprintf(stderr, "%-10.6f %-10.6f %8s  [%s] [%s] %5d %5d  %7s  %7s  %7s  %-8f", avg_loss, avg_loss_since,
      inst_cntr.c_str(), true_label, pred_label, static_cast<int>(priv.read_example_last_pass),
      static_cast<int>(priv.current_policy), total_pred.c_str(), total_cach.c_str(), total_exge.c_str(),
      priv.active_csoaa ? priv.num_calls_to_run : priv.beta);

  if (PRINT_CLOCK_TIME)
  {
    size_t num_sec = static_cast<size_t>((static_cast<float>(clock() - priv.start_clock_time)) / CLOCKS_PER_SEC);
    std::cerr << " " << num_sec << "sec";
  }

  if (use_heldout_loss) fprintf(stderr, " h");

  fprintf(stderr, "\n");
  fflush(stderr);
  all.sd->update_dump_interval(all.progress_add, all.progress_arg);
}

void add_new_feature(search_private& priv, float val, uint64_t idx)
{
  uint64_t mask = priv.all->weights.mask();
  size_t ss = priv.all->weights.stride_shift();

  uint64_t idx2 = ((idx & mask) >> ss) & mask;
  features& fs = priv.dat_new_feature_namespace;
  fs.push_back(val * priv.dat_new_feature_value, ((priv.dat_new_feature_idx + idx2) << ss));
  cdbg << "adding: " << fs.indicies.back() << ':' << fs.values.back() << endl;
  if (priv.all->audit)
  {
    std::stringstream temp;
    temp << "fid=" << ((idx & mask) >> ss) << "_" << priv.dat_new_feature_audit_ss.str();
    fs.space_names.push_back(audit_strings(*priv.dat_new_feature_feature_space, temp.str()));
  }
}

void del_features_in_top_namespace(search_private& /* priv */, example& ec, size_t ns)
{
  if (std::find(ec.feature_space.indices().begin(), ec.feature_space.indices().end(), ns) ==
      ec.feature_space.indices().end())
  {
    return;
    // if (ec.indices.size() == 0)
    //{ THROW("internal error (bug): expecting top namespace to be '" << ns << "' but it was empty"); }
    // else
    //{ THROW("internal error (bug): expecting top namespace to be '" << ns << "' but it was " <<
    //(size_t)ec.indices.last()); }
  }
  features& fs = ec.feature_space.get(ns, ns);
  ec.num_features -= fs.size();
  ec.reset_total_sum_feat_sq();
  ec.feature_space.remove(ns, ns);
}

void add_neighbor_features(search_private& priv, multi_ex& ec_seq)
{
  if (priv.neighbor_features.size() == 0) return;

  uint32_t stride_shift = priv.all->weights.stride_shift();
  for (size_t n = 0; n < ec_seq.size(); n++)  // iterate over every example in the sequence
  {
    example& me = *ec_seq[n];
    for (size_t n_id = 0; n_id < priv.neighbor_features.size(); n_id++)
    {
      int32_t offset = priv.neighbor_features[n_id] >> 24;
      size_t ns = priv.neighbor_features[n_id] & 0xFF;

      priv.dat_new_feature_ec = &me;
      priv.dat_new_feature_value = 1.;
      priv.dat_new_feature_idx = priv.neighbor_features[n_id] * 13748127;
      if (priv.all->audit)
      {
        priv.dat_new_feature_feature_space = &neighbor_feature_space;
        priv.dat_new_feature_audit_ss.str("");
        priv.dat_new_feature_audit_ss << '@' << ((offset > 0) ? '+' : '-') << static_cast<char>(abs(offset) + '0');
        if (ns != ' ') priv.dat_new_feature_audit_ss << static_cast<char>(ns);
      }

      if ((offset < 0) && (n < static_cast<uint64_t>(-offset)))  // add <s> feature
        add_new_feature(priv, 1., static_cast<uint64_t>(925871901) << stride_shift);
      else if (n + offset >= ec_seq.size())  // add </s> feature
        add_new_feature(priv, 1., static_cast<uint64_t>(3824917) << stride_shift);
      else  // this is actually a neighbor
      {
        example& other = *ec_seq[n + offset];
        for (const auto& ns_fs : other.feature_space.get_list(ns))
        { GD::foreach_feature<search_private, add_new_feature>(priv.all, ns_fs.feats, priv, me.ft_offset); }
      }
    }

    if (!priv.dat_new_feature_namespace.empty())
    {
      me.feature_space.get_or_create(neighbor_namespace, neighbor_namespace) =
          std::move(priv.dat_new_feature_namespace);
      priv.dat_new_feature_namespace = features{};
      auto& fs = me.feature_space.get(neighbor_namespace, neighbor_namespace);
      size_t sz = fs.size();
      if ((sz > 0) && (fs.sum_feat_sq > 0.))
      {
        me.reset_total_sum_feat_sq();
        me.num_features += sz;
      }
    }
  }
}

void del_neighbor_features(search_private& priv, multi_ex& ec_seq)
{
  if (priv.neighbor_features.size() == 0) return;
  for (size_t n = 0; n < ec_seq.size(); n++) del_features_in_top_namespace(priv, *ec_seq[n], neighbor_namespace);
}

void reset_search_structure(search_private& priv)
{
  // NOTE: make sure do NOT reset priv.learn_a_idx
  priv.t = 0;
  priv.meta_t = 0;
  priv.loss_declared_cnt = 0;
  priv.done_with_all_actions = false;
  priv.test_loss = 0.;
  priv.learn_loss = 0.;
  priv.train_loss = 0.;
  priv.num_features = 0;
  priv.should_produce_string = false;
  priv.mix_per_roll_policy = -2;
  priv.force_setup_ec_ref = false;
  if (priv.adaptive_beta)
  {
    float x = -log1pf(-priv.alpha) * static_cast<float>(priv.total_examples_generated);
    static constexpr float log_of_2 = static_cast<float>(0.6931471805599453);
    priv.beta = (x <= log_of_2) ? -expm1f(-x) : (1 - expf(-x));  // numerical stability
    // float priv_beta = 1.f - powf(1.f - priv.alpha, (float)priv.total_examples_generated);
    // assert( fabs(priv_beta - priv.beta) < 1e-2 );
    if (priv.beta > 1) priv.beta = 1;
  }

  for (auto& ar : priv.ptag_to_action) delete ar.repr;
  priv.ptag_to_action.clear();

  if (!priv.cb_learner)  // was: if rollout_all_actions
  {
    priv._random_state->set_random_state(
        static_cast<uint32_t>(priv.read_example_last_id * 147483 + 4831921) * 2147483647);
  }
}

void search_declare_loss(search_private& priv, float loss)
{
  priv.loss_declared_cnt++;
  switch (priv.state)
  {
    case INIT_TEST:
      priv.test_loss += loss;
      break;
    case INIT_TRAIN:
      priv.train_loss += loss;
      break;
    case LEARN:
      if ((priv.rollout_num_steps == 0) || (priv.loss_declared_cnt <= priv.rollout_num_steps))
      {
        priv.learn_loss += loss;
        cdbg << "priv.learn_loss += " << loss << " (now = " << priv.learn_loss << ")" << endl;
      }
      break;
    default:
      break;  // get rid of the warning about missing cases (danger!)
  }
}

template <class T>
void cdbg_print_array(std::string str, v_array<T>& A)
{
  cdbg << str << " = [";
  for (size_t i = 0; i < A.size(); i++) cdbg << " " << A[i];
  cdbg << " ]" << endl;
}
template <class T>
void cerr_print_array(std::string str, v_array<T>& A)
{
  logger::errlog_info("{0} = [{1}]", str, fmt::join(A, " "));
}

size_t random(std::shared_ptr<rand_state>& rs, size_t max)
{
  return static_cast<size_t>(rs->get_and_update_random() * static_cast<float>(max));
}
template <class T>
bool array_contains(T target, const T* A, size_t n)
{
  if (A == nullptr) return false;
  for (size_t i = 0; i < n; i++)
    if (A[i] == target) return true;
  return false;
}

// priv.learn_condition_on_act or priv.condition_on_actions
void add_example_conditioning(search_private& priv, example& ec, size_t condition_on_cnt,
    const char* condition_on_names, action_repr* condition_on_actions)
{
  if (condition_on_cnt == 0) return;

  uint64_t extra_offset = 0;
  if (priv.is_ldf)
    if (ec.l.cs.costs.size() > 0) extra_offset = 3849017 * ec.l.cs.costs[0].class_index;

  size_t I = condition_on_cnt;
  size_t N = std::max(priv.acset.max_bias_ngram_length, priv.acset.max_quad_ngram_length);
  for (size_t i = 0; i < I; i++)  // position in conditioning
  {
    uint64_t fid = 71933 + 8491087 * extra_offset;
    if (priv.all->audit)
    {
      priv.dat_new_feature_audit_ss.str("");
      priv.dat_new_feature_audit_ss.clear();
      priv.dat_new_feature_feature_space = &condition_feature_space;
    }

    for (size_t n = 0; n < N; n++)  // length of ngram
    {
      if (i + n >= I) break;  // no more ngrams
      // we're going to add features for the ngram condition_on_actions[i .. i+N]
      uint64_t name = condition_on_names[i + n];
      fid = fid * 328901 + 71933 * ((condition_on_actions[i + n].a + 349101) * (name + 38490137));

      priv.dat_new_feature_ec = &ec;
      priv.dat_new_feature_idx = fid * quadratic_constant;
      priv.dat_new_feature_value = priv.acset.feature_value;

      if (priv.all->audit)
      {
        if (n > 0) priv.dat_new_feature_audit_ss << ',';
        if ((33 <= name) && (name <= 126))
          priv.dat_new_feature_audit_ss << name;
        else
          priv.dat_new_feature_audit_ss << '#' << static_cast<int>(name);
        priv.dat_new_feature_audit_ss << '=' << condition_on_actions[i + n].a;
      }

      // add the single bias feature
      if (n < priv.acset.max_bias_ngram_length)
        add_new_feature(priv, 1., static_cast<uint64_t>(4398201) << priv.all->weights.stride_shift());
      // add the quadratic features
      if (n < priv.acset.max_quad_ngram_length)
        GD::foreach_feature<search_private, uint64_t, add_new_feature>(*priv.all, ec, priv);
    }
  }

  if (priv.acset.use_passthrough_repr)
  {
    cdbg << "BEGIN adding passthrough features" << endl;
    for (size_t i = 0; i < I; i++)
    {
      if (condition_on_actions[i].repr == nullptr) continue;
      features& fs = *(condition_on_actions[i].repr);
      char name = condition_on_names[i];
      for (size_t k = 0; k < fs.size(); k++)
        if ((fs.values[k] > 1e-10) || (fs.values[k] < -1e-10))
        {
          uint64_t fid = 84913 + 48371803 * (extra_offset + 8392817 * name) + 840137 * (4891 + fs.indicies[k]);
          if (priv.all->audit)
          {
            priv.dat_new_feature_audit_ss.str("");
            priv.dat_new_feature_audit_ss.clear();
            priv.dat_new_feature_audit_ss << "passthrough_repr_" << i << '_' << k;
          }

          priv.dat_new_feature_ec = &ec;
          priv.dat_new_feature_idx = fid;
          priv.dat_new_feature_value = fs.values[k];
          add_new_feature(priv, 1., static_cast<uint64_t>(4398201) << priv.all->weights.stride_shift());
        }
    }
    cdbg << "END adding passthrough features" << endl;
  }

  if (!priv.dat_new_feature_namespace.empty())
  {
    ec.feature_space.get_or_create(conditioning_namespace, conditioning_namespace) =
        std::move(priv.dat_new_feature_namespace);
    priv.dat_new_feature_namespace = features{};
    auto& con_fs = ec.feature_space.get(conditioning_namespace, conditioning_namespace);
    ec.reset_total_sum_feat_sq();
    ec.num_features += con_fs.size();
  }
}

void del_example_conditioning(search_private& priv, example& ec)
{
  del_features_in_top_namespace(priv, ec, conditioning_namespace);
}

inline size_t cs_get_costs_size(bool isCB, polylabel& ld) { return isCB ? ld.cb.costs.size() : ld.cs.costs.size(); }

inline uint32_t cs_get_cost_index(bool isCB, polylabel& ld, size_t k)
{
  return isCB ? ld.cb.costs[k].action : ld.cs.costs[k].class_index;
}

inline float cs_get_cost_partial_prediction(bool isCB, polylabel& ld, size_t k)
{
  return isCB ? ld.cb.costs[k].partial_prediction : ld.cs.costs[k].partial_prediction;
}

inline void cs_set_cost_loss(bool isCB, polylabel& ld, size_t k, float val)
{
  if (isCB)
    ld.cb.costs[k].cost = val;
  else
    ld.cs.costs[k].x = val;
}

inline void cs_costs_erase(bool isCB, polylabel& ld)
{
  if (isCB)
    ld.cb.costs.clear();
  else
    ld.cs.costs.clear();
}

inline void cs_costs_reserve(bool isCB, polylabel& ld, size_t new_size)
{
  if (isCB)
    ld.cb.costs.reserve(new_size);
  else
    ld.cs.costs.reserve(new_size);
}

inline void cs_cost_push_back(bool isCB, polylabel& ld, uint32_t index, float value)
{
  if (isCB)
  {
    CB::cb_class cost{value, index, 0.};
    ld.cb.costs.push_back(cost);
  }
  else
  {
    CS::wclass cost = {value, index, 0., 0.};
    ld.cs.costs.push_back(cost);
  }
}

polylabel& allowed_actions_to_ld(search_private& priv, size_t ec_cnt, const action* allowed_actions,
    size_t allowed_actions_cnt, const float* allowed_actions_cost)
{
  bool isCB = priv.cb_learner;
  polylabel& ld = priv.allowed_actions_cache;
  uint32_t num_costs = static_cast<uint32_t>(cs_get_costs_size(isCB, ld));

  if (priv.is_ldf)  // LDF version easier
  {
    if (num_costs > ec_cnt)
      cs_costs_reserve(isCB, ld, ec_cnt);
    else if (num_costs < ec_cnt)
      for (action k = num_costs; k < ec_cnt; k++) cs_cost_push_back(isCB, ld, k, FLT_MAX);
  }
  else if (priv.use_action_costs)
  {
    // TODO: Weight
    if (allowed_actions == nullptr)
    {
      if (cs_get_costs_size(isCB, ld) != priv.A)
      {
        cs_costs_erase(isCB, ld);
        for (action k = 0; k < priv.A; k++) cs_cost_push_back(isCB, ld, k + 1, 0.);
      }
      for (action k = 0; k < priv.A; k++) cs_set_cost_loss(isCB, ld, k, allowed_actions_cost[k]);
    }
    else  // manually specified actions
    {
      cs_costs_erase(isCB, ld);
      for (action k = 0; k < allowed_actions_cnt; k++)
        cs_cost_push_back(isCB, ld, allowed_actions[k], allowed_actions_cost[k]);
    }
  }
  else  // non-LDF version, no action costs
  {
    if ((allowed_actions == nullptr) || (allowed_actions_cnt == 0))  // any action is allowed
    {
      if (num_costs != priv.A)  // if there are already A-many actions, they must be the right ones, unless the user did
                                // something stupid like putting duplicate allowed_actions...
      {
        cs_costs_erase(isCB, ld);
        for (action k = 0; k < priv.A; k++) cs_cost_push_back(isCB, ld, k + 1, FLT_MAX);  //+1 because MC is 1-based
      }
    }
    else  // we need to peek at allowed_actions
    {
      cs_costs_erase(isCB, ld);
      for (size_t i = 0; i < allowed_actions_cnt; i++) cs_cost_push_back(isCB, ld, allowed_actions[i], FLT_MAX);
    }
  }

  return ld;
}

void allowed_actions_to_label(search_private& priv, size_t ec_cnt, const action* allowed_actions,
    size_t allowed_actions_cnt, const float* allowed_actions_cost, const action* oracle_actions,
    size_t oracle_actions_cnt, polylabel& lab)
{
  bool isCB = priv.cb_learner;
  if (priv.is_ldf)  // LDF version easier
  {
    cs_costs_erase(isCB, lab);
    for (action k = 0; k < ec_cnt; k++)
      cs_cost_push_back(isCB, lab, k, array_contains<action>(k, oracle_actions, oracle_actions_cnt) ? 0.f : 1.f);
  }
  else if (priv.use_action_costs)
  {
    // TODO: Weight
    if (allowed_actions == nullptr)
    {
      if (cs_get_costs_size(isCB, lab) != priv.A)
      {
        cs_costs_erase(isCB, lab);
        for (action k = 0; k < priv.A; k++) cs_cost_push_back(isCB, lab, k + 1, 0.);
      }
      for (action k = 0; k < priv.A; k++) cs_set_cost_loss(isCB, lab, k, allowed_actions_cost[k]);
    }
    else  // manually specified actions
    {
      cs_costs_erase(isCB, lab);
      for (action k = 0; k < allowed_actions_cnt; k++)
        cs_cost_push_back(isCB, lab, allowed_actions[k], allowed_actions_cost[k]);
    }
  }
  else  // non-LDF, no action costs
  {
    if ((allowed_actions == nullptr) || (allowed_actions_cnt == 0))  // any action is allowed
    {
      bool set_to_one = false;
      if (cs_get_costs_size(isCB, lab) != priv.A)
      {
        cs_costs_erase(isCB, lab);
        for (action k = 0; k < priv.A; k++) cs_cost_push_back(isCB, lab, k + 1, 1.);
        set_to_one = true;
      }
      if (oracle_actions_cnt <= 1)  // common case to speed up
      {
        if (!set_to_one)
          for (action k = 0; k < priv.A; k++) cs_set_cost_loss(isCB, lab, k, 1.);
        if (oracle_actions_cnt == 1) cs_set_cost_loss(isCB, lab, oracle_actions[0] - 1, 0.);
      }
      else
      {
        for (action k = 0; k < priv.A; k++)
          cs_set_cost_loss(isCB, lab, k, array_contains<action>(k + 1, oracle_actions, oracle_actions_cnt) ? 0.f : 1.f);
      }
    }
    else  // only some actions are allowed
    {
      cs_costs_erase(isCB, lab);
      float w = 1.;  // array_contains<action>(3, oracle_actions, oracle_actions_cnt) ? 5.f : 1.f;
      for (size_t i = 0; i < allowed_actions_cnt; i++)
      {
        action k = allowed_actions[i];
        cs_cost_push_back(
            isCB, lab, k, (array_contains<action>(k, oracle_actions, oracle_actions_cnt)) ? 0.f : w);  // 1.f );
      }
    }
  }
}

template <class T>
void ensure_size(v_array<T>& A, size_t sz)
{
  A.resize_but_with_stl_behavior(sz);
}

template <class T>
void push_at(v_array<T>& v, T item, size_t pos)
{
  if (pos > v.size()) { v.resize_but_with_stl_behavior(pos); }
  v.insert(v.begin() + pos, item);
}

action choose_oracle_action(search_private& priv, size_t ec_cnt, const action* oracle_actions,
    size_t oracle_actions_cnt, const action* allowed_actions, size_t allowed_actions_cnt,
    const float* allowed_actions_cost)
{
  action a = static_cast<action>(-1);
  if (priv.use_action_costs)
  {
    size_t K = (allowed_actions == nullptr) ? priv.A : allowed_actions_cnt;
    cdbg << "costs = [";
    for (size_t k = 0; k < K; k++) cdbg << ' ' << allowed_actions_cost[k];
    cdbg << " ]" << endl;
    float min_cost = FLT_MAX;
    for (size_t k = 0; k < K; k++) min_cost = std::min(min_cost, allowed_actions_cost[k]);
    cdbg << "min_cost = " << min_cost;
    if (min_cost < FLT_MAX)
    {
      size_t count = 0;
      for (size_t k = 0; k < K; k++)
        if (allowed_actions_cost[k] <= min_cost)
        {
          cdbg << ", hit @ " << k;
          count++;
          if ((count == 1) || (priv._random_state->get_and_update_random() < 1. / static_cast<float>(count)))
          {
            a = (allowed_actions == nullptr) ? static_cast<uint32_t>(k + 1) : allowed_actions[k];
            cdbg << "***";
          }
        }
    }
    cdbg << endl;
  }

  if (a == static_cast<action>(-1))
  {
    if ((priv.perturb_oracle > 0.) && (priv.state == INIT_TRAIN) &&
        (priv._random_state->get_and_update_random() < priv.perturb_oracle))
      oracle_actions_cnt = 0;
    a = (oracle_actions_cnt > 0)
        ? oracle_actions[random(priv._random_state, oracle_actions_cnt)]
        : (allowed_actions_cnt > 0) ? allowed_actions[random(priv._random_state, allowed_actions_cnt)]
                                    : priv.is_ldf ? static_cast<action>(random(priv._random_state, ec_cnt))
                                                  : static_cast<action>(1 + random(priv._random_state, priv.A));
  }
  cdbg << "choose_oracle_action from oracle_actions = [";
  for (size_t i = 0; i < oracle_actions_cnt; i++) cdbg << " " << oracle_actions[i];
  cdbg << " ], ret=" << a << endl;
  if (need_memo_foreach_action(priv) && (priv.state == INIT_TRAIN))
  {
    v_array<action_cache>* this_cache = new v_array<action_cache>();
    // TODO we don't really need to construct this polylabel
    polylabel l = allowed_actions_to_ld(priv, 1, allowed_actions, allowed_actions_cnt, allowed_actions_cost);
    size_t K = cs_get_costs_size(priv.cb_learner, l);
    for (size_t k = 0; k < K; k++)
    {
      action cl = cs_get_cost_index(priv.cb_learner, l, k);
      float cost = array_contains(cl, oracle_actions, oracle_actions_cnt) ? 0.f : 1.f;
      this_cache->push_back(action_cache(0., cl, cl == a, cost));
    }
    assert(priv.memo_foreach_action.size() == priv.meta_t + priv.t - 1);
    priv.memo_foreach_action.push_back(this_cache);
    cdbg << "memo_foreach_action[" << priv.meta_t + priv.t - 1 << "] = " << this_cache << " from oracle" << endl;
  }
  return a;
}

action single_prediction_notLDF(search_private& priv, example& ec, int policy, const action* allowed_actions,
    size_t allowed_actions_cnt, const float* allowed_actions_cost, float& a_cost,
    action override_action)  // if override_action != -1, then we return it as the action and a_cost is set to the
                             // appropriate cost for that action
{
  vw& all = *priv.all;
  polylabel old_label = ec.l;
  bool need_partial_predictions = need_memo_foreach_action(priv) ||
      (priv.metaoverride && priv.metaoverride->_foreach_action) || (override_action != static_cast<action>(-1)) ||
      priv.active_csoaa;
  if ((allowed_actions_cnt > 0) || need_partial_predictions)
    ec.l = allowed_actions_to_ld(priv, 1, allowed_actions, allowed_actions_cnt, allowed_actions_cost);
  else
    ec.l.cs = priv.empty_cs_label;

  cdbg << "allowed_actions_cnt=" << allowed_actions_cnt << ", ec.l = [";
  for (size_t i = 0; i < ec.l.cs.costs.size(); i++)
    cdbg << ' ' << ec.l.cs.costs[i].class_index << ':' << ec.l.cs.costs[i].x;
  cdbg << " ]" << endl;

  as_singleline(priv.base_learner)->predict(ec, policy);

  uint32_t act = priv.active_csoaa ? ec.pred.active_multiclass.predicted_class : ec.pred.multiclass;
  cdbg << "a=" << act << " from";
  if (allowed_actions)
  {
    for (size_t ii = 0; ii < allowed_actions_cnt; ii++) cdbg << ' ' << allowed_actions[ii];
  }
  cdbg << endl;
  a_cost = ec.partial_prediction;
  cdbg << "a_cost = " << a_cost << endl;

  if (override_action != static_cast<action>(-1)) act = override_action;

  if (need_partial_predictions)
  {
    size_t K = cs_get_costs_size(priv.cb_learner, ec.l);
    float min_cost = FLT_MAX;
    for (size_t k = 0; k < K; k++)
    {
      float cost = cs_get_cost_partial_prediction(priv.cb_learner, ec.l, k);
      if (cost < min_cost) min_cost = cost;
    }
    v_array<action_cache>* this_cache = nullptr;
    if (need_memo_foreach_action(priv) && (override_action == static_cast<action>(-1)))
    {
      this_cache = new v_array<action_cache>();
    }
    for (size_t k = 0; k < K; k++)
    {
      action cl = cs_get_cost_index(priv.cb_learner, ec.l, k);
      float cost = cs_get_cost_partial_prediction(priv.cb_learner, ec.l, k);
      if (priv.metaoverride && priv.metaoverride->_foreach_action)
        priv.metaoverride->_foreach_action(*priv.metaoverride->sch, priv.t - 1, min_cost, cl, cl == act, cost);
      if (override_action == cl) a_cost = cost;
      if (this_cache) this_cache->push_back(action_cache(min_cost, cl, cl == act, cost));
    }
    if (this_cache)
    {
      assert(priv.memo_foreach_action.size() == priv.meta_t + priv.t - 1);
      priv.memo_foreach_action.push_back(this_cache);
      cdbg << "memo_foreach_action[" << priv.meta_t + priv.t - 1 << "] = " << this_cache << endl;
    }
  }

  if ((priv.state == INIT_TRAIN) && (priv.subsample_timesteps <= -1))  // active learning
  {
    size_t K = cs_get_costs_size(priv.cb_learner, ec.l);
    float min_cost = FLT_MAX, min_cost2 = FLT_MAX;
    for (size_t k = 0; k < K; k++)
    {
      float cost = cs_get_cost_partial_prediction(priv.cb_learner, ec.l, k);
      if (cost < min_cost)
      {
        min_cost2 = min_cost;
        min_cost = cost;
      }
      else if (cost < min_cost2)
      {
        min_cost2 = cost;
      }
    }
    if (min_cost2 < FLT_MAX)
      priv.active_uncertainty.push_back(std::make_pair(min_cost2 - min_cost, priv.t + priv.meta_t));
  }
  if ((priv.state == INIT_TRAIN) && priv.active_csoaa)
  {
    if (priv.cb_learner) THROW("cannot use active_csoaa with cb learning");
    size_t cur_t = priv.t + priv.meta_t - 1;
    while (priv.active_known.size() <= cur_t)
    {
      priv.active_known.push_back({});
      cdbg << "active_known length now " << priv.active_known.size() << endl;
    }
    priv.active_known[cur_t].clear();
    assert(ec.l.cs.costs.size() > 0);
    for (size_t k = 0; k < ec.l.cs.costs.size(); k++)
    {
      /* priv.active_known[cur_t].push_back( ec.l.cs.costs[k].pred_is_certain
                                          ? ec.l.cs.costs[k].partial_prediction
                                          : FLT_MAX );
                                          cdbg << "active_known[" << cur_t << "][" << (priv.active_known[cur_t].size() -
         1) << "] = certain=" << ec.l.cs.costs[k].pred_is_certain << ", cost=" << ec.l.cs.costs[k].partial_prediction <<
         "}" << endl; */
      CS::wclass& wc = ec.l.cs.costs[k];
      // Get query_needed from pred
      const auto& query_list = ec.pred.active_multiclass.more_info_required_for_classes;
      bool query_needed = std::find(query_list.begin(), query_list.end(), wc.class_index) != query_list.end();
      std::pair<CS::wclass&, bool> p = {wc, query_needed};
      // Push into active_known[cur_t] with wc
      priv.active_known[cur_t].push_back(p);
      // cdbg << "active_known[" << cur_t << "][" << (priv.active_known[cur_t].size() - 1) << "] = " << wc.class_index
      // << ':' << wc.x << " pp=" << wc.partial_prediction << " query_needed=" << wc.query_needed << " max_pred=" <<
      // wc.max_pred << " min_pred=" << wc.min_pred << " is_range_overlapped=" << wc.is_range_overlapped << "
      // is_range_large=" << wc.is_range_large << endl;
      // query_needed=" << ec.l.cs.costs[k].query_needed << ", cost=" << ec.l.cs.costs[k].partial_prediction << "}" <<
      // endl;
    }
  }

  // generate raw predictions if necessary
  if ((priv.state == INIT_TEST) && (all.raw_prediction != nullptr))
  {
    priv.rawOutputStringStream->str("");
    for (size_t k = 0; k < cs_get_costs_size(priv.cb_learner, ec.l); k++)
    {
      if (k > 0) (*priv.rawOutputStringStream) << ' ';
      (*priv.rawOutputStringStream) << cs_get_cost_index(priv.cb_learner, ec.l, k) << ':'
                                    << cs_get_cost_partial_prediction(priv.cb_learner, ec.l, k);
    }
    all.print_text_by_ref(all.raw_prediction.get(), priv.rawOutputStringStream->str(), ec.tag);
  }

  ec.l = old_label;

  priv.total_predictions_made++;
  priv.num_features += ec.get_num_features();

  return act;
}

action single_prediction_LDF(search_private& priv, example* ecs, size_t ec_cnt, int policy, float& a_cost,
    action override_action)  // if override_action != -1, then we return it as the action and a_cost is set to the
                             // appropriate cost for that action
{
  bool need_partial_predictions = need_memo_foreach_action(priv) ||
      (priv.metaoverride && priv.metaoverride->_foreach_action) || (override_action != static_cast<action>(-1));

  CS::default_label(priv.ldf_test_label);
  CS::wclass wc = {0., 1, 0., 0.};
  priv.ldf_test_label.costs.push_back(wc);

  // keep track of best (aka chosen) action
  float best_prediction = 0.;
  action best_action = 0;

  size_t start_K = (priv.is_ldf && COST_SENSITIVE::ec_is_example_header(ecs[0])) ? 1 : 0;

  v_array<action_cache>* this_cache = nullptr;
  if (need_partial_predictions)
  {
    this_cache = new v_array<action_cache>();
  }

  for (action a = static_cast<uint32_t>(start_K); a < ec_cnt; a++)
  {
    cdbg << "== single_prediction_LDF a=" << a << "==" << endl;
    if (start_K > 0) LabelDict::add_example_namespaces_from_example(ecs[a], ecs[0]);

    polylabel old_label = ecs[a].l;
    ecs[a].l.cs = priv.ldf_test_label;

    multi_ex tmp;
    uint64_t old_offset = ecs[a].ft_offset;
    ecs[a].ft_offset = priv.offset;
    tmp.push_back(&ecs[a]);
    as_multiline(priv.base_learner)->predict(tmp, policy);

    ecs[a].ft_offset = old_offset;
    cdbg << "partial_prediction[" << a << "] = " << ecs[a].partial_prediction << endl;

    if (override_action != static_cast<action>(-1))
    {
      if (a == override_action) a_cost = ecs[a].partial_prediction;
    }
    else if ((a == start_K) || (ecs[a].partial_prediction < best_prediction))
    {
      best_prediction = ecs[a].partial_prediction;
      best_action = a;
      a_cost = best_prediction;
    }
    if (this_cache) this_cache->push_back(action_cache(0., a, false, ecs[a].partial_prediction));

    priv.num_features += ecs[a].get_num_features();
    ecs[a].l = old_label;
    if (start_K > 0) LabelDict::del_example_namespaces_from_example(ecs[a], ecs[0]);
  }
  if (override_action != static_cast<action>(-1))
    best_action = override_action;
  else
    a_cost = best_prediction;

  if (this_cache)
  {
    for (size_t i = 0; i < this_cache->size(); i++)
    {
      action_cache& ac = (*this_cache)[i];
      ac.min_cost = a_cost;
      ac.is_opt = (ac.k == best_action);
      if (priv.metaoverride && priv.metaoverride->_foreach_action)
        priv.metaoverride->_foreach_action(*priv.metaoverride->sch, priv.t - 1, ac.min_cost, ac.k, ac.is_opt, ac.cost);
    }
    if (need_memo_foreach_action(priv) && (override_action == static_cast<action>(-1)))
      priv.memo_foreach_action.push_back(this_cache);
    else
    {
      delete this_cache;
    }
  }

  // TODO: generate raw predictions if necessary

  priv.total_predictions_made++;
  return best_action;
}

int choose_policy(search_private& priv, bool advance_prng = true)
{
  RollMethod method = (priv.state == INIT_TEST) ? POLICY
                                                : (priv.state == LEARN)
          ? priv.rollout_method
          : (priv.state == INIT_TRAIN) ? priv.rollin_method : NO_ROLLOUT;  // this should never happen
  switch (method)
  {
    case POLICY:
      return random_policy(priv, priv.allow_current_policy || (priv.state == INIT_TEST), false, advance_prng);

    case ORACLE:
      return -1;

    case MIX_PER_STATE:
      return random_policy(priv, priv.allow_current_policy, true, advance_prng);

    case MIX_PER_ROLL:
      if (priv.mix_per_roll_policy == -2)  // then we have to choose one!
        priv.mix_per_roll_policy = random_policy(priv, priv.allow_current_policy, true, advance_prng);
      return priv.mix_per_roll_policy;

    case NO_ROLLOUT:
    default:
      THROW("internal error (bug): trying to rollin or rollout with NO_ROLLOUT");
  }
}

bool cached_item_equivalent(unsigned char* const& A, unsigned char* const& B)
{
  size_t sz_A = *A;
  size_t sz_B = *B;
  if (sz_A != sz_B) return false;
  return memcmp(A, B, sz_A) == 0;
}

// returns true if found and do_store is false. if do_store is true, always returns true.
bool cached_action_store_or_find(search_private& priv, ptag mytag, const ptag* condition_on,
    const char* condition_on_names, action_repr* condition_on_actions, size_t condition_on_cnt, int policy,
    size_t learner_id, action& a, bool do_store, float& a_cost)
{
  if (priv.no_caching) return do_store;
  if (mytag == 0) return do_store;  // don't attempt to cache when tag is zero

  size_t sz = sizeof(size_t) + sizeof(ptag) + sizeof(int) + sizeof(size_t) + sizeof(size_t) +
      condition_on_cnt * (sizeof(ptag) + sizeof(action) + sizeof(char));
  if (sz % 4 != 0) sz += 4 - (sz % 4);  // make sure sz aligns to 4 so that uniform_hash does the right thing

  byte_array item(new uint8_t[sz]);
  uint8_t* here = item.get();
  // get rid of a valgrind warning about uninitialized memory
  memset(here, 0, sz);
  *here = static_cast<unsigned char>(sz);
  here += sizeof(size_t);
  *here = static_cast<uint8_t>(mytag);
  here += sizeof(ptag);
  *here = static_cast<uint8_t>(policy);
  here += sizeof(int);
  *here = static_cast<unsigned char>(learner_id);
  here += sizeof(size_t);
  *here = static_cast<unsigned char>(condition_on_cnt);
  here += sizeof(size_t);
  for (size_t i = 0; i < condition_on_cnt; i++)
  {
    *here = static_cast<uint8_t>(condition_on[i]);
    here += sizeof(ptag);
    *here = static_cast<uint8_t>(condition_on_actions[i].a);
    here += sizeof(action);
    *here = condition_on_names[i];
    here += sizeof(char);  // SPEEDUP: should we align this at 4?
  }
  if (do_store)
  {
    priv.cache_hash_map.emplace(std::move(item), scored_action(a, a_cost));
    return true;
  }
  else  // its a find
  {
    auto sa_iter = priv.cache_hash_map.find(item);
    if (sa_iter == priv.cache_hash_map.end()) return false;
    a = sa_iter->second.a;
    a_cost = sa_iter->second.s;
    return a != static_cast<action>(-1);
  }
}

void generate_training_example(search_private& priv, polylabel& losses, float weight, bool add_conditioning = true,
    float min_loss = FLT_MAX)  // min_loss = FLT_MAX means "please compute it for me as the actual min"; any other value
                               // means to use this
{
  // should we really subtract out min-loss?
  // float min_loss = FLT_MAX;
  if (priv.cb_learner)
  {
    if (min_loss == FLT_MAX)
      for (size_t i = 0; i < losses.cb.costs.size(); i++) min_loss = std::min(min_loss, losses.cb.costs[i].cost);
    for (size_t i = 0; i < losses.cb.costs.size(); i++) losses.cb.costs[i].cost = losses.cb.costs[i].cost - min_loss;
  }
  else
  {
    if (min_loss == FLT_MAX)
      for (size_t i = 0; i < losses.cs.costs.size(); i++) min_loss = std::min(min_loss, losses.cs.costs[i].x);
    for (size_t i = 0; i < losses.cs.costs.size(); i++)
      losses.cs.costs[i].x = (losses.cs.costs[i].x - min_loss) * weight;
  }

  if (!priv.is_ldf)  // not LDF
  {
    // since we're not LDF, it should be the case that ec_ref_cnt == 1
    // and learn_ec_ref[0] is a pointer to a single example
    assert(priv.learn_ec_ref_cnt == 1);
    assert(priv.learn_ec_ref != nullptr);

    example& ec = priv.learn_ec_ref[0];
    polylabel old_label = ec.l;
    ec.l = losses;  // labels;
    if (add_conditioning)
      add_example_conditioning(priv, ec, priv.learn_condition_on.size(), priv.learn_condition_on_names.begin(),
          priv.learn_condition_on_act.begin());
    for (size_t is_local = 0; is_local <= static_cast<size_t>(priv.xv); is_local++)
    {
      int learner = select_learner(priv, priv.current_policy, priv.learn_learner_id, true, is_local > 0);
      cdbg << "BEGIN base_learner->learn(ec, " << learner << ")" << endl;
      as_singleline(priv.base_learner)->learn(ec, learner);
      cdbg << "END   base_learner->learn(ec, " << learner << ")" << endl;
    }
    if (add_conditioning) del_example_conditioning(priv, ec);
    ec.l = old_label;
    priv.total_examples_generated++;
  }
  else  // is  LDF
  {
    assert(cs_get_costs_size(priv.cb_learner, losses) == priv.learn_ec_ref_cnt);
    size_t start_K = (priv.is_ldf && COST_SENSITIVE::ec_is_example_header(priv.learn_ec_ref[0])) ? 1 : 0;

    // TODO: weight
    if (add_conditioning)
      for (action a = static_cast<uint32_t>(start_K); a < priv.learn_ec_ref_cnt; a++)
      {
        example& ec = priv.learn_ec_ref[a];
        add_example_conditioning(priv, ec, priv.learn_condition_on.size(), priv.learn_condition_on_names.begin(),
            priv.learn_condition_on_act.begin());
      }

    for (size_t is_local = 0; is_local <= static_cast<size_t>(priv.xv); is_local++)
    {
      int learner = select_learner(priv, priv.current_policy, priv.learn_learner_id, true, is_local > 0);

      // create an example collection for

      multi_ex tmp;
      uint64_t tmp_offset = 0;
      if (priv.learn_ec_ref_cnt > start_K) tmp_offset = priv.learn_ec_ref[start_K].ft_offset;
      for (action a = static_cast<uint32_t>(start_K); a < priv.learn_ec_ref_cnt; a++)
      {
        example& ec = priv.learn_ec_ref[a];
        CS::label& lab = ec.l.cs;
        if (lab.costs.size() == 0)
        {
          CS::wclass wc = {0., a - static_cast<uint32_t>(start_K), 0., 0.};
          lab.costs.push_back(wc);
        }
        lab.costs[0].x = losses.cs.costs[a - start_K].x;
        // store the offset to restore it later
        ec.ft_offset = priv.offset;
        // create the example collection used to learn
        tmp.push_back(&ec);
        cdbg << "generate_training_example called learn on action a=" << a << ", costs.size=" << lab.costs.size()
             << " ec=" << &ec << endl;
        priv.total_examples_generated++;
      }

      // learn with the multiline example
      as_multiline(priv.base_learner)->learn(tmp, learner);

      // restore the offsets in examples
      int i = 0;
      for (action a = static_cast<uint32_t>(start_K); a < priv.learn_ec_ref_cnt; a++, i++)
        priv.learn_ec_ref[a].ft_offset = tmp_offset;
    }

    if (add_conditioning)
      for (action a = static_cast<uint32_t>(start_K); a < priv.learn_ec_ref_cnt; a++)
      {
        example& ec = priv.learn_ec_ref[a];
        del_example_conditioning(priv, ec);
      }
  }
}

bool search_predictNeedsExample(search_private& priv)
{
  // this is basically copied from the logic of search_predict()
  switch (priv.state)
  {
    case INITIALIZE:
      return false;
    case GET_TRUTH_STRING:
      return false;
    case INIT_TEST:
      return true;
    case INIT_TRAIN:
      // TODO: do we need to do something here for metatasks?
      // if (priv.beam && (priv.t < priv.beam_actions.size()))
      //  return false;
      if (priv.rollout_method == NO_ROLLOUT) return true;
      break;
    case LEARN:
      if (priv.t + priv.meta_t < priv.learn_t)
        return false;  // TODO: in meta search mode with foreach feature we'll need it even here
      if (priv.t + priv.meta_t == priv.learn_t)
        return true;  // SPEEDUP: we really only need it on the last learn_a, but this is hard to know...
      // t > priv.learn_t
      if ((priv.rollout_num_steps > 0) && (priv.loss_declared_cnt >= priv.rollout_num_steps)) return false;  // skipping
      break;
  }

  int pol = choose_policy(priv, false);  // choose a policy but don't advance prng
  return (pol != -1);
}

void foreach_action_from_cache(search_private& priv, size_t t, action override_a = static_cast<action>(-1))
{
  cdbg << "foreach_action_from_cache: t=" << t << ", memo_foreach_action.size()=" << priv.memo_foreach_action.size()
       << ", override_a=" << override_a << endl;
  assert(t < priv.memo_foreach_action.size());
  v_array<action_cache>* cached = priv.memo_foreach_action[t];
  if (!cached) return;  // the only way this can happen is if the metatask overrode this action
  cdbg << "memo_foreach_action size = " << cached->size() << endl;
  for (size_t id = 0; id < cached->size(); id++)
  {
    action_cache& ac = (*cached)[id];
    priv.metaoverride->_foreach_action(*priv.metaoverride->sch, t - priv.meta_t, ac.min_cost, ac.k,
        (override_a == static_cast<action>(-1)) ? ac.is_opt : (ac.k == override_a), ac.cost);
  }
}

// note: ec_cnt should be 1 if we are not LDF
action search_predict(search_private& priv, example* ecs, size_t ec_cnt, ptag mytag, const action* oracle_actions,
    size_t oracle_actions_cnt, const ptag* condition_on, const char* condition_on_names, const action* allowed_actions,
    size_t allowed_actions_cnt, const float* allowed_actions_cost, size_t learner_id, float& a_cost, float /* weight */)
{
  size_t condition_on_cnt = condition_on_names ? strlen(condition_on_names) : 0;
  size_t t = priv.t + priv.meta_t;
  priv.t++;

  // make sure parameters come in pairs correctly
  assert((oracle_actions == nullptr) == (oracle_actions_cnt == 0));
  assert((condition_on == nullptr) == (condition_on_names == nullptr));
  assert(((allowed_actions == nullptr) && (allowed_actions_cost == nullptr)) == (allowed_actions_cnt == 0));
  assert(priv.use_action_costs == (allowed_actions_cost != nullptr));
  if (allowed_actions_cost != nullptr) assert(oracle_actions == nullptr);

  // if we're just after the string, choose an oracle action
  if ((priv.state == GET_TRUTH_STRING) || priv.force_oracle)
  {
    action a = choose_oracle_action(
        priv, ec_cnt, oracle_actions, oracle_actions_cnt, allowed_actions, allowed_actions_cnt, allowed_actions_cost);
    // if (priv.metaoverride && priv.metaoverride->_post_prediction)
    //  priv.metaoverride->_post_prediction(*priv.metaoverride->sch, t-priv.meta_t, a, 0.);
    a_cost = 0.;
    return a;
  }

  // if we're in LEARN mode and before learn_t, return the train action
  if ((priv.state == LEARN) && (t < priv.learn_t))
  {
    assert(t < priv.train_trajectory.size());
    action a = priv.train_trajectory[t].a;
    a_cost = priv.train_trajectory[t].s;
    cdbg << "LEARN " << t << " < priv.learn_t ==> a=" << a << ", a_cost=" << a_cost << endl;
    if (priv.metaoverride && priv.metaoverride->_foreach_action) foreach_action_from_cache(priv, t);
    if (priv.metaoverride && priv.metaoverride->_post_prediction)
      priv.metaoverride->_post_prediction(*priv.metaoverride->sch, t - priv.meta_t, a, a_cost);
    return a;
  }

  // for LDF, # of valid actions is ec_cnt; otherwise it's either allowed_actions_cnt or A
  size_t valid_action_cnt = priv.is_ldf ? ec_cnt : (allowed_actions_cnt > 0) ? allowed_actions_cnt : priv.A;

  // if we're in LEARN mode and _at_ learn_t, then:
  //   - choose the next action
  //   - decide if we're done
  //   - if we are, then copy/mark the example ref
  if ((priv.state == LEARN) && (t == priv.learn_t))
  {
    action a = static_cast<action>(priv.learn_a_idx);
    priv.loss_declared_cnt = 0;

    cdbg << "LEARN " << t << " = priv.learn_t ==> a=" << a << ", learn_a_idx=" << priv.learn_a_idx
         << " valid_action_cnt=" << valid_action_cnt << endl;
    priv.learn_a_idx++;

    // check to see if we're done with available actions
    if (priv.learn_a_idx >= valid_action_cnt)
    {
      priv.done_with_all_actions = true;
      priv.learn_learner_id = learner_id;

      // set reference or copy example(s)
      if (oracle_actions_cnt > 0) priv.learn_oracle_action = oracle_actions[0];
      priv.learn_ec_ref_cnt = ec_cnt;
      if (priv.examples_dont_change)
        priv.learn_ec_ref = ecs;
      else
      {
        priv.learn_ec_copy.resize(ec_cnt);
        for (size_t i = 0; i < ec_cnt; i++) { VW::copy_example_data_with_label(&priv.learn_ec_copy[i], ecs + i); }

        priv.learn_ec_ref = priv.learn_ec_copy.data();
      }

      // copy conditioning stuff and allowed actions
      if (priv.auto_condition_features)
      {
        priv.learn_condition_on.resize_but_with_stl_behavior(condition_on_cnt);
        ensure_size(priv.learn_condition_on_act, condition_on_cnt);

        memcpy(priv.learn_condition_on.begin(), condition_on, condition_on_cnt * sizeof(ptag));

        for (size_t i = 0; i < condition_on_cnt; i++)
        {
          push_at(priv.learn_condition_on_act,
              action_repr(((1 <= condition_on[i]) && (condition_on[i] < priv.ptag_to_action.size()))
                      ? priv.ptag_to_action[condition_on[i]]
                      : 0),
              i);
        }

        if (condition_on_names == nullptr)
        {
          ensure_size(priv.learn_condition_on_names, 1);
          priv.learn_condition_on_names[0] = 0;
        }
        else
        {
          ensure_size(priv.learn_condition_on_names, strlen(condition_on_names) + 1);
          VW::string_cpy(priv.learn_condition_on_names.begin(), (strlen(condition_on_names) + 1), condition_on_names);
        }
      }

      if (allowed_actions && (allowed_actions_cnt > 0))
      {
        ensure_size(priv.learn_allowed_actions, allowed_actions_cnt);
        memcpy(priv.learn_allowed_actions.begin(), allowed_actions, allowed_actions_cnt * sizeof(action));
        cdbg_print_array("in LEARN, learn_allowed_actions", priv.learn_allowed_actions);
      }
    }

    assert((allowed_actions_cnt == 0) || (a < allowed_actions_cnt));

    a_cost = 0.;
    action a_name = (allowed_actions && (allowed_actions_cnt > 0)) ? allowed_actions[a] : priv.is_ldf ? a : (a + 1);
    if (priv.metaoverride && priv.metaoverride->_foreach_action)
    {
      foreach_action_from_cache(priv, t, a_name);
      if (priv.memo_foreach_action[t])
      {
        cdbg << "@ memo_foreach_action: t=" << t << ", a=" << a << ", cost=" << (*priv.memo_foreach_action[t])[a].cost
             << endl;
        a_cost = (*priv.memo_foreach_action[t])[a].cost;
      }
    }

    a = a_name;

    if (priv.metaoverride && priv.metaoverride->_post_prediction)
      priv.metaoverride->_post_prediction(*priv.metaoverride->sch, t - priv.meta_t, a, a_cost);
    return a;
  }

  if ((priv.state == LEARN) && (t > priv.learn_t) && (priv.rollout_num_steps > 0) &&
      (priv.loss_declared_cnt >= priv.rollout_num_steps))
  {
    cdbg << "... skipping" << endl;
    action a = priv.is_ldf ? 0 : ((allowed_actions && (allowed_actions_cnt > 0)) ? allowed_actions[0] : 1);
    if (priv.metaoverride && priv.metaoverride->_post_prediction)
      priv.metaoverride->_post_prediction(*priv.metaoverride->sch, t - priv.meta_t, a, 0.);
    if (priv.metaoverride && priv.metaoverride->_foreach_action) foreach_action_from_cache(priv, t);
    a_cost = 0.;
    return a;
  }

  if ((priv.state == INIT_TRAIN) || (priv.state == INIT_TEST) || ((priv.state == LEARN) && (t > priv.learn_t)))
  {
    // we actually need to run the policy

    int policy = choose_policy(priv);
    action a = 0;

    cdbg << "executing policy " << policy << endl;

    bool gte_here = (priv.state == INIT_TRAIN) && (priv.rollout_method == NO_ROLLOUT) &&
        ((oracle_actions_cnt > 0) || (priv.use_action_costs));
    a_cost = 0.;
    bool skip = false;

    if (priv.metaoverride && priv.metaoverride->_maybe_override_prediction &&
        (priv.state != LEARN))  // if LEARN and t>learn_t,then we cannot allow overrides!
    {
      skip = priv.metaoverride->_maybe_override_prediction(*priv.metaoverride->sch, t - priv.meta_t, a, a_cost);
      cdbg << "maybe_override_prediction --> " << skip << ", a=" << a << ", a_cost=" << a_cost << endl;
      if (skip && need_memo_foreach_action(priv)) priv.memo_foreach_action.push_back(nullptr);
    }

    if ((!skip) && (policy == -1))
      a = choose_oracle_action(priv, ec_cnt, oracle_actions, oracle_actions_cnt, allowed_actions, allowed_actions_cnt,
          allowed_actions_cost);  // TODO: we probably want to actually get costs for oracle actions???

    bool need_fea = (policy == -1) && priv.metaoverride && priv.metaoverride->_foreach_action;

    if ((policy >= 0) || gte_here || need_fea)  // the last case is we need to do foreach action
    {
      int learner = select_learner(priv, policy, learner_id, false, priv.state != INIT_TEST);

      ensure_size(priv.condition_on_actions, condition_on_cnt);
      for (size_t i = 0; i < condition_on_cnt; i++)
      {
        priv.condition_on_actions[i] = ((1 <= condition_on[i]) && (condition_on[i] < priv.ptag_to_action.size()))
            ? priv.ptag_to_action[condition_on[i]]
            : action_repr(0);
      }

      bool not_test = priv.all->training && !ecs[0].test_only;

      if ((!skip) && (!need_fea) && not_test &&
          cached_action_store_or_find(priv, mytag, condition_on, condition_on_names, priv.condition_on_actions.begin(),
              condition_on_cnt, policy, learner_id, a, false, a_cost))
        // if this succeeded, 'a' has the right action
        priv.total_cache_hits++;
      else  // we need to predict, and then cache, and maybe run foreach_action
      {
        size_t start_K = (priv.is_ldf && COST_SENSITIVE::ec_is_example_header(ecs[0])) ? 1 : 0;
        priv.last_action_repr.clear();
        if (priv.auto_condition_features)
          for (size_t n = start_K; n < ec_cnt; n++)
            add_example_conditioning(
                priv, ecs[n], condition_on_cnt, condition_on_names, priv.condition_on_actions.begin());

        if (((!skip) && (policy >= 0)) || need_fea)  // only make a prediction if we're going to use the output
        {
          if (priv.auto_condition_features && priv.acset.use_passthrough_repr)
          {
            if (priv.is_ldf) { THROW("search cannot use state representations in ldf mode"); }
            if (ecs[0].passthrough) { THROW("search cannot passthrough"); }
            ecs[0].passthrough = &priv.last_action_repr;
          }
          a = priv.is_ldf
              ? single_prediction_LDF(priv, ecs, ec_cnt, learner, a_cost, need_fea ? a : static_cast<action>(-1))
              : single_prediction_notLDF(priv, *ecs, learner, allowed_actions, allowed_actions_cnt,
                    allowed_actions_cost, a_cost, need_fea ? a : static_cast<action>(-1));

          cdbg << "passthrough = [";
          for (size_t kk = 0; kk < priv.last_action_repr.size(); kk++)
            cdbg << ' ' << priv.last_action_repr.indicies[kk] << ':' << priv.last_action_repr.values[kk];
          cdbg << " ]" << endl;

          ecs[0].passthrough = nullptr;
        }

        if (need_fea)
        {
          // TODO this
        }

        if (gte_here)
        {
          cdbg << "INIT_TRAIN, NO_ROLLOUT, at least one oracle_actions, a=" << a << endl;
          // we can generate a training example _NOW_ because we're not doing rollouts
          // allowed_actions_to_losses(priv, ec_cnt, allowed_actions, allowed_actions_cnt, oracle_actions,
          // oracle_actions_cnt, losses);
          allowed_actions_to_label(priv, ec_cnt, allowed_actions, allowed_actions_cnt, allowed_actions_cost,
              oracle_actions, oracle_actions_cnt, priv.gte_label);
          cdbg << "priv.gte_label = [";
          for (size_t i = 0; i < priv.gte_label.cs.costs.size(); i++)
            cdbg << ' ' << priv.gte_label.cs.costs[i].class_index << ':' << priv.gte_label.cs.costs[i].x;
          cdbg << " ]" << endl;

          priv.learn_ec_ref = ecs;
          priv.learn_ec_ref_cnt = ec_cnt;
          if (allowed_actions)
          {
            ensure_size(priv.learn_allowed_actions, allowed_actions_cnt);  // TODO: do we really need this?
            memcpy(priv.learn_allowed_actions.begin(), allowed_actions, allowed_actions_cnt * sizeof(action));
          }
          size_t old_learner_id = priv.learn_learner_id;
          priv.learn_learner_id = learner_id;
          generate_training_example(
              priv, priv.gte_label, 1., false);  // this is false because the conditioning has already been added!
          priv.learn_learner_id = old_learner_id;
        }

        if (priv.auto_condition_features)
          for (size_t n = start_K; n < ec_cnt; n++) del_example_conditioning(priv, ecs[n]);

        if (not_test && (!skip))
          cached_action_store_or_find(priv, mytag, condition_on, condition_on_names, priv.condition_on_actions.begin(),
              condition_on_cnt, policy, learner_id, a, true, a_cost);
      }
    }

    if (priv.state == INIT_TRAIN)
      priv.train_trajectory.push_back(scored_action(a, a_cost));  // note the action for future reference

    if (priv.metaoverride && priv.metaoverride->_post_prediction)
      priv.metaoverride->_post_prediction(*priv.metaoverride->sch, t - priv.meta_t, a, a_cost);

    return a;
  }

  THROW("error: predict called in unknown state");
}

inline bool cmp_size_t(const size_t a, const size_t b) { return a < b; }
inline bool cmp_size_t_pair(const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b)
{
  return ((a.first == b.first) && (a.second < b.second)) || (a.first < b.first);
}

inline size_t absdiff(size_t a, size_t b) { return (a < b) ? (b - a) : (a - b); }

void hoopla_permute(size_t* B, size_t* end)
{
  // from Curtis IPL 2004, "Darts and hoopla board design"
  // first sort
  size_t N = end - B;
  std::sort(B, end, cmp_size_t);
  // make some temporary space
  size_t* A = calloc_or_throw<size_t>((N + 1) * 2);
  A[N] = B[0];                // arbitrarily choose the maximum in the middle
  A[N + 1] = B[N - 1];        // so the maximum goes next to it
  size_t lo = N, hi = N + 1;  // which parts of A have we filled in? [lo,hi]
  size_t i = 0, j = N - 1;    // which parts of B have we already covered? [0,i] and [j,N-1]
  while (i + 1 < j)
  {
    // there are four options depending on where things get placed
    size_t d1 = absdiff(A[lo], B[i + 1]);  // put B[i+1] at the bottom
    size_t d2 = absdiff(A[lo], B[j - 1]);  // put B[j-1] at the bottom
    size_t d3 = absdiff(A[hi], B[i + 1]);  // put B[i+1] at the top
    size_t d4 = absdiff(A[hi], B[j - 1]);  // put B[j-1] at the top
    size_t mx = std::max(std::max(d1, d2), std::max(d3, d4));
    if (d1 >= mx)
      A[--lo] = B[++i];
    else if (d2 >= mx)
      A[--lo] = B[--j];
    else if (d3 >= mx)
      A[++hi] = B[++i];
    else
      A[++hi] = B[--j];
  }
  // copy it back to B
  memcpy(B, A + lo, N * sizeof(size_t));
  // clean up
  free(A);
}

void get_training_timesteps(search_private& priv, v_array<size_t>& timesteps)
{
  timesteps.clear();

  // if there's active learning, we need to
  if (priv.subsample_timesteps <= -1)
  {
    for (size_t i = 0; i < priv.active_uncertainty.size(); i++)
      if (priv._random_state->get_and_update_random() > priv.active_uncertainty[i].first)
        timesteps.push_back(priv.active_uncertainty[i].second - 1);
    /*
    float k = (float)priv.total_examples_generated;
    priv.ec_seq[t]->revert_weight = priv.all->loss->getRevertingWeight(priv.all->sd, priv.ec_seq[t].pred.scalar,
    priv.all->eta / powf(k, priv.all->power_t)); float importance = query_decision(active_str, *priv.ec_seq[t], k); if
    (importance > 0.) timesteps.push_back(pair<size_t,size_t>(0,t));
    */
  }
  // if there's no subsampling to do, just return [0,T)
  else if (priv.subsample_timesteps <= 0)
    for (size_t t = 0; t < priv.T; t++)
    {
      uint32_t count = 99;
      if (priv.active_csoaa && (t < priv.active_known.size()))
      {
        count = 0;
        for (std::pair<CS::wclass&, bool> wcq : priv.active_known[t])
          if (wcq.second)
          {
            count++;
            if (count > 1) break;
          }
      }
      if (count > 1) timesteps.push_back(t);
    }

  // if subsample in (0,1) then pick steps with that probability, but ensuring there's at least one!
  else if (priv.subsample_timesteps < 1)
  {
    for (size_t t = 0; t < priv.T; t++)
      if (priv._random_state->get_and_update_random() <= priv.subsample_timesteps) timesteps.push_back(t);

    if (timesteps.size() == 0)  // ensure at least one
      timesteps.push_back(static_cast<size_t>(priv._random_state->get_and_update_random() * priv.T));
  }

  // finally, if subsample >= 1, then pick (int) that many uniformly at random without replacement; could use an LFSR
  // but why? :P
  else
  {
    while ((timesteps.size() < static_cast<size_t>(priv.subsample_timesteps)) && (timesteps.size() < priv.T))
    {
      size_t t = static_cast<size_t>(priv._random_state->get_and_update_random() * static_cast<float>(priv.T));
      if (std::find(timesteps.begin(), timesteps.end(), t) == timesteps.end()) { timesteps.push_back(t); }
    }
    std::sort(timesteps.begin(), timesteps.end(), cmp_size_t);
  }

  if (!priv.linear_ordering) hoopla_permute(timesteps.begin(), timesteps.end());
}

struct final_item
{
  v_array<scored_action>* prefix;
  std::string str;
  float total_cost;
  final_item(v_array<scored_action>* p, std::string s, float ic) : prefix(p), str(s), total_cost(ic) {}
};

void free_final_item(final_item* p)
{
  delete p->prefix;
  delete p;
}

void BaseTask::Run()
{
  search_private& priv = *sch->priv;
  // make sure output is correct
  bool old_should_produce_string = priv.should_produce_string;
  if (!_final_run && !_with_output_string) priv.should_produce_string = false;
  // if this isn't a final run, it shouldn't count for loss
  float old_test_loss = priv.test_loss;
  // float old_learn_loss = priv.learn_loss;
  priv.learn_loss *= 0.5;
  float old_train_loss = priv.train_loss;

  if (priv.should_produce_string) priv.pred_string->str("");

  priv.t = 0;
  priv.metaoverride = this;
  priv.task->run(*sch, ec);
  priv.metaoverride = nullptr;
  priv.meta_t += priv.t;

  // restore
  if (_with_output_string && old_should_produce_string) _with_output_string(*sch, *priv.pred_string);

  priv.should_produce_string = old_should_produce_string;
  if (!_final_run)
  {
    priv.test_loss = old_test_loss;
    // priv.learn_loss = old_learn_loss;
    priv.train_loss = old_train_loss;
  }
}

void run_task(search& sch, multi_ex& ec)
{
  search_private& priv = *sch.priv;
  priv.num_calls_to_run++;
  if (priv.metatask && (priv.state != GET_TRUTH_STRING))
    priv.metatask->run(sch, ec);
  else
    priv.task->run(sch, ec);
}

void verify_active_csoaa(
    COST_SENSITIVE::label& losses, const std::vector<std::pair<CS::wclass&, bool>>& known, size_t t, float multiplier)
{
  float threshold = multiplier / std::sqrt(static_cast<float>(t));
  cdbg << "verify_active_csoaa, losses = [";
  for (COST_SENSITIVE::wclass& wc : losses.costs) cdbg << " " << wc.class_index << ":" << wc.x;
  cdbg << " ]" << endl;
  // cdbg_print_array("verify_active_csoaa,  known", known);
  size_t i = 0;
  for (COST_SENSITIVE::wclass& wc : losses.costs)
  {
    if (!known[i].second)
    {
      float err = static_cast<float>(std::pow(known[i].first.partial_prediction - wc.x, 2));
      if (err > threshold)
      {
        logger::errlog_error("verify_active_csoaa failed: truth {0}:{1}, known[{2}]={3}, error={4} vs threshold {5}",
                             wc.class_index /*0*/, wc.x/*1*/, i/*2*/, known[i].first.partial_prediction/*3*/,
                             err/*4*/, threshold/*5*/);
      }
    }
    i++;
  }
}

void advance_from_known_actions(search_private& priv)
{
  size_t t = priv.learn_t;
  if (!priv.active_csoaa) return;
  if (priv.active_csoaa_verify > 0.) return;
  if (t >= priv.active_known.size()) return;
  cdbg << "advance_from_known_actions t=" << t << " active_known.size()=" << priv.active_known.size()
       << " learn_a_idx=" << priv.learn_a_idx << endl;
  // cdbg_print_array(" active_known[t]", priv.active_known[t]);
  if (priv.learn_a_idx >= priv.active_known[t].size())
  {
    cdbg << "advance_from_known_actions setting done_with_all_actions=true (active_known[t].size()="
         << priv.active_known[t].size() << ")" << endl;
    priv.done_with_all_actions = true;
    return;
  }
  // if (priv.active_known[t][priv.learn_a_idx] >= FLT_MAX) return;
  if (priv.active_known[t][priv.learn_a_idx].second) return;
  // return;
  // wow, we actually found something we were confident about!
  /*
  cs_cost_push_back(priv.cb_learner,
                    priv.learn_losses,
                    priv.is_ldf ? (uint32_t)(priv.learn_a_idx - 1) : (uint32_t)priv.learn_a_idx,
                    priv.active_known[t][priv.learn_a_idx],
                    true);
  */
  priv.learn_losses.cs.costs.push_back(priv.active_known[t][priv.learn_a_idx].first);
  cdbg << "  --> adding " << priv.learn_a_idx << ":" << priv.active_known[t][priv.learn_a_idx].first.x << endl;
  priv.learn_a_idx++;
  advance_from_known_actions(priv);
}

template <bool is_learn>
void train_single_example(search& sch, bool is_test_ex, bool is_holdout_ex, multi_ex& ec_seq)
{
  search_private& priv = *sch.priv;
  vw& all = *priv.all;
  bool ran_test = false;  // we must keep track so that even if we skip test, we still update # of examples seen

  // if (! priv.no_caching)
  priv.cache_hash_map.clear();

  cdbg << "is_test_ex=" << is_test_ex << " vw_is_main=" << all.vw_is_main << endl;
  cdbg << "must_run_test = " << must_run_test(all, ec_seq, is_test_ex) << endl;
  // do an initial test pass to compute output (and loss)
  if (must_run_test(all, ec_seq, is_test_ex))
  {
    cdbg << "======================================== INIT TEST (" << priv.current_policy << ","
         << priv.read_example_last_pass << ") ========================================" << endl;

    ran_test = true;

    // do the prediction
    reset_search_structure(priv);
    priv.state = INIT_TEST;
    priv.should_produce_string =
        might_print_update(all) || (all.final_prediction_sink.size() > 0) || (all.raw_prediction != nullptr);
    priv.pred_string->str("");
    priv.test_action_sequence.clear();
    run_task(sch, ec_seq);

    // accumulate loss
    if (!is_test_ex) all.sd->update(ec_seq[0]->test_only, !is_test_ex, priv.test_loss, 1.f, priv.num_features);

    // generate output
    for (auto& sink : all.final_prediction_sink)
    { all.print_text_by_ref(sink.get(), priv.pred_string->str(), ec_seq[0]->tag); }

    if (all.raw_prediction != nullptr) all.print_text_by_ref(all.raw_prediction.get(), "", ec_seq[0]->tag);
  }

  // if we're not training, then we're done!
  if (!is_learn) return;
  if (is_test_ex || is_holdout_ex || ec_seq[0]->test_only || (!priv.all->training)) return;

  // SPEEDUP: if the oracle was never called, we can skip this!

  // do a pass over the data allowing oracle
  cdbg << "======================================== INIT TRAIN (" << priv.current_policy << ","
       << priv.read_example_last_pass << ") ========================================" << endl;

  priv.cache_hash_map.clear();
  reset_search_structure(priv);
  clear_memo_foreach_action(priv);
  priv.state = INIT_TRAIN;
  priv.active_uncertainty.clear();
  priv.train_trajectory.clear();  // this is where we'll store the training sequence
  run_task(sch, ec_seq);

  if (!ran_test)  // was  && !priv.ec_seq[0]->test_only) { but we know it's not test_only
    all.sd->update(ec_seq[0]->test_only, true, priv.test_loss, 1.f, priv.num_features);

  // if there's nothing to train on, we're done!
  if ((priv.loss_declared_cnt == 0) || (priv.t + priv.meta_t == 0) ||
      (priv.rollout_method == NO_ROLLOUT))  // TODO: make sure NO_ROLLOUT works with beam!
  { return; }

  // otherwise, we have some learn'in to do!
  cdbg << "======================================== LEARN (" << priv.current_policy << ","
       << priv.read_example_last_pass << ") ========================================" << endl;
  priv.T = priv.metatask ? priv.meta_t : priv.t;
  get_training_timesteps(priv, priv.timesteps);
  cdbg << "train_trajectory.size() = " << priv.train_trajectory.size() << ":\t";
  cdbg_print_array<scored_action>("", priv.train_trajectory);
  // cdbg << "memo_foreach_action = " << priv.memo_foreach_action << endl;
  for (size_t i = 0; i < priv.memo_foreach_action.size(); i++)
  {
    cdbg << "memo_foreach_action[" << i << "] = ";
    if (priv.memo_foreach_action[i])
      cdbg << *priv.memo_foreach_action[i];
    else
      cdbg << "null";
    cdbg << endl;
  }

  if (priv.cb_learner)
    priv.learn_losses.cb.costs.clear();
  else
    priv.learn_losses.cs.costs.clear();

  for (size_t tid = 0; tid < priv.timesteps.size(); tid++)
  {
    cdbg << "timestep = " << priv.timesteps[tid] << " [" << tid << "/" << priv.timesteps.size() << "]" << endl;

    if (priv.metatask && !priv.memo_foreach_action[tid])
    {
      cdbg << "skipping because it looks like this was overridden by metatask" << endl;
      continue;
    }

    priv.learn_ec_ref = nullptr;
    priv.learn_ec_ref_cnt = 0;

    reset_search_structure(priv);  // TODO remove this?
    bool skipped_all_actions = true;
    priv.learn_a_idx = 0;
    priv.done_with_all_actions = false;
    // for each action, roll out to get a loss
    while (!priv.done_with_all_actions)
    {
      priv.learn_t = priv.timesteps[tid];
      advance_from_known_actions(priv);
      if (priv.done_with_all_actions) break;

      skipped_all_actions = false;
      reset_search_structure(priv);

      priv.state = LEARN;
      priv.learn_t = priv.timesteps[tid];
      cdbg << "-------------------------------------------------------------------------------------" << endl;
      cdbg << "learn_t = " << priv.learn_t << ", learn_a_idx = " << priv.learn_a_idx << endl;
      run_task(sch, ec_seq);
      float this_loss = priv.learn_loss;
      cs_cost_push_back(priv.cb_learner, priv.learn_losses,
          priv.is_ldf ? static_cast<uint32_t>(priv.learn_a_idx - 1) : static_cast<uint32_t>(priv.learn_a_idx),
          this_loss);
      //                          (priv.learn_allowed_actions.size() > 0) ?
      //                          priv.learn_allowed_actions[priv.learn_a_idx-1] : priv.is_ldf ? (priv.learn_a_idx-1) :
      //                          (priv.learn_a_idx),
      //                           priv.learn_loss);
    }
    if (priv.active_csoaa_verify > 0.)
      verify_active_csoaa(
          priv.learn_losses.cs, priv.active_known[priv.learn_t], ec_seq[0]->example_counter, priv.active_csoaa_verify);

    if (skipped_all_actions)
    {
      reset_search_structure(priv);
      priv.state = LEARN;
      priv.learn_t = priv.timesteps[tid];
      priv.force_setup_ec_ref = true;
      cdbg << "<<<<<" << endl;
      cdbg << "skipped all actions; learn_t = " << priv.learn_t << ", learn_a_idx = " << priv.learn_a_idx << endl;
      run_task(sch, ec_seq);  // TODO: i guess we can break out of this early
      cdbg << ">>>>>" << endl;
    }
    else
      cdbg << "didn't skip all actions" << endl;

    // now we can make a training example
    if (priv.learn_allowed_actions.size() > 0)
    {
      for (size_t i = 0; i < priv.learn_allowed_actions.size(); i++)
      { priv.learn_losses.cs.costs[i].class_index = priv.learn_allowed_actions[i]; }
    }
    // float min_loss = 0.;
    // if (priv.metatask)
    //  for (size_t aid=0; aid<priv.memo_foreach_action[tid]->size(); aid++)
    //    min_loss = std::min(min_loss, priv.memo_foreach_action[tid]->get(aid).cost);
    cdbg << "priv.learn_losses = [";
    for (auto& wc : priv.learn_losses.cs.costs) cdbg << " " << wc.class_index << ":" << wc.x;
    cdbg << " ]" << endl;
    cdbg << "gte" << endl;
    generate_training_example(priv, priv.learn_losses, 1., true);  // , min_loss);  // TODO: weight
    if (!priv.examples_dont_change)
    {
      for (auto& ex : priv.learn_ec_copy)
      {
        // Reset the state of the polylabel
        ex.l = polylabel{};
      }
    }
    if (priv.cb_learner)
      priv.learn_losses.cb.costs.clear();
    else
      priv.learn_losses.cs.costs.clear();
  }

  if (priv.active_csoaa && (priv.save_every_k_runs > 1))
  {
    size_t prev_num = priv.num_calls_to_run_previous / priv.save_every_k_runs;
    size_t this_num = priv.num_calls_to_run / priv.save_every_k_runs;
    if (this_num > prev_num) save_predictor(all, all.final_regressor_name, this_num);
    priv.num_calls_to_run_previous = priv.num_calls_to_run;
  }
}

void inline adjust_auto_condition(search_private& priv)
{
  if (priv.auto_condition_features)
  {
    // turn off auto-condition if it's irrelevant
    if ((priv.history_length == 0) || (priv.acset.feature_value == 0.f))
    {
      logger::errlog_warn("warning: turning off AUTO_CONDITION_FEATURES because settings make it useless");
      priv.auto_condition_features = false;
    }
  }
}

template <bool is_learn>
void do_actual_learning(search& sch, base_learner& base, multi_ex& ec_seq)
{
  if (ec_seq.size() == 0) return;  // nothing to do :)

  bool is_test_ex = false;
  bool is_holdout_ex = false;

  search_private& priv = *sch.priv;
  priv.offset = ec_seq[0]->ft_offset;
  priv.base_learner = &base;

  adjust_auto_condition(priv);
  priv.read_example_last_id = ec_seq[ec_seq.size() - 1]->example_counter;

  // hit_new_pass true would have already triggered a printout
  // finish_example(multi_ex).  so we can reset hit_new_pass here
  priv.hit_new_pass = false;

  for (size_t i = 0; i < ec_seq.size(); i++)
  {
    is_test_ex |= priv.label_is_test(ec_seq[i]->l);
    is_holdout_ex |= ec_seq[i]->test_only;
    if (is_test_ex && is_holdout_ex) break;
  }

  if (priv.task->run_setup) priv.task->run_setup(sch, ec_seq);

  // if we're going to have to print to the screen, generate the "truth" std::string
  cdbg << "======================================== GET TRUTH STRING (" << priv.current_policy << ","
       << priv.read_example_last_pass << ") ========================================" << endl;
  if (might_print_update(*priv.all))
  {
    if (is_test_ex)
      priv.truth_string->str("**test**");
    else
    {
      reset_search_structure(*sch.priv);
      priv.state = GET_TRUTH_STRING;
      priv.should_produce_string = true;
      priv.truth_string->str("");
      run_task(sch, ec_seq);
    }
  }

  add_neighbor_features(priv, ec_seq);
  train_single_example<is_learn>(sch, is_test_ex, is_holdout_ex, ec_seq);
  del_neighbor_features(priv, ec_seq);

  if (priv.task->run_takedown) priv.task->run_takedown(sch, ec_seq);
}

void end_pass(search& sch)
{
  search_private& priv = *sch.priv;
  vw* all = priv.all;
  priv.hit_new_pass = true;
  priv.read_example_last_pass++;
  priv.passes_since_new_policy++;

  if (priv.passes_since_new_policy >= priv.passes_per_policy)
  {
    priv.passes_since_new_policy = 0;
    if (all->training) priv.current_policy++;
    if (priv.current_policy > priv.total_number_of_policies)
    {
      logger::errlog_error("internal error (bug): too many policies; not advancing");
      priv.current_policy = priv.total_number_of_policies;
    }
    // reset search_trained_nb_policies in options_from_file so it is saved to regressor file later
    // TODO work out a better system to update state that will be saved in the model.
    all->options->replace("search_trained_nb_policies", std::to_string(priv.current_policy));
    all->options->get_typed_option<uint32_t>("search_trained_nb_policies").value(priv.current_policy);
  }
}

void finish_multiline_example(vw& all, search& sch, multi_ex& ec_seq)
{
  print_update(*sch.priv);
  VW::finish_example(all, ec_seq);
}

void end_examples(search& sch)
{
  search_private& priv = *sch.priv;
  vw* all = priv.all;

  if (all->training)
  {
    // TODO work out a better system to update state that will be saved in the model.
    // Dig out option and change it in case we already loaded a predictor which had a value stored for
    // --search_trained_nb_policies
    auto val = (priv.passes_since_new_policy == 0) ? priv.current_policy : (priv.current_policy + 1);
    all->options->replace("search_trained_nb_policies", std::to_string(val));
    all->options->get_typed_option<uint32_t>("search_trained_nb_policies").value(val);
    // Dig out option and change it in case we already loaded a predictor which had a value stored for
    // --search_total_nb_policies
    all->options->replace("search_total_nb_policies", std::to_string(priv.total_number_of_policies));
    all->options->get_typed_option<uint32_t>("search_total_nb_policies").value(priv.total_number_of_policies);
  }
}

bool mc_label_is_test(polylabel& lab) { return MC::test_label(lab.multi); }

void search_initialize(vw* all, search& sch)
{
  search_private& priv = *sch.priv;  // priv is zero initialized by default
  priv.all = all;
  priv._random_state = all->get_random_state();

  priv.active_csoaa = false;
  priv.label_is_test = mc_label_is_test;

  priv.num_learners = 1;
  priv.state = INITIALIZE;
  priv.mix_per_roll_policy = -2;

  priv.pred_string = VW::make_unique<std::stringstream>();
  priv.truth_string = VW::make_unique<std::stringstream>();
  priv.bad_string_stream = VW::make_unique<std::stringstream>();
  priv.bad_string_stream->clear(priv.bad_string_stream->badbit);

  priv.rollout_method = MIX_PER_ROLL;
  priv.rollin_method = MIX_PER_ROLL;

  priv.allow_current_policy = true;
  priv.adaptive_beta = true;

  priv.total_number_of_policies = 1;

  priv.acset.max_bias_ngram_length = 1;

  priv.acset.feature_value = 1.;

  scored_action sa(static_cast<action>(-1), 0.);

  sch.task_data = nullptr;

  priv.active_uncertainty.clear();
  priv.active_known.clear();

  CS::default_label(priv.empty_cs_label);

  priv.rawOutputStringStream = VW::make_unique<std::stringstream>();
}

void ensure_param(float& v, float lo, float hi, float def, const char* str)
{
  if ((v < lo) || (v > hi))
  {
    logger::errlog_warn(str);
    v = def;
  }
}

void handle_condition_options(vw& all, auto_condition_settings& acset)
{
  option_group_definition new_options("Search Auto-conditioning Options");
  new_options.add(make_option("search_max_bias_ngram_length", acset.max_bias_ngram_length)
                      .keep()
                      .default_value(1)
                      .help("add a \"bias\" feature for each ngram up to and including this length. eg., if it's 1 "
                            "(default), then you get a single feature for each conditional"));
  new_options.add(make_option("search_max_quad_ngram_length", acset.max_quad_ngram_length)
                      .keep()
                      .default_value(0)
                      .help("add bias *times* input features for each ngram up to and including this length (def: 0)"));
  new_options.add(make_option("search_condition_feature_value", acset.feature_value)
                      .keep()
                      .default_value(1.f)
                      .help("how much weight should the conditional features get? (def: 1.)"));
  new_options.add(make_option("search_use_passthrough_repr", acset.use_passthrough_repr)
                      .keep()
                      .help("should we use lower-level reduction _internal state_ as additional features? (def: no)"));
  all.options->add_and_parse(new_options);
}

void search_finish(search& sch)
{
  search_private& priv = *sch.priv;
  cdbg << "search_finish" << endl;

  if (priv.active_csoaa) logger::errlog_info("search calls to run = {}", priv.num_calls_to_run);

  if (priv.task->finish) priv.task->finish(sch);
  if (priv.metatask && priv.metatask->finish) priv.metatask->finish(sch);
}

std::vector<CS::label> read_allowed_transitions(action A, const char* filename)
{
  FILE* f;
  if (VW::file_open(&f, filename, "r") != 0)
    THROW("error: could not read file " << filename << " (" << VW::strerror_to_string(errno)
                                        << "); assuming all transitions are valid");

  bool* bg = calloc_or_throw<bool>((static_cast<size_t>(A + 1)) * (A + 1));
  int rd, from, to, count = 0;
  while ((rd = fscanf_s(f, "%d:%d", &from, &to)) > 0)
  {
    if ((from < 0) || (from > static_cast<int>(A)))
    {
      logger::errlog_warn("warning: ignoring transition from {0} because it's out of the range [0,{1}]", from, A);
    }
    if ((to < 0) || (to > static_cast<int>(A)))
    {
      logger::errlog_warn("warning: ignoring transition to {0} because it's out of the range [0,{1}]", to, A);
    }
    bg[from * (A + 1) + to] = true;
    count++;
  }
  fclose(f);

  std::vector<CS::label> allowed;

  // from
  for (size_t i = 0; i < A; i++)
  {
    v_array<CS::wclass> costs;

    // to
    for (size_t j = 0; j < A; j++)
      if (bg[i * (A + 1) + j])
      {
        CS::wclass c = {FLT_MAX, static_cast<action>(j), 0., 0.};
        costs.push_back(c);
      }

    CS::label ld = {costs};
    allowed.push_back(ld);
  }
  free(bg);

  logger::errlog_info("read {0} allowed transitions from {1}", count, filename);

  return allowed;
}

void parse_neighbor_features(VW::string_view nf_strview, search& sch)
{
  search_private& priv = *sch.priv;
  priv.neighbor_features.clear();
  if (nf_strview.empty()) return;

  std::vector<VW::string_view> cmd;
  size_t end_idx = 0;
  bool reached_end = false;
  while (!reached_end)
  {
    end_idx = nf_strview.find(',');
    VW::string_view strview = nf_strview.substr(0, end_idx);
    // If we haven't reached the end yet, slice off the piece we're currently parsing
    if (end_idx != VW::string_view::npos) { nf_strview.remove_prefix(end_idx + 1); }
    else
    {
      reached_end = true;
    }

    cmd.clear();
    tokenize(':', strview, cmd, true);
    int32_t posn = 0;
    char ns = ' ';
    if (cmd.size() == 1)
    {
      posn = int_of_string(cmd[0]);
      ns = ' ';
    }
    else if (cmd.size() == 2)
    {
      posn = int_of_string(cmd[0]);
      ns = (!cmd[1].empty()) ? cmd[1].front() : ' ';
    }
    else
    {
      logger::errlog_warn("warning: ignoring malformed neighbor specification: '{}'", strview);
    }
    int32_t enc = (posn << 24) | (ns & 0xFF);
    priv.neighbor_features.push_back(enc);
  }
}

base_learner* setup(options_i& options, vw& all)
{
  free_ptr<search> sch = scoped_calloc_or_throw<search>();
  search_private& priv = *sch->priv;
  std::string task_string;
  std::string metatask_string;
  std::string interpolation_string = "data";
  std::string neighbor_features_string;
  std::string rollout_string = "mix_per_state";
  std::string rollin_string = "mix_per_state";

  uint32_t search_trained_nb_policies;
  std::string search_allowed_transitions;

  priv.A = 1;
  option_group_definition new_options("Search options");
  new_options.add(
      make_option("search", priv.A).keep().help("Use learning to search, argument=maximum action id or 0 for LDF"));
  new_options.add(make_option("search_task", task_string)
                      .keep()
                      .necessary()
                      .help("the search task (use \"--search_task list\" to get a list of available tasks)"));
  new_options.add(
      make_option("search_metatask", metatask_string)
          .keep()
          .help("the search metatask (use \"--search_metatask list\" to get a list of available metatasks)"));
  new_options.add(make_option("search_interpolation", interpolation_string)
                      .keep()
                      .help("at what level should interpolation happen? [*data|policy]"));
  new_options.add(
      make_option("search_rollout", rollout_string)
          .help("how should rollouts be executed?           [policy|oracle|*mix_per_state|mix_per_roll|none]"));
  new_options.add(make_option("search_rollin", rollin_string)
                      .help("how should past trajectories be generated? [policy|oracle|*mix_per_state|mix_per_roll]"));
  new_options.add(make_option("search_passes_per_policy", priv.passes_per_policy)
                      .default_value(1)
                      .help("number of passes per policy (only valid for search_interpolation=policy)"));
  new_options.add(make_option("search_beta", priv.beta)
                      .default_value(0.5f)
                      .help("interpolation rate for policies (only valid for search_interpolation=policy)"));
  new_options.add(make_option("search_alpha", priv.alpha)
                      .default_value(1e-10f)
                      .help("annealed beta = 1-(1-alpha)^t (only valid for search_interpolation=data)"));
  new_options.add(make_option("search_total_nb_policies", priv.total_number_of_policies)
                      .help("if we are going to train the policies through multiple separate calls to vw, we need to "
                            "specify this parameter and tell vw how many policies are eventually going to be trained"));
  new_options.add(make_option("search_trained_nb_policies", search_trained_nb_policies)
                      .help("the number of trained policies in a file"));
  new_options.add(make_option("search_allowed_transitions", search_allowed_transitions)
                      .help("read file of allowed transitions [def: all transitions are allowed]"));
  new_options.add(make_option("search_subsample_time", priv.subsample_timesteps)
                      .help("instead of training at all timesteps, use a subset. if value in (0,1), train on a random "
                            "v%. if v>=1, train on precisely v steps per example, if v<=-1, use active learning"));
  new_options.add(
      make_option("search_neighbor_features", neighbor_features_string)
          .keep()
          .help("copy features from neighboring lines. argument looks like: '-1:a,+2' meaning copy previous line "
                "namespace a and next next line from namespace _unnamed_, where ',' separates them"));
  new_options.add(make_option("search_rollout_num_steps", priv.rollout_num_steps)
                      .help("how many calls of \"loss\" before we stop really predicting on rollouts and switch to "
                            "oracle (default means \"infinite\")"));
  new_options.add(make_option("search_history_length", priv.history_length)
                      .keep()
                      .default_value(1)
                      .help("some tasks allow you to specify how much history their depend on; specify that here"));
  new_options.add(make_option("search_no_caching", priv.no_caching)
                      .help("turn off the built-in caching ability (makes things slower, but technically more safe)"));
  new_options.add(
      make_option("search_xv", priv.xv).help("train two separate policies, alternating prediction/learning"));
  new_options.add(make_option("search_perturb_oracle", priv.perturb_oracle)
                      .default_value(0.f)
                      .help("perturb the oracle on rollin with this probability"));
  new_options.add(make_option("search_linear_ordering", priv.linear_ordering)
                      .help("insist on generating examples in linear order (def: hoopla permutation)"));
  new_options.add(make_option("search_active_verify", priv.active_csoaa_verify)
                      .help("verify that active learning is doing the right thing (arg = multiplier, should be = "
                            "cost_range * range_c)"));
  new_options.add(make_option("search_save_every_k_runs", priv.save_every_k_runs).help("save model every k runs"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  search_initialize(&all, *sch.get());

  parse_neighbor_features(neighbor_features_string, *sch.get());

  if (interpolation_string.compare("data") == 0)  // run as dagger
  {
    priv.adaptive_beta = true;
    priv.allow_current_policy = true;
    priv.passes_per_policy = all.numpasses;
    if (priv.current_policy > 1) priv.current_policy = 1;
  }
  else if (interpolation_string.compare("policy") == 0)
    ;
  else
    THROW("error: --search_interpolation must be 'data' or 'policy'");

  if ((rollout_string.compare("policy") == 0) || (rollout_string.compare("learn") == 0))
    priv.rollout_method = POLICY;
  else if ((rollout_string.compare("oracle") == 0) || (rollout_string.compare("ref") == 0))
    priv.rollout_method = ORACLE;
  else if ((rollout_string.compare("mix_per_state") == 0))
    priv.rollout_method = MIX_PER_STATE;
  else if ((rollout_string.compare("mix_per_roll") == 0) || (rollout_string.compare("mix") == 0))
    priv.rollout_method = MIX_PER_ROLL;
  else if ((rollout_string.compare("none") == 0))
  {
    priv.rollout_method = NO_ROLLOUT;
    priv.no_caching = true;
  }
  else
    THROW("error: --search_rollout must be 'learn', 'ref', 'mix', 'mix_per_state' or 'none'");

  if ((rollin_string.compare("policy") == 0) || (rollin_string.compare("learn") == 0))
    priv.rollin_method = POLICY;
  else if ((rollin_string.compare("oracle") == 0) || (rollin_string.compare("ref") == 0))
    priv.rollin_method = ORACLE;
  else if ((rollin_string.compare("mix_per_state") == 0))
    priv.rollin_method = MIX_PER_STATE;
  else if ((rollin_string.compare("mix_per_roll") == 0) || (rollin_string.compare("mix") == 0))
    priv.rollin_method = MIX_PER_ROLL;
  else
    THROW("error: --search_rollin must be 'learn', 'ref', 'mix' or 'mix_per_state'");

  // check if the base learner is contextual bandit, in which case, we dont rollout all actions.
  if (options.was_supplied("cb"))
  {
    priv.cb_learner = true;
    CB::cb_label.default_label(&priv.allowed_actions_cache);
    priv.learn_losses.cb.costs.clear();
    priv.gte_label.cb.costs.clear();
  }
  else
  {
    priv.cb_learner = false;
    CS::cs_label.default_label(&priv.allowed_actions_cache);
    priv.learn_losses.cs.costs.clear();
    priv.gte_label.cs.costs.clear();
  }

  ensure_param(priv.beta, 0.0, 1.0, 0.5, "warning: search_beta must be in (0,1); resetting to 0.5");
  ensure_param(priv.alpha, 0.0, 1.0, 1e-10f, "warning: search_alpha must be in (0,1); resetting to 1e-10");

  priv.num_calls_to_run = 0;

  // compute total number of policies we will have at end of training
  // we add current_policy for cases where we start from an initial set of policies loaded through -i option
  uint32_t tmp_number_of_policies = priv.current_policy;
  if (all.training)
    tmp_number_of_policies +=
        static_cast<int>(ceil((static_cast<float>(all.numpasses)) / (static_cast<float>(priv.passes_per_policy))));

  // the user might have specified the number of policies that will eventually be trained through multiple vw calls,
  // so only set total_number_of_policies to computed value if it is larger
  cdbg << "current_policy=" << priv.current_policy << " tmp_number_of_policies=" << tmp_number_of_policies
       << " total_number_of_policies=" << priv.total_number_of_policies << endl;
  if (tmp_number_of_policies > priv.total_number_of_policies)
  {
    priv.total_number_of_policies = tmp_number_of_policies;
    if (priv.current_policy >
        0)  // we loaded a file but total number of policies didn't match what is needed for training
      logger::errlog_warn("warning: you're attempting to train more classifiers than was allocated initially. "
                          "Likely to cause bad performance.");
  }

  // current policy currently points to a new policy we would train
  // if we are not training and loaded a bunch of policies for testing, we need to subtract 1 from current policy
  // so that we only use those loaded when testing (as run_prediction is called with allow_current to true)
  if (!all.training && priv.current_policy > 0) priv.current_policy--;

  all.options->replace("search_trained_nb_policies", std::to_string(priv.current_policy));
  all.options->get_typed_option<uint32_t>("search_trained_nb_policies").value(priv.current_policy);

  all.options->replace("search_total_nb_policies", std::to_string(priv.total_number_of_policies));
  all.options->get_typed_option<uint32_t>("search_total_nb_policies").value(priv.total_number_of_policies);

  cdbg << "search current_policy = " << priv.current_policy
       << " total_number_of_policies = " << priv.total_number_of_policies << endl;

  if (task_string.compare("list") == 0)
  {
    // command line action, output directly to cerr
    std::cerr << endl << "available search tasks:" << endl;
    for (search_task** mytask = all_tasks; *mytask != nullptr; mytask++)
      std::cerr << "  " << (*mytask)->task_name << endl;
    std::cerr << endl;
    exit(0);
  }
  if (metatask_string.compare("list") == 0)
  {
    // command line action, output directly to cerr
    std::cerr << endl << "available search metatasks:" << endl;
    for (search_metatask** mytask = all_metatasks; *mytask != nullptr; mytask++)
      std::cerr << "  " << (*mytask)->metatask_name << endl;
    std::cerr << endl;
    exit(0);
  }
  for (search_task** mytask = all_tasks; *mytask != nullptr; mytask++)
    if (task_string.compare((*mytask)->task_name) == 0)
    {
      priv.task = *mytask;
      sch->task_name = (*mytask)->task_name;
      break;
    }
  if (priv.task == nullptr)
  {
    if (!options.was_supplied("help"))
      THROW("fail: unknown task for --search_task '" << task_string << "'; use --search_task list to get a list");
  }
  priv.metatask = nullptr;
  for (search_metatask** mytask = all_metatasks; *mytask != nullptr; mytask++)
    if (metatask_string.compare((*mytask)->metatask_name) == 0)
    {
      priv.metatask = *mytask;
      sch->metatask_name = (*mytask)->metatask_name;
      break;
    }
  all.example_parser->emptylines_separate_examples = true;

  if (!options.was_supplied("csoaa") && !options.was_supplied("cs_active") && !options.was_supplied("csoaa_ldf") &&
      !options.was_supplied("wap_ldf") && !options.was_supplied("cb"))
  { options.insert("csoaa", std::to_string(priv.A)); }

  priv.active_csoaa = options.was_supplied("cs_active");
  priv.active_csoaa_verify = -1.;
  if (options.was_supplied("search_active_verify"))
    if (!priv.active_csoaa) THROW("cannot use --search_active_verify without using --cs_active");

  cdbg << "active_csoaa = " << priv.active_csoaa << ", active_csoaa_verify = " << priv.active_csoaa_verify << endl;

  base_learner* base = setup_base(*all.options, all);

  // default to OAA labels unless the task wants to override this (which they can do in initialize)
  all.example_parser->lbl_parser = MC::mc_label;

  if (priv.task && priv.task->initialize) priv.task->initialize(*sch.get(), priv.A, options);
  if (priv.metatask && priv.metatask->initialize) priv.metatask->initialize(*sch.get(), priv.A, options);
  priv.meta_t = 0;

  if (options.was_supplied("search_allowed_transitions"))
    read_allowed_transitions(static_cast<action>(priv.A), search_allowed_transitions.c_str());

  // set up auto-history (used to only do this if AUTO_CONDITION_FEATURES was on, but that doesn't work for hooktask)
  handle_condition_options(all, priv.acset);

  if (!priv.allow_current_policy)  // if we're not dagger
    all.check_holdout_every_n_passes = priv.passes_per_policy;

  all.searchstr = sch.get();

  priv.start_clock_time = clock();

  if (priv.xv) priv.num_learners *= 3;

  cdbg << "num_learners = " << priv.num_learners << endl;

  learner<search, multi_ex>& l = init_learner(sch, make_base(*base), do_actual_learning<true>,
      do_actual_learning<false>, priv.total_number_of_policies * priv.num_learners, all.get_setupfn_name(setup), true);

  l.set_finish_example(finish_multiline_example);
  l.set_end_examples(end_examples);
  l.set_finish(search_finish);
  l.set_end_pass(end_pass);
  return make_base(l);
}

float action_hamming_loss(action a, const action* A, size_t sz)
{
  if (sz == 0) return 0.;  // latent variables have zero loss
  for (size_t i = 0; i < sz; i++)
    if (a == A[i]) return 0.;
  return 1.;
}

float action_cost_loss(action a, const action* act, const float* costs, size_t sz)
{
  if (act == nullptr) return costs[a - 1];
  for (size_t i = 0; i < sz; i++)
    if (act[i] == a) return costs[i];
  THROW("action_cost_loss got action that wasn't allowed: " << a);
}

// the interface:
bool search::is_ldf() { return priv->is_ldf; }

action search::predict(example& ec, ptag mytag, const action* oracle_actions, size_t oracle_actions_cnt,
    const ptag* condition_on, const char* condition_on_names, const action* allowed_actions, size_t allowed_actions_cnt,
    const float* allowed_actions_cost, size_t learner_id, float weight)
{
  float a_cost = 0.;
  action a = search_predict(*priv, &ec, 1, mytag, oracle_actions, oracle_actions_cnt, condition_on, condition_on_names,
      allowed_actions, allowed_actions_cnt, allowed_actions_cost, learner_id, a_cost, weight);
  if (priv->state == INIT_TEST) priv->test_action_sequence.push_back(a);
  if (mytag != 0)
  {
    if (mytag < priv->ptag_to_action.size())
    {
      cdbg << "delete_v at " << mytag << endl;
      if (priv->ptag_to_action[mytag].repr != nullptr)
      {
        delete priv->ptag_to_action[mytag].repr;
        priv->ptag_to_action[mytag].repr = nullptr;
      }
    }
    if (priv->acset.use_passthrough_repr)
    {
      assert((mytag >= priv->ptag_to_action.size()) || (priv->ptag_to_action[mytag].repr == nullptr));
      push_at(priv->ptag_to_action, action_repr(a, &(priv->last_action_repr)), mytag);
    }
    else
      push_at(priv->ptag_to_action, action_repr(a, (features*)nullptr), mytag);
    cdbg << "push_at " << mytag << endl;
  }
  if (priv->auto_hamming_loss)
    loss(priv->use_action_costs ? action_cost_loss(a, allowed_actions, allowed_actions_cost, allowed_actions_cnt)
                                : action_hamming_loss(a, oracle_actions, oracle_actions_cnt));
  cdbg << "predict returning " << a << endl;
  return a;
}

action search::predictLDF(example* ecs, size_t ec_cnt, ptag mytag, const action* oracle_actions,
    size_t oracle_actions_cnt, const ptag* condition_on, const char* condition_on_names, size_t learner_id,
    float weight)
{
  float a_cost = 0.;
  // TODO: action costs for ldf
  action a = search_predict(*priv, ecs, ec_cnt, mytag, oracle_actions, oracle_actions_cnt, condition_on,
      condition_on_names, nullptr, 0, nullptr, learner_id, a_cost, weight);
  if (priv->state == INIT_TEST) priv->test_action_sequence.push_back(a);

  // If there is a shared example (example header), then action "1" is at index 1, but otherwise
  // action "1" is at index 0. Map action to its appropriate index. In particular, this fixes an
  // issue where the predicted action is the last, and there is no example header, causing an index
  // beyond the end of the array (usually resulting in a segfault at some point.)
  size_t action_index = (a - COST_SENSITIVE::ec_is_example_header(ecs[0])) ? 0 : 1;

  if ((mytag != 0) && ecs[action_index].l.cs.costs.size() > 0)
  {
    if (mytag < priv->ptag_to_action.size())
    {
      cdbg << "delete_v at " << mytag << endl;
      if (priv->ptag_to_action[mytag].repr != nullptr)
      {
        delete priv->ptag_to_action[mytag].repr;
        priv->ptag_to_action[mytag].repr = nullptr;
      }
    }
    push_at(priv->ptag_to_action, action_repr(ecs[a].l.cs.costs[0].class_index, &(priv->last_action_repr)), mytag);
  }
  if (priv->auto_hamming_loss) loss(action_hamming_loss(a, oracle_actions, oracle_actions_cnt));  // TODO: action costs
  cdbg << "predict returning " << a << endl;
  return a;
}

void search::loss(float loss) { search_declare_loss(*this->priv, loss); }

bool search::predictNeedsExample() { return search_predictNeedsExample(*this->priv); }

std::stringstream& search::output()
{
  if (!this->priv->should_produce_string)
    return *(this->priv->bad_string_stream);
  else if (this->priv->state == GET_TRUTH_STRING)
    return *(this->priv->truth_string);
  else
    return *(this->priv->pred_string);
}

void search::set_options(uint32_t opts)
{
  if (this->priv->all->vw_is_main && (this->priv->state != INITIALIZE))
    logger::errlog_warn("warning: task should not set options except in initialize function!");
  if ((opts & AUTO_CONDITION_FEATURES) != 0) this->priv->auto_condition_features = true;
  if ((opts & AUTO_HAMMING_LOSS) != 0) this->priv->auto_hamming_loss = true;
  if ((opts & EXAMPLES_DONT_CHANGE) != 0) this->priv->examples_dont_change = true;
  if ((opts & IS_LDF) != 0) this->priv->is_ldf = true;
  if ((opts & NO_CACHING) != 0) this->priv->no_caching = true;
  if ((opts & ACTION_COSTS) != 0) this->priv->use_action_costs = true;

  if (this->priv->is_ldf && this->priv->use_action_costs)
    THROW("using LDF and actions costs is not yet implemented; turn off action costs");  // TODO fix

  if (this->priv->use_action_costs && (this->priv->rollout_method != NO_ROLLOUT))
    logger::errlog_warn(
      "warning: task is designed to use rollout costs, but this only works when --search_rollout none is specified"
    );
}

void search::set_label_parser(label_parser& lp, bool (*is_test)(polylabel&))
{
  if (this->priv->all->vw_is_main && (this->priv->state != INITIALIZE))
    logger::errlog_warn("warning: task should not set label parser except in initialize function!");
  this->priv->all->example_parser->lbl_parser = lp;
  this->priv->all->example_parser->lbl_parser.test_label = reinterpret_cast<bool (*)(polylabel*)>(is_test);
  this->priv->label_is_test = is_test;
}

void search::get_test_action_sequence(std::vector<action>& V)
{
  V.clear();
  for (size_t i = 0; i < this->priv->test_action_sequence.size(); i++) V.push_back(this->priv->test_action_sequence[i]);
}

void search::set_num_learners(size_t num_learners) { this->priv->num_learners = num_learners; }

uint64_t search::get_mask() { return this->priv->all->weights.mask(); }
size_t search::get_stride_shift() { return this->priv->all->weights.stride_shift(); }
uint32_t search::get_history_length() { return static_cast<uint32_t>(this->priv->history_length); }

std::string search::pretty_label(action a)
{
  if (this->priv->all->sd->ldict)
  {
    auto sv = this->priv->all->sd->ldict->get(a);
    return sv.to_string();
  }
  else
  {
    std::ostringstream os;
    os << a;
    return os.str();
  }
}

vw& search::get_vw_pointer_unsafe() { return *this->priv->all; }
void search::set_force_oracle(bool force) { this->priv->force_oracle = force; }

// predictor implementation
predictor::predictor(search& sch, ptag my_tag)
    : is_ldf(false)
    , my_tag(my_tag)
    , ec(nullptr)
    , ec_cnt(0)
    , ec_alloced(false)
    , weight(1.)
    , learner_id(0)
    , sch(sch)
{
}

void predictor::free_ec()
{
  if (ec_alloced)
  {
    if (is_ldf) { VW::dealloc_examples(ec, ec_cnt); }
    else
    {
      VW::dealloc_examples(ec, 1);
    }
  }
}

predictor::~predictor()
{
  free_ec();
}

predictor& predictor::reset()
{
  this->erase_oracles();
  this->erase_alloweds();
  condition_on_tags.clear();
  condition_on_names.clear();
  free_ec();
  return *this;
}

predictor& predictor::set_input(example& input_example)
{
  free_ec();
  is_ldf = false;
  ec = &input_example;
  ec_cnt = 1;
  ec_alloced = false;
  return *this;
}

predictor& predictor::set_input(example* input_example, size_t input_length)
{
  free_ec();
  is_ldf = true;
  ec = input_example;
  ec_cnt = input_length;
  ec_alloced = false;
  return *this;
}

void predictor::set_input_length(size_t input_length)
{
  is_ldf = true;
  if (ec_alloced)
  {
    for (size_t i = ec_cnt; ec_cnt < input_length; ++i) ec[i].~example();
    example* temp = static_cast<example*>(realloc(ec, input_length * sizeof(example)));
    if (temp != nullptr)
      ec = temp;
    else
      THROW("realloc failed in search.cc");
  }
  else
    ec = VW::alloc_examples(input_length);
  ec_cnt = input_length;
  ec_alloced = true;
}
void predictor::set_input_at(size_t posn, example& ex)
{
  if (!ec_alloced) THROW("call to set_input_at without previous call to set_input_length");

  if (posn >= ec_cnt)
    THROW("call to set_input_at with too large a position: posn (" << posn << ") >= ec_cnt(" << ec_cnt << ")");

  VW::copy_example_data_with_label(ec + posn, &ex);
}

predictor& predictor::erase_oracles()
{
  oracle_actions.clear();
  return *this;
}

predictor& predictor::add_oracle(action a)
{
  oracle_actions.push_back(a);
  return *this;
}

predictor& predictor::add_oracle(action* a, size_t action_count)
{
  for (size_t i = 0; i < action_count; i++) { oracle_actions.push_back(*(a + i)); }
  return *this;
}

predictor& predictor::add_oracle(v_array<action>& a)
{
  for (const auto& item : a) { oracle_actions.push_back(item); }
  return *this;
}

predictor& predictor::set_oracle(action a)
{
  oracle_actions.clear();
  return add_oracle(a);
}

predictor& predictor::set_oracle(action* a, size_t action_count)
{
  oracle_actions.clear();
  return add_oracle(a, action_count);
}

predictor& predictor::set_oracle(v_array<action>& a)
{
  oracle_actions.clear();
  return add_oracle(a);
}

predictor& predictor::set_weight(float w)
{
  weight = w;
  return *this;
}

predictor& predictor::erase_alloweds()
{
  allowed_actions.clear();
  allowed_actions_cost.clear();
  return *this;
}

predictor& predictor::add_allowed(action a)
{
  allowed_actions.push_back(a);
  return *this;
}
predictor& predictor::add_allowed(action* a, size_t action_count)
{
  for (size_t i = 0; i < action_count; i++) { allowed_actions.push_back(*(a + i)); }
  return *this;
}
predictor& predictor::add_allowed(v_array<action>& a)
{
  for (const auto& item : a) { allowed_actions.push_back(item); }
  return *this;
}

predictor& predictor::set_allowed(action a)
{
  allowed_actions.clear();
  return add_allowed(a);
}

predictor& predictor::set_allowed(action* a, size_t action_count)
{
  allowed_actions.clear();
  return add_allowed(a, action_count);
}

predictor& predictor::set_allowed(v_array<action>& a)
{
  allowed_actions.clear();
  return add_allowed(a);
}

predictor& predictor::add_allowed(action a, float cost)
{
  allowed_actions_cost.push_back(cost);
  allowed_actions.push_back(a);
  return *this;
}

predictor& predictor::add_allowed(action* a, float* costs, size_t action_count)
{
  if (costs != nullptr)
  {
    for (size_t i = 0; i < action_count; i++) { allowed_actions_cost.push_back(*(costs + i)); }
  }
  if (a != nullptr)
  {
    for (size_t i = 0; i < action_count; i++) { allowed_actions.push_back(*(a + i)); }
  }
  return *this;
}

predictor& predictor::add_allowed(std::vector<std::pair<action, float>>& a)
{
  for (const auto& item : a)
  {
    allowed_actions.push_back(item.first);
    allowed_actions_cost.push_back(item.second);
  }
  return *this;
}

predictor& predictor::set_allowed(action a, float cost)
{
  allowed_actions_cost.clear();
  allowed_actions.clear();
  return add_allowed(a, cost);
}

predictor& predictor::set_allowed(action* a, float* costs, size_t action_count)
{
  allowed_actions_cost.clear();
  allowed_actions.clear();
  return add_allowed(a, costs, action_count);
}

predictor& predictor::set_allowed(std::vector<std::pair<action, float>>& a)
{
  erase_alloweds();
  return add_allowed(a);
}

predictor& predictor::add_condition(ptag tag, char name)
{
  condition_on_tags.push_back(tag);
  condition_on_names.push_back(name);
  return *this;
}
predictor& predictor::set_condition(ptag tag, char name)
{
  condition_on_tags.clear();
  condition_on_names.clear();
  return add_condition(tag, name);
}

predictor& predictor::add_condition_range(ptag hi, ptag count, char name0)
{
  if (count == 0) return *this;
  for (ptag i = 0; i < count; i++)
  {
    if (i > hi) break;
    char name = name0 + static_cast<char>(i);
    condition_on_tags.push_back(hi - i);
    condition_on_names.push_back(name);
  }
  return *this;
}
predictor& predictor::set_condition_range(ptag hi, ptag count, char name0)
{
  condition_on_tags.clear();
  condition_on_names.clear();
  return add_condition_range(hi, count, name0);
}

predictor& predictor::set_learner_id(size_t id)
{
  learner_id = id;
  return *this;
}

predictor& predictor::set_tag(ptag tag)
{
  my_tag = tag;
  return *this;
}

action predictor::predict()
{
  const action* orA = oracle_actions.size() == 0 ? nullptr : oracle_actions.begin();
  const ptag* cOn = condition_on_names.size() == 0 ? nullptr : condition_on_tags.begin();
  const char* cNa = nullptr;
  if (condition_on_names.size() > 0)
  {
    condition_on_names.push_back(static_cast<char>(0));  // null terminate
    cNa = condition_on_names.begin();
  }
  const action* alA = (allowed_actions.size() == 0) ? nullptr : allowed_actions.begin();
  const float* alAcosts = (allowed_actions_cost.size() == 0) ? nullptr : allowed_actions_cost.begin();
  size_t numAlA = std::max(allowed_actions.size(), allowed_actions_cost.size());
  action p = is_ldf
      ? sch.predictLDF(ec, ec_cnt, my_tag, orA, oracle_actions.size(), cOn, cNa, learner_id, weight)
      : sch.predict(*ec, my_tag, orA, oracle_actions.size(), cOn, cNa, alA, numAlA, alAcosts, learner_id, weight);

  if (condition_on_names.size() > 0) condition_on_names.pop_back();  // un-null-terminate
  return p;
}
}  // namespace Search

// TODO: valgrind --leak-check=full ./vw --search 2 -k -c --passes 1 --search_task sequence -d test_beam --holdout_off
// --search_rollin policy --search_metatask selective_branching 2>&1 | less
