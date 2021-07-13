// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <float.h>
#include <sstream>
#include <memory>

#include "reductions.h"
#include "rand48.h"

using namespace VW::LEARNER;
using namespace VW::config;

namespace recall_tree_ns
{
struct node_pred
{
  uint32_t label;
  double label_count;

  node_pred() = default;
  node_pred(uint32_t a) : label(a), label_count(0) {}
};

static_assert(std::is_trivial<node_pred>::value, "To be used in v_array node_pred must be trivial");

struct node
{
  uint32_t parent;
  float recall_lbest;

  bool internal;
  uint32_t depth;

  uint32_t base_router;
  uint32_t left;
  uint32_t right;
  double n;
  double entropy;
  double passes;

  v_array<node_pred> preds;

  node()
      : parent(0)
      , recall_lbest(0)
      , internal(false)
      , depth(0)
      , base_router(0)
      , left(0)
      , right(0)
      , n(0)
      , entropy(0)
      , passes(1)
  {
  }
};

struct recall_tree
{
  vw* all;
  std::shared_ptr<rand_state> _random_state;
  uint32_t k;
  bool node_only;

  std::vector<node> nodes;

  size_t max_candidates;
  size_t max_routers;
  size_t max_depth;
  float bern_hyper;

  bool randomized_routing;
};

float to_prob(float x)
{
  static const float alpha = 2.0f;
  // http://stackoverflow.com/questions/2789481/problem-calling-stdmax
  return std::max(0.f, std::min(1.f, 0.5f * (1.0f + alpha * x)));
}

void init_tree(recall_tree& b, uint32_t root, uint32_t depth, uint32_t& routers_used)
{
  if (depth <= b.max_depth)
  {
    uint32_t left_child;
    uint32_t right_child;
    left_child = static_cast<uint32_t>(b.nodes.size());
    b.nodes.push_back(node());
    right_child = static_cast<uint32_t>(b.nodes.size());
    b.nodes.push_back(node());
    b.nodes[root].base_router = routers_used++;

    b.nodes[root].internal = true;
    b.nodes[root].left = left_child;
    b.nodes[left_child].parent = root;
    b.nodes[left_child].depth = depth;
    b.nodes[root].right = right_child;
    b.nodes[right_child].parent = root;
    b.nodes[right_child].depth = depth;

    init_tree(b, left_child, depth + 1, routers_used);
    init_tree(b, right_child, depth + 1, routers_used);
  }
}

void init_tree(recall_tree& b)
{
  uint32_t routers_used = 0;

  b.nodes.push_back(node());
  init_tree(b, 0, 1, routers_used);
  b.max_routers = routers_used;
}

node_pred* find(recall_tree& b, uint32_t cn, example& ec)
{
  node_pred* ls;

  for (ls = b.nodes[cn].preds.begin(); ls != b.nodes[cn].preds.end() && ls->label != ec.l.multi.label; ++ls)
    ;

  return ls;
}

node_pred* find_or_create(recall_tree& b, uint32_t cn, example& ec)
{
  node_pred* ls = find(b, cn, ec);

  if (ls == b.nodes[cn].preds.end())
  {
    node_pred newls(ec.l.multi.label);
    b.nodes[cn].preds.push_back(newls);
    ls = b.nodes[cn].preds.end() - 1;
  }

  return ls;
}

void compute_recall_lbest(recall_tree& b, node* n)
{
  if (n->n <= 0) return;

  double mass_at_k = 0;

  for (node_pred* ls = n->preds.begin(); ls != n->preds.end() && ls < n->preds.begin() + b.max_candidates; ++ls)
  { mass_at_k += ls->label_count; }

  float f = static_cast<float>(mass_at_k) / static_cast<float>(n->n);
  float stdf = std::sqrt(f * (1.f - f) / static_cast<float>(n->n));
  float diamf = 15.f / (std::sqrt(18.f) * static_cast<float>(n->n));

  n->recall_lbest = std::max(0.f, f - std::sqrt(b.bern_hyper) * stdf - b.bern_hyper * diamf);
}

double plogp(double c, double n) { return (c == 0) ? 0 : (c / n) * log(c / n); }

double updated_entropy(recall_tree& b, uint32_t cn, example& ec)
{
  node_pred* ls = find(b, cn, ec);

  // entropy = -\sum_k (c_k/n) Log[c_k/n]
  // c_0 <- c_0 + 1, n <- n + 1
  // entropy <- + (c_0/n) Log[c_0/n]
  //            - n/(n+1) \sum_{k>0} (c_k/n) Log[c_k/n]
  //            - Log[n/(n+1)] \sum_{k>0} (c_k/(n+1))
  //            - ((c_0+1)/(n+1)) Log[(c_0+1)/(n+1)]

  double c0 = (ls == b.nodes[cn].preds.end()) ? 0 : ls->label_count;
  double deltac0 = ec.weight;
  double n = b.nodes[cn].n;

  double novernp1 = n / (deltac0 + n);
  double lognovernp1 = (novernp1 == 0) ? 0 : log(novernp1);
  double nminusc0overnp1 = (n - c0) / (n + deltac0);

  double newentropy = b.nodes[cn].entropy;

  newentropy += plogp(c0, n);
  newentropy *= novernp1;
  newentropy -= lognovernp1 * nminusc0overnp1;
  newentropy -= plogp(c0 + deltac0, n + deltac0);

  return newentropy;
}

void insert_example_at_node(recall_tree& b, uint32_t cn, example& ec)
{
  node_pred* ls = find_or_create(b, cn, ec);

  b.nodes[cn].entropy = updated_entropy(b, cn, ec);

  ls->label_count += ec.weight;

  while (ls != b.nodes[cn].preds.begin() && ls[-1].label_count < ls[0].label_count)
  {
    std::swap(ls[-1], ls[0]);
    --ls;
  }

  b.nodes[cn].n += ec.weight;

  compute_recall_lbest(b, &b.nodes[cn]);
}

// TODO: handle if features already in this namespace

void add_node_id_feature(recall_tree& b, uint32_t cn, example& ec)
{
  vw* all = b.all;
  uint64_t mask = all->weights.mask();
  size_t ss = all->weights.stride_shift();

  ec.indices.push_back(node_id_namespace);
  features& fs = ec.feature_space[node_id_namespace];

  if (b.node_only) { fs.push_back(1., ((static_cast<uint64_t>(868771) * cn) << ss) & mask); }
  else
  {
    while (cn > 0)
    {
      fs.push_back(1., ((static_cast<uint64_t>(868771) * cn) << ss) & mask);
      cn = b.nodes[cn].parent;
    }
  }

  // TODO: audit ?
  // TODO: if namespace already exists ?
}

void remove_node_id_feature(recall_tree& /* b */, uint32_t /* cn */, example& ec)
{
  features& fs = ec.feature_space[node_id_namespace];
  fs.clear();
  ec.indices.pop_back();
}

uint32_t oas_predict(recall_tree& b, single_learner& base, uint32_t cn, example& ec)
{
  MULTICLASS::label_t mc = ec.l.multi;
  uint32_t save_pred = ec.pred.multiclass;

  uint32_t amaxscore = 0;

  add_node_id_feature(b, cn, ec);
  ec.l.simple = {FLT_MAX};
  ec._reduction_features.template get<simple_label_reduction_features>().reset_to_default();
  float maxscore = std::numeric_limits<float>::lowest();
  for (node_pred* ls = b.nodes[cn].preds.begin();
       ls != b.nodes[cn].preds.end() && ls < b.nodes[cn].preds.begin() + b.max_candidates; ++ls)
  {
    base.predict(ec, b.max_routers + ls->label - 1);
    if (amaxscore == 0 || ec.partial_prediction > maxscore)
    {
      maxscore = ec.partial_prediction;
      amaxscore = ls->label;
    }
  }

  remove_node_id_feature(b, cn, ec);

  ec.l.multi = mc;
  ec.pred.multiclass = save_pred;

  return amaxscore;
}

bool is_candidate(recall_tree& b, uint32_t cn, example& ec)
{
  for (node_pred* ls = b.nodes[cn].preds.begin();
       ls != b.nodes[cn].preds.end() && ls < b.nodes[cn].preds.begin() + b.max_candidates; ++ls)
  {
    if (ls->label == ec.l.multi.label) return true;
  }

  return false;
}

inline uint32_t descend(node& n, float prediction) { return prediction < 0 ? n.left : n.right; }

struct predict_type
{
  uint32_t node_id;
  uint32_t class_prediction;

  predict_type(uint32_t a, uint32_t b) : node_id(a), class_prediction(b) {}
};

bool stop_recurse_check(recall_tree& b, uint32_t parent, uint32_t child)
{
  return b.bern_hyper > 0 && b.nodes[parent].recall_lbest >= b.nodes[child].recall_lbest;
}

predict_type predict_from(recall_tree& b, single_learner& base, example& ec, uint32_t cn)
{
  MULTICLASS::label_t mc = ec.l.multi;
  uint32_t save_pred = ec.pred.multiclass;

  ec.l.simple = {FLT_MAX};
  ec._reduction_features.template get<simple_label_reduction_features>().reset_to_default();
  while (b.nodes[cn].internal)
  {
    base.predict(ec, b.nodes[cn].base_router);
    uint32_t newcn = descend(b.nodes[cn], ec.partial_prediction);
    bool cond = stop_recurse_check(b, cn, newcn);

    if (cond) break;

    cn = newcn;
  }

  ec.l.multi = mc;
  ec.pred.multiclass = save_pred;

  return predict_type(cn, oas_predict(b, base, cn, ec));
}

void predict(recall_tree& b, single_learner& base, example& ec)
{
  predict_type pred = predict_from(b, base, ec, 0);

  ec.pred.multiclass = pred.class_prediction;
}

float train_node(recall_tree& b, single_learner& base, example& ec, uint32_t cn)
{
  MULTICLASS::label_t mc = ec.l.multi;
  uint32_t save_pred = ec.pred.multiclass;

  // minimize entropy
  // better than maximize expected likelihood, and the proofs go through :)
  double new_left = updated_entropy(b, b.nodes[cn].left, ec);
  double new_right = updated_entropy(b, b.nodes[cn].right, ec);
  double old_left = b.nodes[b.nodes[cn].left].entropy;
  double old_right = b.nodes[b.nodes[cn].right].entropy;
  double nl = b.nodes[b.nodes[cn].left].n;
  double nr = b.nodes[b.nodes[cn].right].n;
  double delta_left = nl * (new_left - old_left) + mc.weight * new_left;
  double delta_right = nr * (new_right - old_right) + mc.weight * new_right;
  float route_label = delta_left < delta_right ? -1.f : 1.f;

  // Bug?
  // Notes: imp_weight does not seem to be used since it was not set to
  // ec.weight before.  ec.l.simple.weight was not used in gd.
  // float imp_weight = fabs((float)(delta_left - delta_right));

  ec.l.simple = {route_label};
  ec._reduction_features.template get<simple_label_reduction_features>().reset_to_default();
  // Bug?
  // Notes: looks like imp_weight was not used since ec.l.simple.weight is not
  // used in gd.
  // Only ec.weight is used in gd.  ec.imp_weight is now set to 0 instead of
  // ec.l.simple.weight.
  // This causes different results during RunTests
  // ec.weight = imp_weight;

  base.learn(ec, b.nodes[cn].base_router);

  // TODO: using the updated routing seems to help
  // TODO: consider faster version using updated_prediction
  // TODO: (doesn't play well with link function)
  base.predict(ec, b.nodes[cn].base_router);

  float save_scalar = ec.pred.scalar;

  ec.l.multi = mc;
  ec.pred.multiclass = save_pred;

  return save_scalar;
}

void learn(recall_tree& b, single_learner& base, example& ec)
{
  if (b.all->training && ec.l.multi.label != static_cast<uint32_t>(-1))  // if training the tree
  {
    uint32_t cn = 0;

    while (b.nodes[cn].internal)
    {
      float which = train_node(b, base, ec, cn);

      if (b.randomized_routing) which = (b._random_state->get_and_update_random() > to_prob(which) ? -1.f : 1.f);

      uint32_t newcn = descend(b.nodes[cn], which);
      bool cond = stop_recurse_check(b, cn, newcn);
      insert_example_at_node(b, cn, ec);

      if (cond)
      {
        insert_example_at_node(b, newcn, ec);
        break;
      }

      cn = newcn;
    }

    if (!b.nodes[cn].internal) insert_example_at_node(b, cn, ec);

    if (is_candidate(b, cn, ec))
    {
      MULTICLASS::label_t mc = ec.l.multi;
      uint32_t save_pred = ec.pred.multiclass;

      add_node_id_feature(b, cn, ec);

      ec.l.simple = {1.f};
      ec._reduction_features.template get<simple_label_reduction_features>().reset_to_default();
      base.learn(ec, b.max_routers + mc.label - 1);
      ec.l.simple = {-1.f};

      for (node_pred* ls = b.nodes[cn].preds.begin();
           ls != b.nodes[cn].preds.end() && ls < b.nodes[cn].preds.begin() + b.max_candidates; ++ls)
      {
        if (ls->label != mc.label) base.learn(ec, b.max_routers + ls->label - 1);
      }

      remove_node_id_feature(b, cn, ec);

      ec.l.multi = mc;
      ec.pred.multiclass = save_pred;
    }
  }
}

void save_load_tree(recall_tree& b, io_buf& model_file, bool read, bool text)
{
  if (model_file.num_files() > 0)
  {
    std::stringstream msg;

    writeit(b.k, "k");
    writeit(b.node_only, "node_only");
    writeitvar(b.nodes.size(), "nodes", n_nodes);

    if (read)
    {
      b.nodes.clear();
      for (uint32_t j = 0; j < n_nodes; ++j) { b.nodes.push_back(node()); }
    }

    writeit(b.max_candidates, "max_candidates");
    writeit(b.max_depth, "max_depth");

    for (uint32_t j = 0; j < n_nodes; ++j)
    {
      node* cn = &b.nodes[j];

      writeit(cn->parent, "parent");
      writeit(cn->recall_lbest, "recall_lbest");
      writeit(cn->internal, "internal");
      writeit(cn->depth, "depth");
      writeit(cn->base_router, "base_router");
      writeit(cn->left, "left");
      writeit(cn->right, "right");
      writeit(cn->n, "n");
      writeit(cn->entropy, "entropy");
      writeit(cn->passes, "passes");

      writeitvar(cn->preds.size(), "n_preds", n_preds);

      if (read)
      {
        cn->preds.clear();

        for (uint32_t k = 0; k < n_preds; ++k) { cn->preds.push_back(node_pred(0)); }
      }

      for (uint32_t k = 0; k < n_preds; ++k)
      {
        node_pred* pred = &cn->preds[k];

        writeit(pred->label, "label");
        writeit(pred->label_count, "label_count");
      }

      if (read) { compute_recall_lbest(b, cn); }
    }
  }
}

}  // namespace recall_tree_ns

using namespace recall_tree_ns;

base_learner* recall_tree_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto tree = scoped_calloc_or_throw<recall_tree>();
  option_group_definition new_options("Recall Tree");
  new_options.add(make_option("recall_tree", tree->k).keep().necessary().help("Use online tree for multiclass"))
      .add(make_option("max_candidates", tree->max_candidates)
               .keep()
               .help("maximum number of labels per leaf in the tree"))
      .add(make_option("bern_hyper", tree->bern_hyper).default_value(1.f).help("recall tree depth penalty"))
      .add(make_option("max_depth", tree->max_depth).keep().help("maximum depth of the tree, default log_2 (#classes)"))
      .add(make_option("node_only", tree->node_only).keep().help("only use node features, not full path features"))
      .add(make_option("randomized_routing", tree->randomized_routing).keep().help("randomized routing"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  tree->all = &all;
  tree->_random_state = all.get_random_state();
  tree->max_candidates = options.was_supplied("max_candidates")
      ? tree->max_candidates
      : std::min(tree->k, 4 * static_cast<uint32_t>(ceil(log(tree->k) / log(2.0))));
  tree->max_depth = options.was_supplied("max_depth")
      ? tree->max_depth
      : static_cast<uint32_t>(std::ceil(std::log(tree->k) / std::log(2.0)));

  init_tree(*tree.get());

  if (!all.logger.quiet)
    *(all.trace_message) << "recall_tree:"
                         << " node_only = " << tree->node_only << " bern_hyper = " << tree->bern_hyper
                         << " max_depth = " << tree->max_depth << " routing = "
                         << (all.training ? (tree->randomized_routing ? "randomized" : "deterministic")
                                          : "n/a testonly")
                         << std::endl;

  learner<recall_tree, example>& l = init_multiclass_learner(tree, as_singleline(setup_base(options, all)), learn,
      predict, all.example_parser, tree->max_routers + tree->k, all.get_setupfn_name(recall_tree_setup));
  all.example_parser->lbl_parser.label_type = label_type_t::multiclass;
  l.set_save_load(save_load_tree);

  return make_base(l);
}
