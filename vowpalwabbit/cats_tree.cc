// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <algorithm>
#include <cassert>
#include <limits>

#include "cats_tree.h"
#include "parse_args.h"  // setup_base()
#include "learner.h"     // init_learner()
#include "reductions.h"
#include "debug_log.h"
#include "explore_internal.h"
#include "hash.h"
#include "guard.h"
#include "label_parser.h"

using namespace VW::config;
using namespace VW::LEARNER;

using CB::cb_class;
using std::vector;

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::cats_tree

namespace VW
{
namespace cats_tree
{

bool tree_node::operator==(const tree_node& rhs) const
{
  if (this == &rhs) return true;
  return (id == rhs.id && left_id == rhs.left_id && right_id == rhs.right_id && parent_id == rhs.parent_id &&
      depth == rhs.depth && left_only == rhs.left_only && right_only == rhs.right_only && is_leaf == rhs.is_leaf);
}

bool tree_node::operator!=(const tree_node& rhs) const { return !(*this == rhs); }

void cats_tree::init_node_costs(v_array<cb_class>& ac)
{
  assert(ac.size() > 0);
  assert(ac[0].action > 0);

  _cost_star = ac[0].cost / ac[0].probability;

  uint32_t node_id = ac[0].action + _binary_tree.internal_node_count() - 1;
  // stay inside the node boundaries
  if (node_id >= _binary_tree.nodes.size()) { node_id = static_cast<uint32_t>(_binary_tree.nodes.size()) - 1; }
  _a = {node_id, _cost_star};

  node_id = ac[ac.size() - 1].action + _binary_tree.internal_node_count() - 1;
  // stay inside the node boundaries
  if (node_id >= _binary_tree.nodes.size()) { node_id = static_cast<uint32_t>(_binary_tree.nodes.size()) - 1; }
  _b = {node_id, _cost_star};
}

constexpr float RIGHT = 1.0f;
constexpr float LEFT = -1.0f;

float cats_tree::return_cost(const tree_node& w)
{
  if (w.id < _a.node_id)
    return 0;
  else if (w.id == _a.node_id)
    return _a.cost;
  else if (w.id < _b.node_id)
    return _cost_star;
  else if (w.id == _b.node_id)
    return _b.cost;
  else
    return 0;
}

void cats_tree::learn(LEARNER::single_learner& base, example& ec)
{
  const float saved_weight = ec.weight;
  auto saved_pred = stash_guard(ec.pred);

  const vector<tree_node>& nodes = _binary_tree.nodes;
  v_array<cb_class>& ac = ec.l.cb.costs;

  VW_DBG(ec) << "tree_c: learn() -- tree_traversal -- " << std::endl;

  init_node_costs(ac);

  for (uint32_t d = _binary_tree.depth(); d > 0; d--)
  {
    std::vector<node_cost> set_d = {_a};
    if (nodes[_a.node_id].parent_id != nodes[_b.node_id].parent_id) { set_d.push_back(_b); }
    float a_parent_cost = _a.cost;
    float b_parent_cost = _b.cost;
    for (uint32_t i = 0; i < set_d.size(); i++)
    {
      const node_cost& n_c = set_d[i];
      const tree_node& v = nodes[n_c.node_id];
      const float cost_v = n_c.cost;
      const tree_node& v_parent = nodes[v.parent_id];
      float cost_parent = cost_v;
      if (v_parent.right_only || v_parent.left_only) continue;
      const tree_node& w = _binary_tree.get_sibling(v);  // w is sibling of v
      float cost_w = return_cost(w);
      if (cost_v != cost_w)
      {
        VW_DBG(ec) << "tree_c: learn() cost_w = " << cost_w << ", cost_v != cost_w" << std::endl;
        float local_action = RIGHT;
        if (((cost_v < cost_w) ? v : w).id == v_parent.left_id) { local_action = LEFT; }

        ec.l.simple.label = local_action;
        ec.weight = std::abs(cost_v - cost_w);

        bool filter = false;
        const float weight_th = 0.00001f;
        if (ec.weight < weight_th)
        {
          // generate a new seed
          uint64_t new_random_seed = uniform_hash(&app_seed, sizeof(app_seed), app_seed);
          // pick a uniform random number between 0.0 - .001f
          float random_draw = exploration::uniform_random_merand48(new_random_seed) * weight_th;
          if (random_draw < ec.weight) { ec.weight = weight_th; }
          else
          {
            filter = true;
          }
        }
        if (!filter)
        {
          VW_DBG(ec) << "tree_c: learn() #### binary learning the node " << v.parent_id << std::endl;
          base.learn(ec, v.parent_id);
          _binary_tree.nodes[v.parent_id].learn_count++;
          base.predict(ec, v.parent_id);
          VW_DBG(ec) << "tree_c: learn() after binary predict:" << scalar_pred_to_string(ec)
                     << ", local_action = " << (local_action) << std::endl;
          float trained_action = (ec.pred.scalar < 0) ? LEFT : RIGHT;

          if (trained_action == local_action)
          {
            cost_parent = std::min(cost_v, cost_w) * (1 + std::abs(ec.pred.scalar)) / 2.f +
                std::max(cost_v, cost_w) * (1 - std::abs(ec.pred.scalar)) / 2.f;
            VW_DBG(ec) << "tree_c: learn() ec.pred.scalar == local_action" << std::endl;
          }
          else
          {
            cost_parent = std::max(cost_v, cost_w) * (1 + std::abs(ec.pred.scalar)) / 2.f +
                std::min(cost_v, cost_w) * (1 - std::abs(ec.pred.scalar)) / 2.f;
            VW_DBG(ec) << "tree_c: learn() ec.pred.scalar != local_action" << std::endl;
          }
        }
      }
      if (i == 0)
        a_parent_cost = cost_parent;
      else
        b_parent_cost = cost_parent;
    }
    _a = {nodes[_a.node_id].parent_id, a_parent_cost};
    _b = {nodes[_b.node_id].parent_id, b_parent_cost};
  }

  ec.weight = saved_weight;
}

void cats_tree::set_trace_message(std::ostream* vw_ostream, bool quiet)
{
  _trace_stream = vw_ostream;
  _quiet = quiet;
}

void learn(cats_tree& tree, single_learner& base, example& ec)
{
  VW_DBG(ec) << "tree_c: before tree.learn() " << cb_label_to_string(ec) << features_to_string(ec) << std::endl;
  tree.learn(base, ec);
  VW_DBG(ec) << "tree_c: after tree.learn() " << cb_label_to_string(ec) << features_to_string(ec) << std::endl;
}

base_learner* setup(options_i& options, vw& all)
{
  option_group_definition new_options("CATS Tree Options");
  uint32_t num_actions;  // = K = 2^D
  uint32_t bandwidth;    // = 2^h#
  std::string link;
  new_options.add(make_option("cats_tree", num_actions).keep().necessary().help("CATS Tree with <k> labels"))
      .add(make_option("tree_bandwidth", bandwidth)
               .default_value(0)
               .keep()
               .help("tree bandwidth for continuous actions in terms of #actions"))
      .add(make_option("link", link).keep().help("Specify the link function: identity, logistic, glf1 or poisson"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // default behaviour uses binary
  if (!options.was_supplied("link")) { options.insert("binary", ""); }
  else
  {
    // if link was supplied then force glf1
    if (link != "glf1")
    { *(all.trace_message) << "warning: cats_tree only supports glf1; resetting to glf1." << std::endl; }
    options.replace("link", "glf1");
  }

  auto tree = VW::make_unique<cats_tree>();
  tree->init(num_actions, bandwidth);
  tree->set_trace_message(all.trace_message.get(), all.logger.quiet);

  base_learner* base = setup_base(options, all);
  int32_t params_per_weight = tree->learner_count();
  auto* l = make_reduction_learner(std::move(tree), as_singleline(base), learn, predict, all.get_setupfn_name(setup))
                .set_params_per_weight(params_per_weight)
                .set_prediction_type(prediction_type_t::multiclass)
                .set_label_type(label_type_t::cb)
                .build();
  return make_base(*l);
}

}  // namespace cats_tree
}  // namespace VW
