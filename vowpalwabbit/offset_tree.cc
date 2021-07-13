// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "offset_tree.h"
#include "parse_args.h"  // setup_base()
#include "learner.h"     // init_learner()
#include "action_score.h"

using namespace VW::config;
using namespace VW::LEARNER;

namespace VW
{
namespace offset_tree
{
tree_node::tree_node(uint32_t node_id, uint32_t left_node_id, uint32_t right_node_id, uint32_t p_id, bool is_leaf)
    : id(node_id), left_id(left_node_id), right_id(right_node_id), parent_id(p_id), is_leaf(is_leaf)
{
}

bool tree_node::operator==(const tree_node& rhs) const
{
  if (this == &rhs) return true;
  return (id == rhs.id && left_id == rhs.left_id && right_id == rhs.right_id && is_leaf == rhs.is_leaf &&
      parent_id == rhs.parent_id);
}

bool tree_node::operator!=(const tree_node& rhs) const { return !(*this == rhs); }

void min_depth_binary_tree::build_tree(uint32_t num_nodes)
{
  // Sanity checks
  if (_initialized)
  {
    if (num_nodes != _num_leaf_nodes)
    {
      THROW("Tree already initialized.  New leaf node count (" << num_nodes << ") does not equal current value. ("
                                                               << _num_leaf_nodes << ")");
    }
    return;
  }

  _num_leaf_nodes = num_nodes;
  // deal with degenerate cases of 0 and 1 actions
  if (_num_leaf_nodes == 0)
  {
    _initialized = true;
    return;
  }

  try
  {
    nodes.reserve(2 * _num_leaf_nodes - 1);
    std::vector<uint32_t> tournaments;
    // Add all leaf nodes to node collection
    tournaments.reserve(_num_leaf_nodes);
    for (uint32_t i = 0; i < _num_leaf_nodes; i++)
    {
      nodes.emplace_back(i, 0, 0, 0, true);
      tournaments.emplace_back(i);
    }
    while (tournaments.size() > 1)
    {
      // ids of next set of nodes start with the current
      const auto num_tournaments = tournaments.size();
      // node count (since ids are zero based)
      auto id = static_cast<uint32_t>(nodes.size());
      std::vector<uint32_t> new_tournaments;
      new_tournaments.reserve(num_tournaments / 2 + 1);
      for (size_t j = 0; j < num_tournaments / 2; ++j)
      {
        auto left = tournaments[2 * j];
        auto right = tournaments[2 * j + 1];
        new_tournaments.emplace_back(id);
        nodes[left].parent_id = id;
        nodes[right].parent_id = id;
        nodes.emplace_back(id++, left, right, 0, false);
      }
      if (num_tournaments % 2 == 1) new_tournaments.emplace_back(tournaments.back());
      tournaments = std::move(new_tournaments);
    }
    root_idx = tournaments[0];
    nodes[nodes.size() - 1].parent_id = root_idx;
  }
  catch (std::bad_alloc& e)
  {
    THROW("Unable to allocate memory for offset tree.  Label count:" << _num_leaf_nodes << " bad_alloc:" << e.what());
  }
  _initialized = true;
}

uint32_t min_depth_binary_tree::internal_node_count() const
{
  return static_cast<uint32_t>(nodes.size()) - _num_leaf_nodes;
}

uint32_t min_depth_binary_tree::leaf_node_count() const { return _num_leaf_nodes; }

void offset_tree::init() { binary_tree.build_tree(_num_actions); }

offset_tree::offset_tree(uint32_t num_actions)
    : _num_actions{num_actions}, _prediction_buffer(num_actions), _scores(num_actions)
{
}

int32_t offset_tree::learner_count() const { return binary_tree.internal_node_count(); }

// Helper to deal with collections that don't start with an index of 0
template <typename T>
struct offset_helper
{
  // typedef verbose prediction buffer type
  offset_helper(T& b, uint32_t index_offset) : start_index_offset{index_offset}, collection(b) {}

  // intercept index operator to adjust the offset before
  // passing to underlying collection
  typename T::const_reference operator[](size_t idx) const { return collection[idx - start_index_offset]; }

  typename T::reference operator[](size_t idx) { return collection[idx - start_index_offset]; }

private:
  uint32_t start_index_offset = 0;
  T& collection;
};

const offset_tree::scores_t& offset_tree::predict(LEARNER::single_learner& base, example& ec)
{
  // - pair<float,float> stores the scores for left and right nodes
  // - prediction_buffer stores predictions for all the nodes in the tree for the duration
  //   of the predict() call
  // - This is to reduce memory allocations in steady state.

  auto& t = binary_tree;

  _prediction_buffer.clear();
  _scores.resize(t.leaf_node_count());

  // Handle degenerate cases of zero and one node trees
  if (t.leaf_node_count() == 0) return _scores;
  if (t.leaf_node_count() == 1)
  {
    _scores[0] = 1.0f;
    return _scores;
  }

  const CB::label saved_label = ec.l.cb;
  ec.l.cb.costs.clear();

  // Get predictions for all internal nodes
  for (uint32_t idx = 0; idx < t.internal_node_count(); ++idx)
  {
    base.predict(ec, idx);
    _prediction_buffer.emplace_back(ec.pred.a_s[0].score, ec.pred.a_s[1].score);
  }

  // Restore example label.
  ec.l.cb = saved_label;

  // use a offset helper to deal with start index offset
  offset_helper<predict_buffer_t> buffer_helper(_prediction_buffer, t.leaf_node_count());

  // Compute action scores
  for (auto rit = t.nodes.rbegin(); rit != t.nodes.rend(); ++rit)
  {
    // done processing all internal nodes
    if (rit->is_leaf) break;

    // update probabilities for left node
    const float left_p = buffer_helper[rit->id].first;
    if (t.nodes[rit->left_id].is_leaf) { _scores[rit->left_id] = left_p; }
    else
    {
      const auto left_left_p = buffer_helper[rit->left_id].first;
      buffer_helper[rit->left_id].first = left_left_p * left_p;
      const auto left_right_p = buffer_helper[rit->left_id].second;
      buffer_helper[rit->left_id].second = left_right_p * left_p;
    }

    // update probabilities for right node
    const float right_p = buffer_helper[rit->id].second;
    if (t.nodes[rit->right_id].is_leaf) { _scores[rit->right_id] = right_p; }
    else
    {
      const auto right_left_p = buffer_helper[rit->right_id].first;
      buffer_helper[rit->right_id].first = right_left_p * right_p;
      const auto right_right_p = buffer_helper[rit->right_id].second;
      buffer_helper[rit->right_id].second = right_right_p * right_p;
    }
  }

  return _scores;
}

void offset_tree::learn(LEARNER::single_learner& base, example& ec)
{
  const auto global_action = ec.l.cb.costs[0].action;
  const auto global_weight = ec.weight;

  // for brevity
  auto& nodes = binary_tree.nodes;

  tree_node node = nodes[global_action - 1];
  do
  {  // ascend
    const auto previous_id = node.id;
    node = nodes[node.parent_id];

    // learn
    uint32_t local_action = 2;
    if (node.left_id == previous_id) local_action = 1;
    ec.l.cb.costs[0].action = local_action;
    base.learn(ec, node.id - binary_tree.leaf_node_count());

    // re-weight
    base.predict(ec, node.id - binary_tree.leaf_node_count());
    ec.weight *= ec.pred.a_s[local_action - 1].score;
  } while (node.parent_id != node.id);

  ec.l.cb.costs[0].action = global_action;
  ec.weight = global_weight;
}

inline void copy_to_action_scores(const offset_tree::scores_t& scores, ACTION_SCORE::action_scores& a_s)
{
  a_s.clear();
  for (uint32_t idx = 0; idx < scores.size(); ++idx) { a_s.push_back({idx, scores[idx]}); }
}

void predict(offset_tree& tree, single_learner& base, example& ec)
{
  // get predictions for all internal nodes in binary tree.
  ec.pred.a_s.clear();
  const auto& scores = tree.predict(base, ec);
  copy_to_action_scores(scores, ec.pred.a_s);
}

void learn(offset_tree& tree, single_learner& base, example& ec)
{
  ec.pred.a_s.clear();

  // store predictions before learning
  const offset_tree::scores_t& saved_scores = tree.predict(base, ec);

  // learn
  tree.learn(base, ec);

  // restore predictions
  copy_to_action_scores(saved_scores, ec.pred.a_s);
}

base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all)
{
  option_group_definition new_options("Offset tree Options");
  uint32_t num_actions;
  new_options.add(make_option("ot", num_actions).keep().necessary().help("Offset tree with <k> labels"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // Ensure that cb_explore will be the base reduction
  if (!options.was_supplied("cb_explore")) { options.insert("cb_explore", "2"); }

  // default to legacy cb implementation
  options.insert("cb_force_legacy", "");

  auto otree = scoped_calloc_or_throw<offset_tree>(num_actions);
  otree->init();

  base_learner* base = setup_base(options, all);

  learner<offset_tree, example>& l = init_learner(otree, as_singleline(base), learn, predict, otree->learner_count(),
      prediction_type_t::action_probs, all.get_setupfn_name(setup));

  return make_base(l);
}
}  // namespace offset_tree
}  // namespace VW
