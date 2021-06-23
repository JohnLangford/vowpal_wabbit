// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "reductions_fwd.h"
#include "error_reporting.h"
#include "cb.h"
#include "example.h"
#include "learner_no_throw.h"

#include <vector>
#include <sstream>

namespace VW
{
namespace cats_tree
{
LEARNER::base_learner* setup(config::options_i& options, vw& all);

struct tree_node
{
  tree_node(uint32_t node_id, uint32_t left_node_id, uint32_t right_node_id, uint32_t p_id, uint32_t depth,
      bool left_only, bool right_only, bool is_leaf)
      : id(node_id)
      , left_id(left_node_id)
      , right_id(right_node_id)
      , parent_id(p_id)
      , depth(depth)
      , left_only(left_only)
      , right_only(right_only)
      , is_leaf(is_leaf)
      , learn_count(0)
  {
  }

  inline bool operator==(const tree_node& rhs) const;
  bool operator!=(const tree_node& rhs) const;

  inline bool is_root() const { return id == parent_id; }

  uint32_t id;
  uint32_t left_id;
  uint32_t right_id;
  uint32_t parent_id;
  uint32_t depth;
  bool left_only;
  bool right_only;
  bool is_leaf;
  uint32_t learn_count;
};

struct min_depth_binary_tree
{
  bool build_tree(uint32_t num_nodes, uint32_t bandwidth)
  {
    // Sanity checks
    if (_initialized)
    {
      if (num_nodes != _num_leaf_nodes) { return false; }
      return true;
    }

    _num_leaf_nodes = num_nodes;
    // deal with degenerate cases of 0 and 1 actions
    if (_num_leaf_nodes == 0)
    {
      _initialized = true;
      return true;
    }

    try
    {
      // Number of nodes in a minimal binary tree := (2 * LeafCount) - 1
      nodes.reserve(2 * _num_leaf_nodes - 1);

      //  Insert Root Node: First node in the collection, Parent is itself
      //  {node_id, left_id, right_id, parent_id, depth, right_only, left_only, is_leaf}
      nodes.emplace_back(0, 0, 0, 0, 0, false, false, true);

      uint32_t depth = 0, depth_const = 1;
      for (uint32_t i = 0; i < _num_leaf_nodes - 1; ++i)
      {
        nodes[i].left_id = 2 * i + 1;
        nodes[i].right_id = 2 * i + 2;
        nodes[i].is_leaf = false;
        if (2 * i + 1 >= depth_const) { depth_const = (1 << (++depth + 1)) - 1; }

        uint32_t id = 2 * i + 1;
        bool right_only = false;
        bool left_only = false;
        if (bandwidth)
        {
          right_only = (id == (_num_leaf_nodes / (2 * bandwidth) - 1));
          left_only = (id == (_num_leaf_nodes / (bandwidth)-2));
        }
        nodes.emplace_back(id, 0, 0, i, depth, left_only, right_only, true);

        id = 2 * i + 2;
        if (bandwidth)
        {
          right_only = (id == (_num_leaf_nodes / (2 * bandwidth) - 1));
          left_only = (id == (_num_leaf_nodes / (bandwidth)-2));
        }
        nodes.emplace_back(id, 0, 0, i, depth, left_only, right_only, true);
      }

      _initialized = true;
      _depth = depth;
    }
    catch (std::bad_alloc&)
    {
      return false;
    }
    return true;
  }
  uint32_t internal_node_count() const { return static_cast<uint32_t>(nodes.size()) - _num_leaf_nodes; }

  uint32_t leaf_node_count() const { return _num_leaf_nodes; }

  uint32_t depth() const { return _depth; }

  const tree_node& get_sibling(const tree_node& v)
  {
    // We expect not to get called on root
    const tree_node& v_parent = nodes[v.parent_id];
    return nodes[(v.id == v_parent.left_id) ? v_parent.right_id : v_parent.left_id];
  }

  std::string tree_stats_to_string()
  {
    std::stringstream treestats;
    treestats << "Learn() count per node: ";
    for (const tree_node& n : nodes)
    {
      if (n.is_leaf || n.id >= 16) break;

      treestats << "id=" << n.id << ", #l=" << n.learn_count << "; ";
    }
    return treestats.str();
  }

  std::vector<tree_node> nodes;
  uint32_t root_idx = 0;

private:
  uint32_t _num_leaf_nodes = 0;
  bool _initialized = false;
  uint32_t _depth = 0;
};

struct node_cost
{
  uint32_t node_id;
  float cost;
  node_cost() : node_id(0), cost(0.f) {}
  node_cost(uint32_t node_id, float cost) : node_id(node_id), cost(cost) {}
};

struct cats_tree
{
  bool init(uint32_t num_actions, uint32_t bandwidth) { return _binary_tree.build_tree(num_actions, bandwidth); }
  int32_t learner_count() const { return _binary_tree.internal_node_count(); }
  uint32_t predict(LEARNER::single_learner& base, example& ec)
  {
    const std::vector<tree_node>& nodes = _binary_tree.nodes;

    // Handle degenerate cases of zero node trees
    if (_binary_tree.leaf_node_count() == 0) return 0;
    CB::label saved_label = std::move(ec.l.cb);
    ec.l.simple.label = std::numeric_limits<float>::max();  // says it is a test example
    auto cur_node = nodes[0];

    while (!(cur_node.is_leaf))
    {
      if (cur_node.right_only) { cur_node = nodes[cur_node.right_id]; }
      else if (cur_node.left_only)
      {
        cur_node = nodes[cur_node.left_id];
      }
      else
      {
        ec.partial_prediction = 0.f;
        ec.pred.scalar = 0.f;
        base.predict(ec, cur_node.id);
        if (ec.pred.scalar < 0) { cur_node = nodes[cur_node.left_id]; }
        else
        {
          cur_node = nodes[cur_node.right_id];
        }
      }
    }
    ec.l.cb = saved_label;
    return (cur_node.id - _binary_tree.internal_node_count() + 1);  // 1 to k
  }

  void init_node_costs(v_array<CB::cb_class>& ac);
  const tree_node& get_sibling(const tree_node& tree_node);
  float return_cost(const tree_node& w);
  void learn(LEARNER::single_learner& base, example& ec);
  void set_trace_message(std::ostream* ostrm, bool quiet);
  ~cats_tree()
  {
    if (_trace_stream != nullptr && !_quiet) (*_trace_stream) << tree_stats_to_string() << std::endl;
  }

private:
  uint64_t app_seed = uniform_hash("vw", 2, 0);
  std::string tree_stats_to_string() { return _binary_tree.tree_stats_to_string(); }
  min_depth_binary_tree _binary_tree;
  float _cost_star = 0.f;
  node_cost _a;
  node_cost _b;
  std::ostream* _trace_stream = nullptr;
  bool _quiet = false;
};

inline void predict(cats_tree& ot, LEARNER::single_learner& base, example& ec)
{
  ec.pred.multiclass = ot.predict(base, ec);
}

inline void learn(cats_tree& tree, VW::LEARNER::single_learner& base, example& ec);

}  // namespace cats_tree
}  // namespace VW
