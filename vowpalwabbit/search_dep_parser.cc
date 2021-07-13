// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include "search_dep_parser.h"
#include "gd.h"
#include "cost_sensitive.h"
#include "label_dictionary.h"  // for add_example_namespaces_from_example
#include "vw.h"
#include "vw_exception.h"

using namespace VW::config;

#define val_namespace 100  // valency and distance feature space
#define offset_const 344429
#define arc_hybrid 1
#define arc_eager 2

namespace DepParserTask
{
Search::search_task task = {"dep_parser", run, initialize, finish, setup, nullptr};
}

struct task_data
{
  example ex;
  size_t root_label;
  uint32_t num_label;
  v_array<uint32_t> valid_actions;
  v_array<uint32_t> action_loss;
  v_array<uint32_t> gold_heads;
  v_array<uint32_t> gold_tags;
  v_array<uint32_t> stack;
  v_array<uint32_t> heads;
  v_array<uint32_t> tags;
  v_array<uint32_t> temp;
  v_array<uint32_t> valid_action_temp;
  v_array<action> gold_actions;
  v_array<action> gold_action_temp;
  std::vector<std::pair<action, float>> gold_action_losses;

  // [0]:num_left_arcs, [1]:num_right_arcs; [2]: leftmost_arc, [3]: second_leftmost_arc,
  // [4]:rightmost_arc, [5]: second_rightmost_arc
  std::array<v_array<uint32_t>, 6> children;

  std::array<example*, 13> ec_buf;
  bool old_style_labels;
  bool cost_to_go;
  bool one_learner;
  uint32_t transition_system;
};

namespace DepParserTask
{
using namespace Search;

constexpr action SHIFT = 1;
constexpr action REDUCE_RIGHT = 2;
constexpr action REDUCE_LEFT = 3;
constexpr action REDUCE = 4;
constexpr uint32_t my_null = 9999999; /*representing_default*/

void initialize(Search::search &sch, size_t & /*num_actions*/, options_i &options)
{
  vw &all = sch.get_vw_pointer_unsafe();
  task_data *data = new task_data();
  data->action_loss.resize_but_with_stl_behavior(5);
  sch.set_task_data<task_data>(data);

  option_group_definition new_options("Dependency Parser Options");
  new_options.add(make_option("root_label", data->root_label)
                      .keep()
                      .default_value(8)
                      .help("Ensure that there is only one root in each sentence"));
  new_options.add(make_option("num_label", data->num_label).keep().default_value(12).help("Number of arc labels"));
  new_options.add(make_option("transition_system", data->transition_system)
                      .keep()
                      .default_value(1)
                      .help("1: arc-hybrid 2: arc-eager"));
  new_options.add(make_option("one_learner", data->one_learner)
                      .keep()
                      .help("Using one learner instead of three learners for labeled parser"));
  new_options.add(make_option("cost_to_go", data->cost_to_go)
                      .keep()
                      .help("Estimating cost-to-go matrix based on dynamic oracle rathan than rolling-out"));
  new_options.add(
      make_option("old_style_labels", data->old_style_labels).keep().help("Use old hack of label information"));
  options.add_and_parse(new_options);

  data->ex.feature_space.get_or_create(val_namespace, val_namespace);
  for (size_t i = 1; i < 14; i++)
  {
    auto current_index = static_cast<unsigned char>(i) + 'A';
    data->ex.feature_space.get_or_create(current_index, current_index);
  }
  data->ex.feature_space.get_or_create(constant_namespace, val_namespace);
  data->ex.interactions = &sch.get_vw_pointer_unsafe().interactions;

  if (data->one_learner)
    sch.set_num_learners(1);
  else
    sch.set_num_learners(3);

  std::vector<std::vector<namespace_index>> newpairs{{'B', 'C'}, {'B', 'E'}, {'B', 'B'}, {'C', 'C'}, {'D', 'D'},
      {'E', 'E'}, {'F', 'F'}, {'G', 'G'}, {'E', 'F'}, {'B', 'H'}, {'B', 'J'}, {'E', 'L'}, {'d', 'B'}, {'d', 'C'},
      {'d', 'D'}, {'d', 'E'}, {'d', 'F'}, {'d', 'G'}, {'d', 'd'}};
  std::vector<std::vector<namespace_index>> newtriples{{'E', 'F', 'G'}, {'B', 'E', 'F'}, {'B', 'C', 'E'},
      {'B', 'C', 'D'}, {'B', 'E', 'L'}, {'E', 'L', 'M'}, {'B', 'H', 'I'}, {'B', 'C', 'C'}, {'B', 'E', 'J'},
      {'B', 'E', 'H'}, {'B', 'J', 'K'}, {'B', 'E', 'N'}};

  all.interactions.clear();
  all.interactions.insert(std::end(all.interactions), std::begin(newpairs), std::end(newpairs));
  all.interactions.insert(std::end(all.interactions), std::begin(newtriples), std::end(newtriples));

  if (data->cost_to_go)
    sch.set_options(AUTO_CONDITION_FEATURES | NO_CACHING | ACTION_COSTS);
  else
    sch.set_options(AUTO_CONDITION_FEATURES | NO_CACHING);

  sch.set_label_parser(COST_SENSITIVE::cs_label, [](polylabel &l) -> bool { return l.cs.costs.size() == 0; });
}

void finish(Search::search &sch)
{
  task_data* data = sch.get_task_data<task_data>();
  delete data;
}

void inline add_feature(
    example &ex, uint64_t idx, unsigned char ns, uint64_t mask, uint64_t multiplier, bool /* audit */ = false)
{
  ex.feature_space.get_or_create(ns, ns).push_back(1.0f, (idx * multiplier) & mask);
}

void add_all_features(example &ex, example &src, unsigned char tgt_ns, uint64_t mask, uint64_t multiplier,
    uint64_t offset, bool /* audit */ = false)
{
  for (auto& bucket : src.feature_space)
  {
    for (auto it = bucket.begin(); it != bucket.end(); ++it)
    {
      if (it->index == constant_namespace) { continue; }
      features& tgt_fs = ex.feature_space.get_or_create(tgt_ns, tgt_ns);

      for (feature_index i : it->feats.indicies)
      { tgt_fs.push_back(1.0f, ((i / multiplier + offset) * multiplier) & mask); }
    }
  }
}

void inline reset_ex(example& ex)
{
  ex.num_features = 0;
  ex.reset_total_sum_feat_sq();
  ex.feature_space.clear();
}

// arc-hybrid System.
size_t transition_hybrid(Search::search &sch, uint64_t a_id, uint32_t idx, uint32_t t_id, uint32_t /* n */)
{
  task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t>& heads = data->heads;
  v_array<uint32_t>& stack = data->stack;
  v_array<uint32_t>& gold_heads = data->gold_heads;
  v_array<uint32_t>& gold_tags = data->gold_tags;
  v_array<uint32_t>& tags = data->tags;
  v_array<uint32_t>* children = data->children.data();
  if (a_id == SHIFT)
  {
    stack.push_back(idx);
    return idx + 1;
  }
  else if (a_id == REDUCE_RIGHT)
  {
    uint32_t last = stack.back();
    uint32_t hd = stack[stack.size() - 2];
    heads[last] = hd;
    children[5][hd] = children[4][hd];
    children[4][hd] = last;
    children[1][hd]++;
    tags[last] = t_id;
    sch.loss(gold_heads[last] != heads[last] ? 2 : (gold_tags[last] != t_id) ? 1.f : 0.f);
    assert(!stack.empty());
    stack.pop_back();
    return idx;
  }
  else if (a_id == REDUCE_LEFT)
  {
    size_t last = stack.back();
    uint32_t hd = idx;
    heads[last] = hd;
    children[3][hd] = children[2][hd];
    children[2][hd] = static_cast<uint32_t>(last);
    children[0][hd]++;
    tags[last] = t_id;
    sch.loss(gold_heads[last] != heads[last] ? 2 : (gold_tags[last] != t_id) ? 1.f : 0.f);
    assert(!stack.empty());
    stack.pop_back();
    return idx;
  }
  THROW("transition_hybrid failed");
}

// arc-eager system
size_t transition_eager(Search::search &sch, uint64_t a_id, uint32_t idx, uint32_t t_id, uint32_t n)
{
  task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t>& heads = data->heads;
  v_array<uint32_t>& stack = data->stack;
  v_array<uint32_t>& gold_heads = data->gold_heads;
  v_array<uint32_t>& gold_tags = data->gold_tags;
  v_array<uint32_t>& tags = data->tags;
  v_array<uint32_t>* children = data->children.data();
  if (a_id == SHIFT)
  {
    stack.push_back(idx);
    return idx + 1;
  }
  else if (a_id == REDUCE_RIGHT)
  {
    uint32_t hd = stack.back();
    stack.push_back(idx);
    uint32_t last = idx;
    heads[last] = hd;
    children[5][hd] = children[4][hd];
    children[4][hd] = last;
    children[1][hd]++;
    tags[last] = t_id;
    sch.loss(gold_heads[last] != heads[last] ? 2 : (gold_tags[last] != t_id) ? 1.f : 0.f);
    return idx + 1;
  }
  else if (a_id == REDUCE_LEFT)
  {
    size_t last = stack.back();
    uint32_t hd = (idx > n) ? 0 : idx;
    heads[last] = hd;
    children[3][hd] = children[2][hd];
    children[2][hd] = static_cast<uint32_t>(last);
    children[0][hd]++;
    tags[last] = t_id;
    sch.loss(gold_heads[last] != heads[last] ? 2 : (gold_tags[last] != t_id) ? 1.f : 0.f);
    assert(!stack.empty());
    stack.pop_back();
    return idx;
  }
  else if (a_id == REDUCE)
  {
    assert(!stack.empty());
    stack.pop_back();
    return idx;
  }
  THROW("transition_eager failed");
}

void extract_features(Search::search &sch, uint32_t idx, multi_ex &ec)
{
  vw &all = sch.get_vw_pointer_unsafe();
  task_data *data = sch.get_task_data<task_data>();
  reset_ex(data->ex);
  uint64_t mask = sch.get_mask();
  uint64_t multiplier = static_cast<uint64_t>(all.wpp) << all.weights.stride_shift();

  auto &stack = data->stack;
  auto &tags = data->tags;
  auto &children = data->children;
  auto &temp = data->temp;
  example** ec_buf = data->ec_buf.data();
  example& ex = data->ex;

  size_t n = ec.size();
  bool empty = stack.empty();
  size_t last = empty ? 0 : stack.back();

  for (size_t i = 0; i < 13; i++) ec_buf[i] = nullptr;

  // feature based on the top three examples in stack ec_buf[0]: s1, ec_buf[1]: s2, ec_buf[2]: s3
  for (size_t i = 0; i < 3; i++)
    ec_buf[i] = (stack.size() > i && *(stack.end() - (i + 1)) != 0) ? ec[*(stack.end() - (i + 1)) - 1] : nullptr;

  // features based on examples in string buffer ec_buf[3]: b1, ec_buf[4]: b2, ec_buf[5]: b3
  for (size_t i = 3; i < 6; i++) ec_buf[i] = (idx + (i - 3) - 1 < n) ? ec[idx + i - 3 - 1] : nullptr;

  // features based on the leftmost and the rightmost children of the top element stack ec_buf[6]: sl1, ec_buf[7]: sl2,
  // ec_buf[8]: sr1, ec_buf[9]: sr2;
  for (size_t i = 6; i < 10; i++)
    if (!empty && last != 0 && children[i - 4][last] != 0) ec_buf[i] = ec[children[i - 4][last] - 1];

  // features based on leftmost children of the top element in bufer ec_buf[10]: bl1, ec_buf[11]: bl2
  for (size_t i = 10; i < 12; i++)
    ec_buf[i] = (idx <= n && children[i - 8][idx] != 0) ? ec[children[i - 8][idx] - 1] : nullptr;
  ec_buf[12] = (stack.size() > 1 && *(stack.end() - 2) != 0 && children[2][*(stack.end() - 2)] != 0)
      ? ec[children[2][*(stack.end() - 2)] - 1]
      : nullptr;

  // unigram features
  for (size_t i = 0; i < 13; i++)
  {
    uint64_t additional_offset = static_cast<uint64_t>(i * offset_const);
    if (!ec_buf[i])
      add_feature(ex, static_cast<uint64_t>(438129041) + additional_offset, static_cast<unsigned char>((i + 1) + 'A'),
          mask, multiplier);
    else
      add_all_features(
          ex, *ec_buf[i], 'A' + static_cast<unsigned char>(i + 1), mask, multiplier, additional_offset, false);
  }

  // Other features
  temp.resize_but_with_stl_behavior(10);
  temp[0] = empty ? 0 : (idx > n ? 1 : 2 + std::min(static_cast<uint32_t>(5), idx - static_cast<uint32_t>(last)));
  temp[1] = empty ? 1 : 1 + std::min(static_cast<uint32_t>(5), children[0][last]);
  temp[2] = empty ? 1 : 1 + std::min(static_cast<uint32_t>(5), children[1][last]);
  temp[3] = idx > n ? 1 : 1 + std::min(static_cast<uint32_t>(5), children[0][idx]);
  for (size_t i = 4; i < 8; i++) temp[i] = (!empty && children[i - 2][last] != 0) ? tags[children[i - 2][last]] : 15;
  for (size_t i = 8; i < 10; i++) temp[i] = (idx <= n && children[i - 6][idx] != 0) ? tags[children[i - 6][idx]] : 15;

  uint64_t additional_offset = val_namespace * offset_const;
  for (size_t j = 0; j < 10; j++)
  {
    additional_offset += j * 1023;
    add_feature(ex, temp[j] + additional_offset, val_namespace, mask, multiplier);
  }
  size_t count = 0;
  for (auto& bucket : data->ex)
  {
    for (auto& fs : bucket)
    {
      fs.feats.sum_feat_sq = static_cast<float>(fs.feats.size());
      count += fs.feats.size();
    }
  }

  data->ex.num_features = count;
}

void get_valid_actions(Search::search &sch, v_array<uint32_t> &valid_action, uint64_t idx, uint64_t n,
    uint64_t stack_depth, uint64_t state)
{
  task_data *data = sch.get_task_data<task_data>();
  uint32_t &sys = data->transition_system;
  v_array<uint32_t> &stack = data->stack, &heads = data->heads, &temp = data->temp;
  valid_action.clear();
  if (sys == arc_hybrid)
  {
    if (idx <= n)  // SHIFT
      valid_action.push_back(SHIFT);
    if (stack_depth >= 2)  // RIGHT
      valid_action.push_back(REDUCE_RIGHT);
    if (stack_depth >= 1 && state != 0 && idx <= n)  // LEFT
      valid_action.push_back(REDUCE_LEFT);
  }
  else if (sys == arc_eager)  // assume root is in N+1
  {
    temp.clear();
    for (size_t i = 0; i <= 4; i++) temp.push_back(1);
    if (idx > n)
    {
      temp[SHIFT] = 0;
      temp[REDUCE_RIGHT] = 0;
    }

    if (stack_depth == 0)
      temp[REDUCE] = 0;
    else if (idx <= n + 1 && heads[stack.back()] == my_null)
      temp[REDUCE] = 0;

    if (stack_depth == 0)
    {
      temp[REDUCE_LEFT] = 0;
      temp[REDUCE_RIGHT] = 0;
    }
    else
    {
      if (heads[stack.back()] != my_null) temp[REDUCE_LEFT] = 0;
      if (idx <= n && heads[idx] != my_null) temp[REDUCE_RIGHT] = 0;
    }
    for (uint32_t i = 1; i <= 4; i++)
    {
      if (temp[i]) valid_action.push_back(i);
    }
  }
}

bool is_valid(uint64_t action, const v_array<uint32_t> &valid_actions)
{
  for (size_t i = 0; i < valid_actions.size(); i++)
    if (valid_actions[i] == action) return true;
  return false;
}

void get_eager_action_cost(Search::search &sch, uint32_t idx, uint64_t n)
{
  task_data *data = sch.get_task_data<task_data>();
  auto &action_loss = data->action_loss;
  auto &stack = data->stack;
  auto &gold_heads = data->gold_heads;
  auto &heads = data->heads;
  size_t size = stack.size();
  size_t last = (size == 0) ? 0 : stack.back();
  for (size_t i = 1; i <= 4; i++) action_loss[i] = 0;
  if (!stack.empty())
    for (size_t i = 0; i < size; i++)
    {
      if (gold_heads[stack[i]] == idx && heads[stack[i]] == my_null)
      {
        action_loss[SHIFT] += 1;
        action_loss[REDUCE_RIGHT] += 1;
      }
      if (idx <= n && (gold_heads[idx] == stack[i]))
      {
        if (stack[i] != 0) action_loss[SHIFT] += 1;
        if (stack[i] != last) action_loss[REDUCE_RIGHT] += 1;
      }
    }
  for (size_t i = idx; i <= n + 1; i++)
  {
    if (i <= n && gold_heads[i] == last)
    {
      action_loss[REDUCE] += 1;
      action_loss[REDUCE_LEFT] += 1;
    }
    if (i != idx && gold_heads[last] == i) action_loss[REDUCE_LEFT] += 1;
  }
  // if(size>0  && idx <=n && gold_heads[last] == 0 && stack[0] ==0) //should not fire
  //  action_loss[REDUCE_LEFT] +=1;

  if (gold_heads[idx] > idx || (gold_heads[idx] == 0 && size > 0 && stack[0] != 0)) action_loss[REDUCE_RIGHT] += 1;
}

void get_hybrid_action_cost(Search::search &sch, size_t idx, uint64_t n)
{
  task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t> &action_loss = data->action_loss, &stack = data->stack, &gold_heads = data->gold_heads;
  size_t size = stack.size();
  size_t last = (size == 0) ? 0 : stack.back();

  for (size_t i = 1; i <= 3; i++) action_loss[i] = 0;
  if (!stack.empty())
    for (size_t i = 0; i < size - 1; i++)
      if (idx <= n && (gold_heads[stack[i]] == idx || gold_heads[idx] == stack[i])) action_loss[SHIFT] += 1;

  if (size > 0 && gold_heads[last] == idx) action_loss[SHIFT] += 1;

  for (size_t i = idx + 1; i <= n; i++)
    if (gold_heads[i] == last || gold_heads[last] == i) action_loss[REDUCE_LEFT] += 1;
  if (size > 0 && idx <= n && gold_heads[idx] == last) action_loss[REDUCE_LEFT] += 1;
  if (size >= 2 && gold_heads[last] == stack[size - 2]) action_loss[REDUCE_LEFT] += 1;

  if (gold_heads[last] >= idx) action_loss[REDUCE_RIGHT] += 1;

  for (size_t i = idx; i <= n; i++)
    if (gold_heads[i] == static_cast<uint32_t>(last)) action_loss[REDUCE_RIGHT] += 1;
}

void get_cost_to_go_losses(Search::search &sch, std::vector<std::pair<action, float>> &gold_action_losses,
    uint32_t left_label, uint32_t right_label)
{
  task_data *data = sch.get_task_data<task_data>();
  bool &one_learner = data->one_learner;
  uint32_t &sys = data->transition_system;
  auto &action_loss = data->action_loss;
  auto &valid_actions = data->valid_actions;
  uint32_t &num_label = data->num_label;
  gold_action_losses.clear();

  if (one_learner)
  {
    if (is_valid(SHIFT, valid_actions))
      gold_action_losses.push_back(std::make_pair(SHIFT, static_cast<float>(action_loss[SHIFT])));
    for (uint32_t i = 2; i <= 3; i++)
      if (is_valid(i, valid_actions))
      {
        for (uint32_t j = 1; j <= num_label; j++)
          if (sys == arc_eager || j != data->root_label)
            gold_action_losses.push_back(std::make_pair((1 + j + (i - 2) * num_label),
                action_loss[i] + static_cast<float>(j != (i == REDUCE_LEFT ? left_label : right_label))));
      }
    if (sys == arc_eager && is_valid(REDUCE, valid_actions))
      gold_action_losses.push_back(std::make_pair(2 + num_label * 2, static_cast<float>(action_loss[REDUCE])));
  }
  else
  {
    for (action i = 1; i <= 3; i++)
      if (is_valid(i, valid_actions))
        gold_action_losses.push_back(std::make_pair(i, static_cast<float>(action_loss[i])));
    if (sys == arc_eager && is_valid(REDUCE, valid_actions))
      gold_action_losses.push_back(std::make_pair(REDUCE, static_cast<float>(action_loss[REDUCE])));
  }
}

void get_gold_actions(Search::search &sch, uint32_t idx, uint64_t /* n */, v_array<action> &gold_actions)
{
  task_data *data = sch.get_task_data<task_data>();
  auto &action_loss = data->action_loss;
  auto &stack = data->stack;
  auto &gold_heads = data->gold_heads;
  auto &valid_actions = data->valid_actions;
  gold_actions.clear();
  size_t size = stack.size();
  size_t last = (size == 0) ? 0 : stack.back();
  uint32_t &sys = data->transition_system;

  if (sys == arc_hybrid && is_valid(SHIFT, valid_actions) && (stack.empty() || gold_heads[idx] == last))
  {
    gold_actions.push_back(SHIFT);
    return;
  }

  if (sys == arc_hybrid && is_valid(REDUCE_LEFT, valid_actions) && gold_heads[last] == idx)
  {
    gold_actions.push_back(REDUCE_LEFT);
    return;
  }
  size_t best_action = 1;
  size_t count = 0;
  for (uint32_t i = 1; i <= 4; i++)
  {
    if (i == 4 && sys == arc_hybrid) continue;
    if (action_loss[i] < action_loss[best_action] && is_valid(i, valid_actions))
    {
      best_action = i;
      count = 1;
      gold_actions.clear();
      gold_actions.push_back(i);
    }
    else if (action_loss[i] == action_loss[best_action] && is_valid(i, valid_actions))
    {
      count++;
      gold_actions.push_back(i);
    }
  }
}

void convert_to_onelearner_actions(Search::search &sch, v_array<action> &actions, v_array<action> &actions_onelearner,
    uint32_t left_label, uint32_t right_label)
{
  task_data *data = sch.get_task_data<task_data>();
  uint32_t &sys = data->transition_system;
  uint32_t &num_label = data->num_label;
  actions_onelearner.clear();
  if (is_valid(SHIFT, actions)) actions_onelearner.push_back(SHIFT);
  if (sys == arc_eager && is_valid(REDUCE, actions)) actions_onelearner.push_back(2 + 2 * num_label);
  if (left_label != my_null && is_valid(REDUCE_RIGHT, actions)) actions_onelearner.push_back(1 + right_label);
  if (left_label != my_null && is_valid(REDUCE_LEFT, actions)) actions_onelearner.push_back(1 + left_label + num_label);
  if (left_label == my_null && is_valid(REDUCE_RIGHT, actions))
    for (uint32_t i = 0; i < num_label; i++)
      if (i != data->root_label - 1) actions_onelearner.push_back(i + 2);
  if (left_label == my_null && is_valid(REDUCE_LEFT, actions))
    for (uint32_t i = 0; i < num_label; i++)
      if (sys == arc_eager || i != data->root_label - 1)
        actions_onelearner.push_back(static_cast<uint32_t>(i + 2 + num_label));
}

void setup(Search::search &sch, multi_ex &ec)
{
  task_data *data = sch.get_task_data<task_data>();
  auto &gold_heads = data->gold_heads;
  auto &heads = data->heads;
  auto &gold_tags = data->gold_tags;
  auto &tags = data->tags;
  size_t n = ec.size();
  heads.resize_but_with_stl_behavior(n + 1);
  tags.resize_but_with_stl_behavior(n + 1);
  gold_heads.clear();
  gold_heads.push_back(0);
  gold_tags.clear();
  gold_tags.push_back(0);
  for (size_t i = 0; i < n; i++)
  {
    const auto &costs = ec[i]->l.cs.costs;
    uint32_t head, tag;
    if (data->old_style_labels)
    {
      uint32_t label = costs[0].class_index;
      head = (label & 255) - 1;
      tag = label >> 8;
    }
    else
    {
      head = (costs.size() == 0) ? 0 : costs[0].class_index;
      tag = (costs.size() <= 1) ? static_cast<uint32_t>(data->root_label) : costs[1].class_index;
    }
    if (tag > data->num_label) THROW("invalid label " << tag << " which is > num actions=" << data->num_label);

    gold_heads.push_back(head);
    gold_tags.push_back(tag);
    heads[i + 1] = my_null;
    tags[i + 1] = my_null;
  }
  for (size_t i = 0; i < 6; i++) data->children[i].resize_but_with_stl_behavior(n + static_cast<size_t>(1));
}

void run(Search::search &sch, multi_ex &ec)
{
  task_data *data = sch.get_task_data<task_data>();
  v_array<uint32_t> &stack = data->stack, &gold_heads = data->gold_heads, &valid_actions = data->valid_actions,
                    &heads = data->heads, &gold_tags = data->gold_tags, &tags = data->tags,
                    &valid_action_temp = data->valid_action_temp;
  v_array<uint32_t> &gold_action_temp = data->gold_action_temp;
  std::vector<std::pair<action, float>> &gold_action_losses = data->gold_action_losses;
  v_array<action> &gold_actions = data->gold_actions;
  bool &cost_to_go = data->cost_to_go, &one_learner = data->one_learner;
  uint32_t &num_label = data->num_label;
  uint32_t &sys = data->transition_system;
  uint32_t n = static_cast<uint32_t>(ec.size());
  uint32_t left_label, right_label;
  stack.clear();
  stack.push_back((data->root_label == 0 && sys == arc_hybrid) ? 0 : 1);
  for (size_t i = 0; i < 6; i++)
    for (size_t j = 0; j < n + 1; j++) data->children[i][j] = 0;
  for (size_t i = 0; i < n; i++)
  {
    heads[i + 1] = my_null;
    tags[i + 1] = my_null;
  }
  ptag count = 1;
  uint32_t idx = ((data->root_label == 0 && sys == arc_hybrid) ? 1 : 2);
  Search::predictor P(sch, static_cast<ptag>(0));
  while (true)
  {
    if (sys == arc_hybrid && stack.size() <= 1 && idx > n)
      break;
    else if (sys == arc_eager && stack.size() == 0 && idx > n)
      break;
    bool computedFeatures = false;
    if (sch.predictNeedsExample())
    {
      extract_features(sch, idx, ec);
      computedFeatures = true;
    }
    get_valid_actions(
        sch, valid_actions, idx, n, static_cast<uint64_t>(stack.size()), stack.empty() ? 0 : stack.back());
    if (sys == arc_hybrid)
      get_hybrid_action_cost(sch, idx, n);
    else if (sys == arc_eager)
      get_eager_action_cost(sch, idx, n);

    // get gold tag labels
    left_label = stack.empty() ? my_null : gold_tags[stack.back()];
    if (sys == arc_hybrid)
      right_label = stack.empty() ? my_null : gold_tags[stack.back()];
    else if (sys == arc_eager)
      right_label = idx <= n ? gold_tags[idx] : static_cast<uint32_t>(data->root_label);
    else
      THROW("unknown transition system");

    uint32_t a_id = 0, t_id = 0;
    if (one_learner)
    {
      if (cost_to_go)
      {
        get_cost_to_go_losses(sch, gold_action_losses, left_label, right_label);
        a_id = P.set_tag(count)
                   .set_input(data->ex)
                   .set_allowed(gold_action_losses)
                   .set_condition_range(count - 1, sch.get_history_length(), 'p')
                   .set_learner_id(0)
                   .predict();
      }
      else
      {
        get_gold_actions(sch, idx, n, gold_actions);
        convert_to_onelearner_actions(sch, gold_actions, gold_action_temp, left_label, right_label);
        convert_to_onelearner_actions(sch, valid_actions, valid_action_temp, my_null, my_null);
        a_id = P.set_tag(count)
                   .set_input(data->ex)
                   .set_oracle(gold_action_temp)
                   .set_allowed(valid_action_temp)
                   .set_condition_range(count - 1, sch.get_history_length(), 'p')
                   .set_learner_id(0)
                   .predict();
      }
      if (a_id == SHIFT)
        t_id = 0;
      else if (a_id == 2 * num_label + 2)
      {
        t_id = 0;
        a_id = REDUCE;
      }
      else if (a_id > 1 && a_id - 1 <= num_label)
      {
        t_id = a_id - 1;
        a_id = REDUCE_RIGHT;
      }
      else
      {
        t_id = static_cast<uint64_t>(a_id) - num_label - 1;
        a_id = REDUCE_LEFT;
      }
    }
    else
    {
      if (cost_to_go)
      {
        get_cost_to_go_losses(sch, gold_action_losses, left_label, right_label);
        a_id = P.set_tag(count)
                   .set_input(data->ex)
                   .set_allowed(gold_action_losses)
                   .set_condition_range(count - 1, sch.get_history_length(), 'p')
                   .set_learner_id(0)
                   .predict();
      }
      else
      {
        get_gold_actions(sch, idx, n, gold_actions);
        a_id = P.set_tag(count)
                   .set_input(data->ex)
                   .set_oracle(gold_actions)
                   .set_allowed(valid_actions)
                   .set_condition_range(count - 1, sch.get_history_length(), 'p')
                   .set_learner_id(0)
                   .predict();
      }

      // Predict the next action {SHIFT, REDUCE_LEFT, REDUCE_RIGHT}
      count++;

      if (a_id != SHIFT && a_id != REDUCE)
      {
        if ((!computedFeatures) && sch.predictNeedsExample()) extract_features(sch, idx, ec);

        if (cost_to_go)
        {
          gold_action_losses.clear();
          for (size_t i = 1; i <= data->num_label; i++)
            gold_action_losses.push_back(
                std::make_pair(static_cast<action>(i), i != (a_id == REDUCE_LEFT ? left_label : right_label)));
          t_id = P.set_tag(count)
                     .set_input(data->ex)
                     .set_allowed(gold_action_losses)
                     .set_condition_range(count - 1, sch.get_history_length(), 'p')
                     .set_learner_id(a_id - 1)
                     .predict();
        }
        else
        {
          t_id = P.set_tag(count)
                     .set_input(data->ex)
                     .set_oracle(a_id == REDUCE_LEFT ? left_label : right_label)
                     .erase_alloweds()
                     .set_condition_range(count - 1, sch.get_history_length(), 'p')
                     .set_learner_id(a_id - 1)
                     .predict();
        }
      }
    }
    count++;
    if (sys == arc_hybrid)
      idx = static_cast<uint32_t>(transition_hybrid(sch, a_id, idx, t_id, n));
    else if (sys == arc_eager)
      idx = static_cast<uint32_t>(transition_eager(sch, a_id, idx, t_id, n));
  }
  if (sys == arc_hybrid)
  {
    heads[stack.back()] = 0;
    tags[stack.back()] = static_cast<uint32_t>(data->root_label);
    sch.loss((gold_heads[stack.back()] != heads[stack.back()]));
  }
  if (sch.output().good())
    for (size_t i = 1; i <= n; i++) sch.output() << (heads[i]) << ":" << tags[i] << std::endl;
}
}  // namespace DepParserTask
