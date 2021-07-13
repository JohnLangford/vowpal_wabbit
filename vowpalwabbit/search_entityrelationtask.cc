// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include "search_entityrelationtask.h"
#include "vw.h"

#include "io/logger.h"

using namespace VW::config;
namespace logger = VW::io::logger;

#define R_NONE 10      // label for NONE relation
#define LABEL_SKIP 11  // label for SKIP

constexpr size_t NUM_LDF_ENTITY_EXAMPLES = 10;

namespace EntityRelationTask
{
Search::search_task task = {"entity_relation", run, initialize, finish, nullptr, nullptr};
}

namespace EntityRelationTask
{
using namespace Search;
namespace CS = COST_SENSITIVE;

void update_example_indicies(bool audit, example* ec, uint64_t mult_amount, uint64_t plus_amount);

struct task_data
{
  float relation_none_cost;
  float entity_cost;
  float relation_cost;
  float skip_cost;
  bool constraints;
  bool allow_skip;
  v_array<uint32_t> y_allowed_entity;
  v_array<uint32_t> y_allowed_relation;
  size_t search_order;
  example* ldf_entity;
  example* ldf_relation;
};

void initialize(Search::search& sch, size_t& /*num_actions*/, options_i& options)
{
  task_data* my_task_data = new task_data();
  sch.set_task_data<task_data>(my_task_data);

  option_group_definition new_options("Entity Relation Options");
  new_options
      .add(make_option("relation_cost", my_task_data->relation_cost).keep().default_value(1.f).help("Relation Cost"))
      .add(make_option("entity_cost", my_task_data->entity_cost).keep().default_value(1.f).help("Entity Cost"))
      .add(make_option("constraints", my_task_data->constraints).keep().help("Use Constraints"))
      .add(make_option("relation_none_cost", my_task_data->relation_none_cost)
               .keep()
               .default_value(0.5f)
               .help("None Relation Cost"))
      .add(make_option("skip_cost", my_task_data->skip_cost)
               .keep()
               .default_value(0.01f)
               .help("Skip Cost (only used when search_order = skip"))
      .add(make_option("search_order", my_task_data->search_order)
               .keep()
               .default_value(0)
               .help("Search Order 0: EntityFirst 1: Mix 2: Skip 3: EntityFirst(LDF)"));
  options.add_and_parse(new_options);

  // setup entity and relation labels
  // Entity label 1:E_Other 2:E_Peop 3:E_Org 4:E_Loc
  // Relation label 5:R_Live_in 6:R_OrgBased_in 7:R_Located_in 8:R_Work_For 9:R_Kill 10:R_None
  for (int i = 1; i < 5; i++) my_task_data->y_allowed_entity.push_back(i);

  for (int i = 5; i < 11; i++) my_task_data->y_allowed_relation.push_back(i);

  my_task_data->allow_skip = false;

  if (my_task_data->search_order != 3 && my_task_data->search_order != 4) { sch.set_options(0); }
  else
  {
    example* ldf_examples = VW::alloc_examples(NUM_LDF_ENTITY_EXAMPLES);
    CS::wclass default_wclass = {0., 0, 0., 0.};
    for (size_t a = 0; a < NUM_LDF_ENTITY_EXAMPLES; a++)
    {
      ldf_examples[a].l.cs.costs.push_back(default_wclass);
      ldf_examples[a].interactions = &sch.get_vw_pointer_unsafe().interactions;
    }
    my_task_data->ldf_entity = ldf_examples;
    my_task_data->ldf_relation = ldf_examples + 4;
    sch.set_options(Search::IS_LDF);
  }

  sch.set_num_learners(2);
  if (my_task_data->search_order == 4) sch.set_num_learners(3);
}

void finish(Search::search& sch)
{
  task_data* my_task_data = sch.get_task_data<task_data>();
  if (my_task_data->search_order == 3) { VW::dealloc_examples(my_task_data->ldf_entity, NUM_LDF_ENTITY_EXAMPLES); }
  delete my_task_data;
}  // if we had task data, we'd want to free it here

bool check_constraints(size_t ent1_id, size_t ent2_id, size_t rel_id)
{
  size_t valid_ent1_id[] = {2, 3, 4, 2, 2};  // encode the valid entity-relation combinations
  size_t valid_ent2_id[] = {4, 4, 4, 3, 2};
  if (rel_id - 5 == 5) return true;
  if (valid_ent1_id[rel_id - 5] == ent1_id && valid_ent2_id[rel_id - 5] == ent2_id) return true;
  return false;
}

void decode_tag(v_array<char> tag, char& type, int& id1, int& id2)
{
  std::string s1;
  std::string s2;
  type = tag[0];
  size_t idx = 2;
  while (idx < tag.size() && tag[idx] != '_' && tag[idx] != '\0')
  {
    s1.push_back(tag[idx]);
    idx++;
  }
  id1 = atoi(s1.c_str());
  idx++;
  assert(type == 'R');
  while (idx < tag.size() && tag[idx] != '_' && tag[idx] != '\0')
  {
    s2.push_back(tag[idx]);
    idx++;
  }
  id2 = atoi(s2.c_str());
}

size_t predict_entity(
    Search::search& sch, example* ex, v_array<size_t>& /*predictions*/, ptag my_tag, bool isLdf = false)
{
  task_data* my_task_data = sch.get_task_data<task_data>();
  size_t prediction;
  if (my_task_data->allow_skip)
  {
    v_array<uint32_t> star_labels;
    star_labels.push_back(ex->l.multi.label);
    star_labels.push_back(LABEL_SKIP);
    my_task_data->y_allowed_entity.push_back(LABEL_SKIP);
    prediction = Search::predictor(sch, my_tag)
                     .set_input(*ex)
                     .set_oracle(star_labels)
                     .set_allowed(my_task_data->y_allowed_entity)
                     .set_learner_id(1)
                     .predict();
    my_task_data->y_allowed_entity.pop_back();
  }
  else
  {
    if (isLdf)
    {
      for (uint32_t a = 0; a < 4; a++)
      {
        VW::copy_example_data(&my_task_data->ldf_entity[a], ex);
        update_example_indicies(true, &my_task_data->ldf_entity[a], 28904713, 4832917 * static_cast<uint64_t>(a + 1));
        CS::label& lab = my_task_data->ldf_entity[a].l.cs;
        lab.costs[0].x = 0.f;
        lab.costs[0].class_index = a;
        lab.costs[0].partial_prediction = 0.f;
        lab.costs[0].wap_value = 0.f;
      }
      prediction = Search::predictor(sch, my_tag)
                       .set_input(my_task_data->ldf_entity, 4)
                       .set_oracle(ex->l.multi.label - 1)
                       .set_learner_id(1)
                       .predict() +
          1;
    }
    else
    {
      prediction = Search::predictor(sch, my_tag)
                       .set_input(*ex)
                       .set_oracle(ex->l.multi.label)
                       .set_allowed(my_task_data->y_allowed_entity)
                       .set_learner_id(0)
                       .predict();
    }
  }

  // record loss
  float loss = 0.0;
  if (prediction == LABEL_SKIP) { loss = my_task_data->skip_cost; }
  else if (prediction != ex->l.multi.label)
    loss = my_task_data->entity_cost;
  sch.loss(loss);
  return prediction;
}
size_t predict_relation(Search::search& sch, example* ex, v_array<size_t>& predictions, ptag my_tag, bool isLdf = false)
{
  char type;
  int id1, id2;
  task_data* my_task_data = sch.get_task_data<task_data>();
  size_t hist[2];
  decode_tag(ex->tag, type, id1, id2);
  v_array<uint32_t> constrained_relation_labels;
  if (my_task_data->constraints && predictions[id1] != 0 && predictions[id2] != 0)
  {
    hist[0] = predictions[id1];
    hist[1] = predictions[id2];
  }
  else
  {
    hist[0] = 0;
    hist[1] = 0;
  }
  for (size_t j = 0; j < my_task_data->y_allowed_relation.size(); j++)
  {
    if (!my_task_data->constraints || hist[0] == static_cast<size_t>(0) ||
        check_constraints(hist[0], hist[1], my_task_data->y_allowed_relation[j]))
      constrained_relation_labels.push_back(my_task_data->y_allowed_relation[j]);
  }

  size_t prediction;
  if (my_task_data->allow_skip)
  {
    v_array<uint32_t> star_labels;
    star_labels.push_back(ex->l.multi.label);
    star_labels.push_back(LABEL_SKIP);
    constrained_relation_labels.push_back(LABEL_SKIP);
    prediction = Search::predictor(sch, my_tag)
                     .set_input(*ex)
                     .set_oracle(star_labels)
                     .set_allowed(constrained_relation_labels)
                     .set_learner_id(2)
                     .add_condition(id1, 'a')
                     .add_condition(id2, 'b')
                     .predict();
    constrained_relation_labels.pop_back();
  }
  else
  {
    if (isLdf)
    {
      int correct_label = 0;  // if correct label is not in the set, use the first one
      for (size_t a = 0; a < constrained_relation_labels.size(); a++)
      {
        VW::copy_example_data(&my_task_data->ldf_relation[a], ex);
        update_example_indicies(true, &my_task_data->ldf_relation[a], 28904713,
            4832917 * static_cast<uint64_t>(constrained_relation_labels[a]));
        CS::label& lab = my_task_data->ldf_relation[a].l.cs;
        lab.costs[0].x = 0.f;
        lab.costs[0].class_index = constrained_relation_labels[a];
        lab.costs[0].partial_prediction = 0.f;
        lab.costs[0].wap_value = 0.f;
        if (constrained_relation_labels[a] == ex->l.multi.label) { correct_label = static_cast<int>(a); }
      }
      size_t pred_pos = Search::predictor(sch, my_tag)
                            .set_input(my_task_data->ldf_relation, constrained_relation_labels.size())
                            .set_oracle(correct_label)
                            .set_learner_id(2)
                            .predict();
      prediction = constrained_relation_labels[pred_pos];
    }
    else
    {
      prediction = Search::predictor(sch, my_tag)
                       .set_input(*ex)
                       .set_oracle(ex->l.multi.label)
                       .set_allowed(constrained_relation_labels)
                       .set_learner_id(1)
                       .predict();
    }
  }

  float loss = 0.0;
  if (prediction == LABEL_SKIP) { loss = my_task_data->skip_cost; }
  else if (prediction != ex->l.multi.label)
  {
    if (ex->l.multi.label == R_NONE) { loss = my_task_data->relation_none_cost; }
    else
    {
      loss = my_task_data->relation_cost;
    }
  }
  sch.loss(loss);
  return prediction;
}

void entity_first_decoding(Search::search& sch, multi_ex& ec, v_array<size_t>& predictions, bool isLdf = false)
{
  // ec.size = #entity + #entity*(#entity-1)/2
  size_t n_ent = static_cast<size_t>(std::sqrt(ec.size() * 8 + 1) - 1) / 2;
  // Do entity recognition first
  for (size_t i = 0; i < ec.size(); i++)
  {
    if (i < n_ent)
      predictions[i] = predict_entity(sch, ec[i], predictions, static_cast<ptag>(i), isLdf);
    else
      predictions[i] = predict_relation(sch, ec[i], predictions, static_cast<ptag>(i), isLdf);
  }
}

void er_mixed_decoding(Search::search& sch, multi_ex& ec, v_array<size_t>& predictions)
{
  // ec.size = #entity + #entity*(#entity-1)/2
  uint32_t n_ent = static_cast<uint32_t>((std::sqrt(ec.size() * 8 + 1) - 1) / 2);
  for (uint32_t t = 0; t < ec.size(); t++)
  {
    // Do entity recognition first
    uint32_t count = 0;
    for (ptag i = 0; i < n_ent; i++)
    {
      if (count == t)
      {
        predictions[i] = predict_entity(sch, ec[i], predictions, i);
        break;
      }
      count++;
      for (uint32_t j = 0; j < i; j++)
      {
        if (count == t)
        {
          ptag rel_index = static_cast<ptag>(n_ent + (2 * n_ent - j - 1) * j / 2 + i - j - 1);
          predictions[rel_index] = predict_relation(sch, ec[rel_index], predictions, rel_index);
          break;
        }
        count++;
      }
    }
  }
}

void er_allow_skip_decoding(Search::search& sch, multi_ex& ec, v_array<size_t>& predictions)
{
  task_data* my_task_data = sch.get_task_data<task_data>();
  // ec.size = #entity + #entity*(#entity-1)/2
  size_t n_ent = static_cast<size_t>(std::sqrt(ec.size() * 8 + 1) - 1) / 2;

  bool must_predict = false;
  size_t n_predicts = 0;
  size_t p_n_predicts = 0;
  my_task_data->allow_skip = true;

  // loop until all the entity and relation types are predicted
  for (ptag t = 0;; t++)
  {
    ptag i = t % static_cast<uint32_t>(ec.size());
    if (n_predicts == ec.size()) break;

    if (predictions[i] == 0)
    {
      if (must_predict) { my_task_data->allow_skip = false; }
      size_t prediction = 0;
      if (i < n_ent)  // do entity recognition
      { prediction = predict_entity(sch, ec[i], predictions, i); }
      else  // do relation recognition
      {
        prediction = predict_relation(sch, ec[i], predictions, i);
      }

      if (prediction != LABEL_SKIP)
      {
        predictions[i] = prediction;
        n_predicts++;
      }

      if (must_predict)
      {
        my_task_data->allow_skip = true;
        must_predict = false;
      }
    }

    if (i == ec.size() - 1)
    {
      if (n_predicts == p_n_predicts) { must_predict = true; }
      p_n_predicts = n_predicts;
    }
  }
}

void run(Search::search& sch, multi_ex& ec)
{
  task_data* my_task_data = sch.get_task_data<task_data>();

  v_array<size_t> predictions;
  for (size_t i = 0; i < ec.size(); i++) { predictions.push_back(0); }

  switch (my_task_data->search_order)
  {
    case 0:
      entity_first_decoding(sch, ec, predictions, false);
      break;
    case 1:
      er_mixed_decoding(sch, ec, predictions);
      break;
    case 2:
      er_allow_skip_decoding(sch, ec, predictions);
      break;
    case 3:
      entity_first_decoding(sch, ec, predictions, true);  // LDF = true
      break;
    default:
      logger::errlog_error("search order {} is undefined", my_task_data->search_order);
  }

  for (size_t i = 0; i < ec.size(); i++)
  {
    if (sch.output().good()) sch.output() << predictions[i] << ' ';
  }
}
// this is totally bogus for the example -- you'd never actually do this!
void update_example_indicies(bool /* audit */, example* ec, uint64_t mult_amount, uint64_t plus_amount)
{
  for (auto& bucket : *ec)
  {
    for (auto& fs : bucket)
    {
      for (feature_index& idx : fs.feats.indicies) { idx = ((idx * mult_amount) + plus_amount); }
    }
  }
}
}  // namespace EntityRelationTask
