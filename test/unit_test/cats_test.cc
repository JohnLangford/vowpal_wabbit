#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "cats.h"
#include "memory.h"
#include "vw.h"
#include "test_common.h"

using namespace VW::continuous_action::cats;

BOOST_AUTO_TEST_CASE(cats_test_get_loss_zero_for_bad_prediction)
{
  auto data = VW::make_unique<cats>(nullptr, 8, 3, 0, 32, true); 

  VW::cb_continuous::continuous_label cont_label;
  cont_label.costs = v_init<VW::cb_continuous::continuous_label_elm>();
  float action = 10.0f;
  float cost = 1.0f;
  float pdf_value = 0.166666672f;
  cont_label.costs.push_back({action, cost, pdf_value});

  // predicted action not near label action, plus bandwidth too small to provide overlap after smoothing
  float predicted_action = 32.0f;
  auto loss = data->get_loss(cont_label, predicted_action);

  BOOST_CHECK_EQUAL(loss, 0.0f);

  cont_label.costs.delete_v();
}

BOOST_AUTO_TEST_CASE(cats_test_get_loss_not_zero_for_bad_prediction_and_large_b)
{
  auto data = VW::make_unique<cats>(nullptr, 8, 30, 0, 32, true);

  VW::cb_continuous::continuous_label cont_label;
  cont_label.costs = v_init<VW::cb_continuous::continuous_label_elm>();
  float action = 10.0f;
  float cost = 1.0f;
  float pdf_value = 0.166666672f;
  cont_label.costs.push_back({action, cost, pdf_value});

  // predicted action not near label action, but the large bandwidth provides overlap with the logged action
  float predicted_action = 32.0f;
  // centre of predicted action will be 30.0f
  // label is 10.0f and bandiwdth is 30.0f
  // so if: action - bandwidth <= centre <= action + bandwidth then we calculate the loss
  // bandwidth_range = min(max_value, centre + bandwidth) - max(min_value, centre - bandwidth)
  // bandiwdth_range = min(32, 33) - max(0, 0) <=> bandwidth_range = 32 - 0 <=> bandwidth_range = 32
  // loss = cost / (pdf_value * bandwidth_range) <=> loss = 1.0f / (0.166666672f * 32) <=> loss = 0.1875
  auto loss = data->get_loss(cont_label, predicted_action);

  BOOST_CHECK_CLOSE(loss, 0.1875, FLOAT_TOL);

  cont_label.costs.delete_v();
}

BOOST_AUTO_TEST_CASE(cats_test_get_loss_for_good_prediction_and_small_b_not_close_to_range_edges)
{
  auto data = VW::make_unique<cats>(nullptr, 4, 4, 1, 34, true);

  // continous_range = 34 - 1 = 33
  // unit_range = continuous_range / num_action = 33 / 4 = 8.25

  VW::cb_continuous::continuous_label cont_label;
  cont_label.costs = v_init<VW::cb_continuous::continuous_label_elm>();
  float action = 17.0f;
  float cost = 1.0f;
  float pdf_value = 0.125f;
  cont_label.costs.push_back({action, cost, pdf_value});

  // predicted action not near label action, but the large bandwidth provides overlap with the logged action
  float predicted_action = 15.1f;
  // centre of predicted action will be 13.3f
  // label/action is 17.0f and bandiwdth is 2.0f
  // so if: action - bandwidth <= centre <= action + bandwidth then we calculate the loss (13 <= 13.3 <= 21)
  // bandwidth_range = min(max_value, centre + bandwidth) - max(min_value, centre - bandwidth)
  // bandiwdth_range = min(34, 17) - max(1, 9) <=> bandwidth_range = 17 - 9 <=> bandwidth_range = 8
  // loss = cost / (pdf_value * bandwidth_range) <=> loss = 1.0f / (0.125f * 8) <=> loss = 1
  auto loss = data->get_loss(cont_label, predicted_action);

  BOOST_CHECK_EQUAL(loss, 1.0f);

  cont_label.costs.delete_v();
}

BOOST_AUTO_TEST_CASE(cats_test_get_loss_for_good_prediction_and_small_b_close_to_range_edges)
{
  auto data = VW::make_unique<cats>(nullptr, 4, 4, 1, 34, true);

  // continous_range = 34 - 1 = 33
  // unit_range = continuous_range / num_action = 33 / 4 = 8.25

  VW::cb_continuous::continuous_label cont_label;
  cont_label.costs = v_init<VW::cb_continuous::continuous_label_elm>();
  float action = 33.0f;
  float cost = 1.0f;
  float pdf_value = 0.125f;
  cont_label.costs.push_back({action, cost, pdf_value});

  // predicted action not near label action, but the large bandwidth provides overlap with the logged action
  float predicted_action = 26.1f;
  // centre of predicted action will be 29.875
  // label/action is 33.0f and bandiwdth is 4.0f
  // so if: action - bandwidth <= centre <= action + bandwidth then we calculate the loss (29 <= 29.875 <= 37)
  // bandwidth_range = min(max_value, centre + bandwidth) - max(min_value, centre - bandwidth)
  // bandiwdth_range = min(34, 33.875) - max(1, 25.875) <=> bandwidth_range = 33.875 - 25.875 <=> bandwidth_range = 8 (2
  // * bandwidth) loss = cost / (pdf_value * bandwidth_range) <=> loss = 1.0f / (0.125f * 8) <=> loss = 1
  auto loss = data->get_loss(cont_label, predicted_action);

  BOOST_CHECK_CLOSE(loss, 1.0f, FLOAT_TOL);

  cont_label.costs.delete_v();
}

BOOST_AUTO_TEST_CASE(cats_test_get_loss_with_default_bandwidth)
{
  auto data = VW::make_unique<cats>(nullptr, 8, 0, 0, 32, false);
  data->min_value = 0;
  data->max_value = 32;
  data->num_actions = 8;

  float leaf_width = (data->max_value - data->min_value) / (data->num_actions);  // aka unit range
  float half_leaf_width = leaf_width / 2.f;
  data->bandwidth = half_leaf_width;

  // continous_range = 32 - 0 = 32
  // unit_range = continuous_range / num_action = 32 / 8 = 4

  VW::cb_continuous::continuous_label cont_label;
  cont_label.costs = v_init<VW::cb_continuous::continuous_label_elm>();
  float action = 32.0f;
  float cost = 1.0f;
  float pdf_value = 0.25f;
  cont_label.costs.push_back({action, cost, pdf_value});

  // predicted action in corner leaf
  float predicted_action = 32.0f;
  // centre of predicted action will be 30.0
  // label/action is 32.0f and bandiwdth is 2.0f
  // so if: action - bandwidth <= centre <= action + bandwidth then we calculate the loss (30 <= 30 <= 34)
  // bandwidth_range = min(max_value, centre + bandwidth) - max(min_value, centre - bandwidth)
  // bandiwdth_range = min(32, 32) - max(0, 28) <=> bandwidth_range = 32 - 28 <=> bandwidth_range = 4 (i.e. 2 *
  // bandwidth) loss = cost / (pdf_value * bandwidth_range) <=> loss = 1.0f / (0.25f * 4) <=> loss = 1
  auto loss = data->get_loss(cont_label, predicted_action);

  BOOST_CHECK_EQUAL(loss, 1.0f);

  cont_label.costs.delete_v();
}