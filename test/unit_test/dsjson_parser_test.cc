#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "test_common.h"

#include <vector>
#include "conditional_contextual_bandit.h"

BOOST_AUTO_TEST_CASE(parse_dsjson_underscore_p)
{
  const std::string json_text = R"(
{
  "_p": [0.4, 0.6]
}
  )";
  auto vw = VW::initialize("--dsjson --chain_hash --cb_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  DecisionServiceInteraction interaction;

  auto examples = parse_dsjson(*vw, json_text, &interaction);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);

  constexpr float EXPECTED_PDF[2] = {0.4f, 0.6f};
  const size_t num_probabilities = interaction.probabilities.size();
  BOOST_CHECK_EQUAL(num_probabilities, 2);
  for (size_t i = 0; i < num_probabilities; ++i)
  {
    // Check that probabilities are as expected.
    BOOST_CHECK_EQUAL(interaction.probabilities[i], EXPECTED_PDF[i]);
  }
}

BOOST_AUTO_TEST_CASE(parse_dsjson_p)
{
  const std::string json_text = R"(
{
  "p": [0.4, 0.6]
}
  )";
  auto vw = VW::initialize("--dsjson --chain_hash --cb_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  DecisionServiceInteraction interaction;

  auto examples = parse_dsjson(*vw, json_text, &interaction);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);

  constexpr float EXPECTED_PDF[2] = {0.4f, 0.6f};
  const size_t num_probabilities = interaction.probabilities.size();
  BOOST_CHECK_EQUAL(num_probabilities, 2);
  for (size_t i = 0; i < num_probabilities; ++i)
  {
    // Check that probabilities are as expected.
    BOOST_CHECK_EQUAL(interaction.probabilities[i], EXPECTED_PDF[i]);
  }
}

BOOST_AUTO_TEST_CASE(parse_dsjson_p_duplicates)
{
  const std::string json_text = R"(
{
  "c": {
    "_p": [0.4, 0.6]
  },
  "p": [0.4, 0.3, 0.3],
  "_p": [0.5, 0.5]
}
  )";
  auto vw = VW::initialize("--dsjson --chain_hash --cb_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  DecisionServiceInteraction interaction;

  auto examples = parse_dsjson(*vw, json_text, &interaction);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);

  // Use the latest "p" or "_p" field provided. The "_p" is ignored when it's inside "c".
  constexpr float EXPECTED_PDF[2] = {0.5f, 0.5f};
  const size_t num_probabilities = interaction.probabilities.size();
  BOOST_CHECK_EQUAL(num_probabilities, 2);
  for (size_t i = 0; i < num_probabilities; ++i)
  {
    // Check that probabilities are as expected.
    BOOST_CHECK_EQUAL(interaction.probabilities[i], EXPECTED_PDF[i]);
  }
}

BOOST_AUTO_TEST_CASE(parse_dsjson_pdrop_float)
{
  const std::string json_text = R"(
{
  "pdrop": 0.1
}
  )";
  auto vw = VW::initialize("--dsjson --chain_hash --cb_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  DecisionServiceInteraction interaction;

  auto examples = parse_dsjson(*vw, json_text, &interaction);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);

  BOOST_CHECK_CLOSE(0.1f, interaction.probabilityOfDrop, FLOAT_TOL);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_pdrop_uint)
{
  const std::string json_text = R"(
{
  "pdrop": 0
}
  )";
  auto vw = VW::initialize("--dsjson --chain_hash --cb_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  DecisionServiceInteraction interaction;

  auto examples = parse_dsjson(*vw, json_text, &interaction);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);

  BOOST_CHECK_CLOSE(0.0f, interaction.probabilityOfDrop, FLOAT_TOL);
}

// TODO: Make unit test dig out and verify features.
BOOST_AUTO_TEST_CASE(parse_dsjson_cb)
{
  std::string json_text = R"(
{
  "_label_cost": -1,
  "_label_probability": 0.8166667,
  "_label_Action": 2,
  "_labelIndex": 1,
  "Version": "1",
  "EventId": "0074434d3a3a46529f65de8a59631939",
  "a": [
    2,
    1,
    3
  ],
  "c": {
    "shared_ns": {
      "shared_feature": 0
    },
    "_multi": [
      {
        "_tag": "tag",
        "ns1": {
          "f1": 1,
          "f2": "strng"
        },
        "ns2": [
          {
            "f3": "value1"
          },
          {
            "ns3": {
              "f4": 0.994963765
            }
          }
        ]
      },
      {
        "_tag": "tag",
        "ns1": {
          "f1": 1,
          "f2": "strng"
        }
      },
      {
        "_tag": "tag",
        "ns1": {
          "f1": 1,
          "f2": "strng"
        }
      }
    ]
  },
  "p": [
    0.816666663,
    0.183333333,
    0.183333333
  ],
  "VWState": {
    "m": "096200c6c41e42bbb879c12830247637/0639c12bea464192828b250ffc389657"
  }
}
)";
  auto vw = VW::initialize("--dsjson --chain_hash --cb_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 4);

  // Shared example
  BOOST_CHECK_EQUAL(examples[0]->l.cb.costs.size(), 1);
  BOOST_CHECK_CLOSE(examples[0]->l.cb.costs[0].probability, -1.f, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[0]->l.cb.costs[0].cost, FLT_MAX, FLOAT_TOL);

  // Action examples
  BOOST_CHECK_EQUAL(examples[1]->l.cb.costs.size(), 0);
  BOOST_CHECK_EQUAL(examples[2]->l.cb.costs.size(), 1);
  BOOST_CHECK_EQUAL(examples[3]->l.cb.costs.size(), 0);

  BOOST_CHECK_CLOSE(examples[2]->l.cb.costs[0].probability, 0.8166667, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[2]->l.cb.costs[0].cost, -1.0, FLOAT_TOL);
  BOOST_CHECK_EQUAL(examples[2]->l.cb.costs[0].action, 2);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_cats)
{
  std::vector<std::string> features = {"18-25", "4", "C", "0", "1", "2", "15", "M"};
  std::string json_text = R"(
{
  "_label_ca":
  {
    "cost": 0.657567,
    "pdf_value": 6.20426e-05,
    "action": 185.121
  },
  "Version": "1",
  "EventId": "event_id",
  "c": {
    "18-25":1,
    "4":1,
    "C":1,
    "0":1,
    "1":1,
    "2":1,
    "15":1,
    "M":1
  },
  "VWState": {
    "m": "N/A"
  }
}
)";
  auto vw = VW::initialize(
      "--dsjson --chain_hash --cats 4 --min_value=185 --max_value=23959 --bandwidth 1 --no_stdin --quiet", nullptr,
      false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 1);

  BOOST_CHECK_EQUAL(examples[0]->l.cb_cont.costs.size(), 1);
  BOOST_CHECK_CLOSE(examples[0]->l.cb_cont.costs[0].pdf_value, 6.20426e-05, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[0]->l.cb_cont.costs[0].cost, 0.657567, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[0]->l.cb_cont.costs[0].action, 185.121, FLOAT_TOL);

  auto& space_names = (*examples[0]->feature_space.begin()).begin()->feats.space_names;
  BOOST_CHECK_EQUAL(features.size(), space_names.size());
  for (size_t i = 0; i < space_names.size(); i++) { BOOST_CHECK_EQUAL(space_names[i].second, features[i]); }

  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_cats_no_label)
{
  std::vector<std::string> features = {"18-25", "4", "C", "0", "1", "2", "15", "M"};
  std::string json_text = R"(
{
  "Version": "1",
  "EventId": "event_id",
  "c": {
    "18-25":1,
    "4":1,
    "C":1,
    "0":1,
    "1":1,
    "2":1,
    "15":1,
    "M":1
  },
  "VWState": {
    "m": "N/A"
  }
}
)";
  auto vw = VW::initialize(
      "--dsjson --chain_hash -t --cats 4 --min_value=185 --max_value=23959 --bandwidth 1 --no_stdin --quiet", nullptr,
      false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 1);
  BOOST_CHECK_EQUAL(examples[0]->l.cb_cont.costs.size(), 0);

  auto& space_names = (*examples[0]->feature_space.begin()).begin()->feats.space_names;
  BOOST_CHECK_EQUAL(features.size(), space_names.size());
  for (size_t i = 0; i < space_names.size(); i++) { BOOST_CHECK_EQUAL(space_names[i].second, features[i]); }

  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_cats_w_valid_pdf)
{
  std::vector<std::string> features = {"18-25", "4", "C", "0", "1", "2", "15", "M"};
  std::string json_text = R"(
{
  "Version": "1",
  "EventId": "event_id",
  "pdf": [{"left": 185, "right": 8109.67, "pdf_value": 2.10314e-06},
    {"left": 8109.67, "right": 23959, "pdf_value": 6.20426e-05}],
  "c": {
    "18-25":1,
    "4":1,
    "C":1,
    "0":1,
    "1":1,
    "2":1,
    "15":1,
    "M":1
  },
  "VWState": {
    "m": "N/A"
  }
}
)";
  auto vw = VW::initialize(
      "--dsjson --chain_hash --cats 4 --min_value=185 --max_value=23959 --bandwidth 1 --no_stdin --quiet", nullptr,
      false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 1);
  const auto& reduction_features =
      examples[0]->_reduction_features.template get<VW::continuous_actions::reduction_features>();

  BOOST_CHECK_EQUAL(reduction_features.is_pdf_set(), true);
  BOOST_CHECK_EQUAL(reduction_features.is_chosen_action_set(), false);

  BOOST_CHECK_EQUAL(reduction_features.pdf.size(), 2);
  BOOST_CHECK_CLOSE(reduction_features.pdf[0].left, 185., FLOAT_TOL);
  BOOST_CHECK_CLOSE(reduction_features.pdf[0].right, 8109.67, FLOAT_TOL);
  BOOST_CHECK_CLOSE(reduction_features.pdf[0].pdf_value, 2.10314e-06, FLOAT_TOL);

  BOOST_CHECK_CLOSE(reduction_features.pdf[1].left, 8109.67, FLOAT_TOL);
  BOOST_CHECK_CLOSE(reduction_features.pdf[1].right, 23959., FLOAT_TOL);
  BOOST_CHECK_CLOSE(reduction_features.pdf[1].pdf_value, 6.20426e-05, FLOAT_TOL);

  auto& space_names = (*examples[0]->feature_space.begin()).begin()->feats.space_names;
  BOOST_CHECK_EQUAL(features.size(), space_names.size());
  for (size_t i = 0; i < space_names.size(); i++) { BOOST_CHECK_EQUAL(space_names[i].second, features[i]); }

  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_cats_w_invalid_pdf)
{
  std::vector<std::string> features = {"18-25", "4", "C", "0", "1", "2", "15", "M"};
  std::string json_text = R"(
{
  "Version": "1",
  "EventId": "event_id",
  "pdf": [
    {"left": 185.121}, {"left": 50, "right":50, "pdf_value": 50}
  ],
  "c": {
    "18-25":1,
    "4":1,
    "C":1,
    "0":1,
    "1":1,
    "2":1,
    "15":1,
    "M":1
  },
  "VWState": {
    "m": "N/A"
  }
}
)";
  auto vw = VW::initialize(
      "--dsjson --chain_hash --cats 4 --min_value=185 --max_value=23959 --bandwidth 1 --no_stdin --quiet", nullptr,
      false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 1);

  const auto& reduction_features =
      examples[0]->_reduction_features.template get<VW::continuous_actions::reduction_features>();

  BOOST_CHECK_EQUAL(reduction_features.is_pdf_set(), false);
  BOOST_CHECK_EQUAL(reduction_features.is_chosen_action_set(), false);

  auto& space_names = (*examples[0]->feature_space.begin()).begin()->feats.space_names;
  BOOST_CHECK_EQUAL(features.size(), space_names.size());
  for (size_t i = 0; i < space_names.size(); i++) { BOOST_CHECK_EQUAL(space_names[i].second, features[i]); }

  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_cats_chosen_action)
{
  std::vector<std::string> features = {"18-25", "4", "C", "0", "1", "2", "15", "M"};
  std::string json_text = R"(
{
  "Version": "1",
  "EventId": "event_id",
  "pdf": [
    {"chosen_action": 185}
  ],
  "c": {
    "18-25":1,
    "4":1,
    "C":1,
    "0":1,
    "1":1,
    "2":1,
    "15":1,
    "M":1
  },
  "VWState": {
    "m": "N/A"
  }
}
)";
  auto vw = VW::initialize(
      "--dsjson --chain_hash --cats 4 --min_value=185 --max_value=23959 --bandwidth 1 --no_stdin --quiet", nullptr,
      false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  const auto& reduction_features =
      examples[0]->_reduction_features.template get<VW::continuous_actions::reduction_features>();

  BOOST_CHECK_EQUAL(examples.size(), 1);
  BOOST_CHECK_EQUAL(reduction_features.is_pdf_set(), false);
  BOOST_CHECK_EQUAL(reduction_features.is_chosen_action_set(), true);
  BOOST_CHECK_CLOSE(reduction_features.chosen_action, 185., FLOAT_TOL);

  auto& space_names = (*examples[0]->feature_space.begin()).begin()->feats.space_names;
  BOOST_CHECK_EQUAL(features.size(), space_names.size());
  for (size_t i = 0; i < space_names.size(); i++) { BOOST_CHECK_EQUAL(space_names[i].second, features[i]); }

  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

// TODO: Make unit test dig out and verify features.
BOOST_AUTO_TEST_CASE(parse_dsjson_ccb)
{
  std::string json_text = R"(
{
  "Timestamp":"timestamp_utc",
  "Version": "1",
  "EventId": "test_id",
  "c":{
      "_multi": [
        {
          "b_": "1",
          "c_": "1",
          "d_": "1"
        },
        {
          "b_": "2",
          "c_": "2",
          "d_": "2"
        }
      ],
      "_slots":[
          {
              "_id": "00eef1eb-2205-4f47",
              "_inc": [1,2],
              "test": 4
          },
          {
              "_id": "set_id",
              "other": 6
          }
      ]
  },
  "_outcomes":[{
      "_label_cost": 2,
      "_o": [],
      "_a": 1,
      "_p": 0.25
    },
    {
      "_label_cost": 4,
      "_o":[],
      "_a": [2, 1],
      "_p": [0.75, 0.25]
    }
  ],
  "VWState": {
    "m": "096200c6c41e42bbb879c12830247637/0639c12bea464192828b250ffc389657"
  }
}
)";

  auto vw =
      VW::initialize("--ccb_explore_adf --dsjson --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 5);
  BOOST_CHECK_EQUAL(examples[0]->l.conditional_contextual_bandit.type, CCB::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.conditional_contextual_bandit.type, CCB::example_type::slot);
  BOOST_CHECK_EQUAL(examples[4]->l.conditional_contextual_bandit.type, CCB::example_type::slot);

  auto label1 = examples[3]->l.conditional_contextual_bandit;
  BOOST_CHECK_EQUAL(label1.explicit_included_actions.size(), 2);
  BOOST_CHECK_EQUAL(label1.explicit_included_actions[0], 1);
  BOOST_CHECK_EQUAL(label1.explicit_included_actions[1], 2);
  BOOST_CHECK_CLOSE(label1.outcome->cost, 2.f, .0001f);
  BOOST_CHECK_EQUAL(label1.outcome->probabilities.size(), 1);
  BOOST_CHECK_EQUAL(label1.outcome->probabilities[0].action, 1);
  BOOST_CHECK_CLOSE(label1.outcome->probabilities[0].score, .25f, .0001f);

  auto label2 = examples[4]->l.conditional_contextual_bandit;
  BOOST_CHECK_EQUAL(label2.explicit_included_actions.size(), 0);
  BOOST_CHECK_CLOSE(label2.outcome->cost, 4.f, .0001f);
  BOOST_CHECK_EQUAL(label2.outcome->probabilities.size(), 2);
  BOOST_CHECK_EQUAL(label2.outcome->probabilities[0].action, 2);
  BOOST_CHECK_CLOSE(label2.outcome->probabilities[0].score, .75f, .0001f);
  BOOST_CHECK_EQUAL(label2.outcome->probabilities[1].action, 1);
  BOOST_CHECK_CLOSE(label2.outcome->probabilities[1].score, .25f, .0001f);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_cb_as_ccb)
{
  std::string json_text = R"(
{
  "_label_cost": -1,
  "_label_probability": 0.8166667,
  "_label_Action": 2,
  "_labelIndex": 1,
  "Version": "1",
  "EventId": "0074434d3a3a46529f65de8a59631939",
  "a": [
    2,
    1,
    3
  ],
  "c": {
    "shared_ns": {
      "shared_feature": 0
    },
    "_multi": [
      {
        "_tag": "tag",
        "ns1": {
          "f1": 1,
          "f2": "strng"
        },
        "ns2": [
          {
            "f3": "value1"
          },
          {
            "ns3": {
              "f4": 0.994963765
            }
          }
        ]
      },
      {
        "_tag": "tag",
        "ns1": {
          "f1": 1,
          "f2": "strng"
        }
      },
      {
        "_tag": "tag",
        "ns1": {
          "f1": 1,
          "f2": "strng"
        }
      }
    ]
  },
  "p": [
    0.816666663,
    0.183333333,
    0.183333333
  ],
  "VWState": {
    "m": "096200c6c41e42bbb879c12830247637/0639c12bea464192828b250ffc389657"
  }
}
)";
  auto vw =
      VW::initialize("--ccb_explore_adf --dsjson --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 5);
  BOOST_CHECK_EQUAL(examples[0]->l.conditional_contextual_bandit.type, CCB::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[4]->l.conditional_contextual_bandit.type, CCB::example_type::slot);

  auto label2 = examples[4]->l.conditional_contextual_bandit;
  BOOST_CHECK_EQUAL(label2.explicit_included_actions.size(), 0);
  BOOST_CHECK_CLOSE(label2.outcome->cost, -1.f, .0001f);
  BOOST_CHECK_EQUAL(label2.outcome->probabilities.size(), 1);
  BOOST_CHECK_EQUAL(label2.outcome->probabilities[0].action, 1);
  BOOST_CHECK_CLOSE(label2.outcome->probabilities[0].score, 0.8166667f, .0001f);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_cb_with_nan)
{
  std::string json_text = R"(
{
    "_label_cost": "NaN",
    "_label_probability": "NaN",
    "_label_Action": 2,
    "_labelIndex": 1,
    "o": [
        {
            "v": "NaN",
            "EventId": "123",
            "ActionTaken": false
        }
    ],
    "Timestamp": "2020-01-15T16:23:36.8640000Z",
    "Version": "1",
    "EventId": "abc",
    "a": [
        2,
        1,
        0
    ],
    "c": {
        "shared_feature":1.0,
        "_multi": [
            {
                "id": "a"
            },
            {
                "id": "b"
            },
            {
                "id": "c"
            }
        ]
    },
    "p": [
        "NaN",
        "NaN",
        "NaN"
    ]
}
)";

  auto vw = VW::initialize("--dsjson --chain_hash --cb_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  auto examples = parse_dsjson(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 4);

  // Shared example
  BOOST_CHECK_EQUAL(examples[0]->l.cb.costs.size(), 1);
  BOOST_CHECK_CLOSE(examples[0]->l.cb.costs[0].probability, -1.f, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[0]->l.cb.costs[0].cost, FLT_MAX, FLOAT_TOL);

  // Action examples
  BOOST_CHECK_EQUAL(examples[1]->l.cb.costs.size(), 0);
  BOOST_CHECK_EQUAL(examples[2]->l.cb.costs.size(), 1);
  BOOST_CHECK_EQUAL(examples[3]->l.cb.costs.size(), 0);

  BOOST_CHECK_EQUAL(std::isnan(examples[2]->l.cb.costs[0].probability), true);
  BOOST_CHECK_EQUAL(std::isnan(examples[2]->l.cb.costs[0].cost), true);
  BOOST_CHECK_EQUAL(examples[2]->l.cb.costs[0].action, 2);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_slates)
{
  std::string json_text = R"(
{
    "_label_cost": 1,
    "_outcomes": [
        {
            "_a": 1,
            "_p": 0.8
        },
        {
            "_a": [0, 1],
            "_p": [0.6, 0.4]
        }
    ],
    "EventId":"test_id",
    "pdrop":0.1,
    "_skipLearn":true,
    "c": {
        "shared_feature": 1.0,
        "_multi": [
            {
                "_slot_id": 0,
                "feature": 1.0,
                "namespace": {
                    "one": 1.0,
                    "test": "string",
                    "array": [
                        1,
                        2,
                        3
                    ],
                    "another": {
                        "test":1.1,
                        "inner_namespac": [
                            {
                                "feature ": "inner "
                            }
                        ]
                    }
                }
            },
            {
                "_slot_id": 0,
                "feature": 1.0
            },
            {
                "_slot_id": 0,
                "feature": 1.0
            },
            {
                "_slot_id": 1,
                "feature": 1.0
            },
            {
                "_slot_id": 1,
                "feature": 1.0
            }
        ],
        "_slots": [
            {
                "feature": 1.0
            },
            {
                "feature": 1.0
            }
        ]
    }
})";

  auto vw = VW::initialize("--slates --dsjson --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  DecisionServiceInteraction ds_interaction;
  auto examples = parse_dsjson(*vw, json_text, &ds_interaction);

  BOOST_CHECK_EQUAL(examples.size(), 8);
  BOOST_CHECK_EQUAL(examples[0]->l.slates.type, VW::slates::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.slates.type, VW::slates::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.slates.type, VW::slates::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.slates.type, VW::slates::example_type::action);
  BOOST_CHECK_EQUAL(examples[4]->l.slates.type, VW::slates::example_type::action);
  BOOST_CHECK_EQUAL(examples[5]->l.slates.type, VW::slates::example_type::action);
  BOOST_CHECK_EQUAL(examples[6]->l.slates.type, VW::slates::example_type::slot);
  BOOST_CHECK_EQUAL(examples[7]->l.slates.type, VW::slates::example_type::slot);

  const auto& label0 = examples[0]->l.slates;
  BOOST_CHECK_CLOSE(label0.cost, 1.f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(label0.labeled, true);

  BOOST_CHECK_EQUAL(examples[1]->l.slates.slot_id, 0);
  BOOST_CHECK_EQUAL(examples[2]->l.slates.slot_id, 0);
  BOOST_CHECK_EQUAL(examples[3]->l.slates.slot_id, 0);
  BOOST_CHECK_EQUAL(examples[4]->l.slates.slot_id, 1);
  BOOST_CHECK_EQUAL(examples[5]->l.slates.slot_id, 1);

  const auto& label6 = examples[6]->l.slates;
  check_collections_with_float_tolerance(
      label6.probabilities, std::vector<ACTION_SCORE::action_score>{{1, 0.8f}}, FLOAT_TOL);
  const auto& label7 = examples[7]->l.slates;
  check_collections_with_float_tolerance(
      label7.probabilities, std::vector<ACTION_SCORE::action_score>{{0, 0.6f}, {1, 0.4f}}, FLOAT_TOL);

  // Check values in DecisionServiceInteraction
  BOOST_CHECK_EQUAL(ds_interaction.eventId, "test_id");
  BOOST_CHECK_CLOSE(ds_interaction.probabilityOfDrop, 0.1, FLOAT_TOL);
  BOOST_CHECK_EQUAL(ds_interaction.skipLearn, true);
  check_collections_exact(ds_interaction.actions, std::vector<unsigned int>{1,0});
  check_collections_with_float_tolerance(ds_interaction.probabilities, std::vector<float>{0.8f, 0.6f}, FLOAT_TOL);

  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_dsjson_slates_dom_parser)
{
  std::string json_text = R"(
{
    "c": {
        "aFloatFeature": 1.0,
        "aStringFeature": "value",
        "dArray": [
            1,
            2.0,
            {
                "aIntFeature": 5,
                "aNamespace": {
                    "bIntFeature": 1
                }
            }
        ],
        "bNamespace": {
            "cIntFeature": 1,
            "cNamespace": {
                "aBoolFeature": true
            }
        },
        "eNamespace": {
            "bBoolFeature": false
        },
        "_multi": [],
        "_slots": []
    }
}
)";

  // Assert parsed values against what they should be
  auto slates_vw =
      VW::initialize("--slates --dsjson --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  auto slates_examples = parse_dsjson(*slates_vw, json_text);

  BOOST_CHECK_EQUAL(slates_examples.size(), 1);
  auto& slates_ex = *slates_examples[0];

  std::set<namespace_index> found{slates_ex.feature_space.indices().begin(), slates_ex.feature_space.indices().end()};
  std::set<namespace_index> expected{'a', 'd', 'c', 'b', 32};
  check_collections_exact(found, expected);

  BOOST_CHECK_EQUAL(slates_ex.feature_space.get_list(' ').front().feats.indicies.size(), 2);
  BOOST_CHECK_EQUAL(slates_ex.feature_space.get('a', VW::hash_space(*slates_vw, "aNamespace")).indicies.size(), 1);
  BOOST_CHECK_EQUAL(slates_ex.feature_space.get('b', VW::hash_space(*slates_vw, "bNamespace")).indicies.size(), 1);
  BOOST_CHECK_EQUAL(slates_ex.feature_space.get('c', VW::hash_space(*slates_vw, "cNamespace")).indicies.size(), 1);
  BOOST_CHECK_EQUAL(slates_ex.feature_space.get('d', VW::hash_space(*slates_vw, "dArray")).indicies.size(), 3);

  // Compare the DOM parser to parsing the same features with the CCB SAX parser
  auto ccb_vw =
      VW::initialize("--ccb_explore_adf --dsjson --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  auto ccb_examples = parse_dsjson(*ccb_vw, json_text);
  BOOST_CHECK_EQUAL(ccb_examples.size(), 1);
  const auto& ccb_ex = *ccb_examples[0];
  check_collections_exact(slates_ex.feature_space.get_list(' ').front().feats.indicies,
      ccb_ex.feature_space.get_list(' ').front().feats.indicies);
  check_collections_exact(slates_ex.feature_space.get('a', VW::hash_space(*ccb_vw, "aNamespace")).indicies,
      ccb_ex.feature_space.get('a', VW::hash_space(*slates_vw, "aNamespace")).indicies);
  check_collections_exact(slates_ex.feature_space.get('b', VW::hash_space(*ccb_vw, "bNamespace")).indicies,
      ccb_ex.feature_space.get('b', VW::hash_space(*slates_vw, "bNamespace")).indicies);
  check_collections_exact(slates_ex.feature_space.get('c', VW::hash_space(*ccb_vw, "cNamespace")).indicies,
      ccb_ex.feature_space.get('c', VW::hash_space(*slates_vw, "cNamespace")).indicies);
  check_collections_exact(slates_ex.feature_space.get('d', VW::hash_space(*ccb_vw, "dArray")).indicies,
      ccb_ex.feature_space.get('d', VW::hash_space(*slates_vw, "dArray")).indicies);

  check_collections_with_float_tolerance(slates_ex.feature_space.get_list(' ').front().feats.values,
      ccb_ex.feature_space.get_list(' ').front().feats.values, FLOAT_TOL);
  check_collections_with_float_tolerance(
      slates_ex.feature_space.get('a', VW::hash_space(*slates_vw, "aNamespace")).values,
      ccb_ex.feature_space.get('a', VW::hash_space(*ccb_vw, "aNamespace")).values, FLOAT_TOL);
  check_collections_with_float_tolerance(
      slates_ex.feature_space.get('b', VW::hash_space(*slates_vw, "bNamespace")).values,
      ccb_ex.feature_space.get('b', VW::hash_space(*ccb_vw, "bNamespace")).values, FLOAT_TOL);
  check_collections_with_float_tolerance(
      slates_ex.feature_space.get('c', VW::hash_space(*slates_vw, "cNamespace")).values,
      ccb_ex.feature_space.get('c', VW::hash_space(*ccb_vw, "cNamespace")).values, FLOAT_TOL);
  check_collections_with_float_tolerance(slates_ex.feature_space.get('d', VW::hash_space(*slates_vw, "dArray")).values,
      ccb_ex.feature_space.get('d', VW::hash_space(*ccb_vw, "dArray")).values, FLOAT_TOL);

  VW::finish_example(*slates_vw, slates_examples);
  VW::finish(*slates_vw);
  VW::finish_example(*ccb_vw, ccb_examples);
  VW::finish(*ccb_vw);
}
