#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "test_common.h"

#include <vector>
#include "conditional_contextual_bandit.h"
#include "vw.h"

template <typename It, typename T>
bool contains(It begin_it, It end_it, T item)
{
  auto it = std::find(begin_it, end_it, item);
  return it != end_it;
}

// TODO: Make unit test dig out and verify features.
BOOST_AUTO_TEST_CASE(parse_json_simple)
{
  auto vw = VW::initialize("--json --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  std::string json_text = R"(
    {
      "_label": 1,
      "features": {
        "13": 3.9656971e-02,
        "24303": 2.2660980e-01,
        "const": 0.01
      }
    })";

  auto examples = parse_json(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 1);
  BOOST_CHECK_CLOSE(examples[0]->l.simple.label, 1.f, FLOAT_TOL);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

// TODO: Make unit test dig out and verify features.
BOOST_AUTO_TEST_CASE(parse_json_cb)
{
  std::string json_text = R"(
    {
      "s_": "1",
      "s_": "2",
      "_labelIndex": 0,
      "_label_Action": 1,
      "_label_Cost": 1,
      "_label_Probability": 0.5,
      "_multi": [
        {
          "a_": "1",
          "b_": "1",
          "c_": "1"
        },
        {
          "a_": "2",
          "b_": "2",
          "c_": "2"
        },
        {
          "a_": "3",
          "b_": "3",
          "c_": "3"
        }
      ]
    })";

  auto vw = VW::initialize("--cb_adf --json --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  auto examples = parse_json(*vw, json_text);
  BOOST_CHECK_EQUAL(examples.size(), 4);

  BOOST_CHECK_EQUAL(examples[0]->l.cb.costs.size(), 1);
  BOOST_CHECK_CLOSE(examples[0]->l.cb.costs[0].probability, -1.f, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[0]->l.cb.costs[0].cost, FLT_MAX, FLOAT_TOL);

  // Action examples
  BOOST_CHECK_EQUAL(examples[1]->l.cb.costs.size(), 1);
  BOOST_CHECK_EQUAL(examples[2]->l.cb.costs.size(), 0);
  BOOST_CHECK_EQUAL(examples[3]->l.cb.costs.size(), 0);

  BOOST_CHECK_CLOSE(examples[1]->l.cb.costs[0].probability, 0.5, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[1]->l.cb.costs[0].cost, 1.0, FLOAT_TOL);
  BOOST_CHECK_EQUAL(examples[1]->l.cb.costs[0].action, 1);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_cats)
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
  "18-25":1,
  "4":1,
  "C":1,
  "0":1,
  "1":1,
  "2":1,
  "15":1,
  "M":1
}
)";

  auto vw =
      VW::initialize("--json --chain_hash --cats 4 --min_value=185 --max_value=23959 --bandwidth 1 --no_stdin --quiet",
          nullptr, false, nullptr, nullptr);
  auto examples = parse_json(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 1);

  BOOST_CHECK_EQUAL(examples[0]->l.cb_cont.costs.size(), 1);
  BOOST_CHECK_CLOSE(examples[0]->l.cb_cont.costs[0].pdf_value, 6.20426e-05, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[0]->l.cb_cont.costs[0].cost, 0.657567, FLOAT_TOL);
  BOOST_CHECK_CLOSE(examples[0]->l.cb_cont.costs[0].action, 185.121, FLOAT_TOL);

  auto& space_names = examples[0]->feature_space.get_list(' ').front().feats.space_names;
  BOOST_CHECK_EQUAL(features.size(), space_names.size());
  for (size_t i = 0; i < space_names.size(); i++) { BOOST_CHECK_EQUAL(space_names[i].second, features[i]); }

  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_cats_no_label)
{
  std::vector<std::string> features = {"18-25", "4", "C", "0", "1", "2", "15", "M"};
  std::string json_text = R"(
{
  "18-25":1,
  "4":1,
  "C":1,
  "0":1,
  "1":1,
  "2":1,
  "15":1,
  "M":1
}
)";
  auto vw = VW::initialize(
      "--json --chain_hash -t --cats 4 --min_value=185 --max_value=23959 --bandwidth 1 --no_stdin --quiet", nullptr,
      false, nullptr, nullptr);
  auto examples = parse_json(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 1);
  BOOST_CHECK_EQUAL(examples[0]->l.cb_cont.costs.size(), 0);

  auto& space_names = examples[0]->feature_space.get_list(' ').front().feats.space_names;
  BOOST_CHECK_EQUAL(features.size(), space_names.size());
  for (size_t i = 0; i < space_names.size(); i++) { BOOST_CHECK_EQUAL(space_names[i].second, features[i]); }

  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

// TODO: Make unit test dig out and verify features.
BOOST_AUTO_TEST_CASE(parse_json_ccb)
{
  std::string json_text = R"(
    {
      "s_": "1",
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
        },
        {
          "b_": "2",
          "c_": "2",
          "d_": "2"
        },
        {
          "b_": "2",
          "c_": "2",
          "d_": "2"
        }
      ],
      "_slots": [
        {
          "_id": "00eef1eb-2205-4f47",
          "_inc": [1,2],
          "test": 4,
          "_label_cost": 2,
          "_o": [],
          "_a": 1,
          "_p": 0.25
        },
        {
          "other_feature": 3
        },
        {
          "_id": "set_id",
          "other": 6,
          "_label_cost": 4,
          "_o": [],
          "_a": [2,1],
          "_p": [0.75,0.25]
        }
      ]
    })";

  auto vw =
      VW::initialize("--ccb_explore_adf --json --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  auto examples = parse_json(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 8);
  BOOST_CHECK_EQUAL(examples[0]->l.conditional_contextual_bandit.type, CCB::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[4]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[5]->l.conditional_contextual_bandit.type, CCB::example_type::slot);
  BOOST_CHECK_EQUAL(examples[6]->l.conditional_contextual_bandit.type, CCB::example_type::slot);
  BOOST_CHECK_EQUAL(examples[7]->l.conditional_contextual_bandit.type, CCB::example_type::slot);

  auto label1 = examples[5]->l.conditional_contextual_bandit;
  BOOST_CHECK_EQUAL(label1.explicit_included_actions.size(), 2);
  BOOST_CHECK_EQUAL(label1.explicit_included_actions[0], 1);
  BOOST_CHECK_EQUAL(label1.explicit_included_actions[1], 2);
  BOOST_CHECK_CLOSE(label1.outcome->cost, 2.f, .0001f);
  BOOST_CHECK_EQUAL(label1.outcome->probabilities.size(), 1);
  BOOST_CHECK_EQUAL(label1.outcome->probabilities[0].action, 1);
  BOOST_CHECK_CLOSE(label1.outcome->probabilities[0].score, .25f, .0001f);

  auto label2 = examples[6]->l.conditional_contextual_bandit;
  BOOST_CHECK_EQUAL(label2.explicit_included_actions.size(), 0);
  BOOST_CHECK(label2.outcome == nullptr);

  auto label3 = examples[7]->l.conditional_contextual_bandit;
  BOOST_CHECK_EQUAL(label3.explicit_included_actions.size(), 0);
  BOOST_CHECK_CLOSE(label3.outcome->cost, 4.f, .0001f);
  BOOST_CHECK_EQUAL(label3.outcome->probabilities.size(), 2);
  BOOST_CHECK_EQUAL(label3.outcome->probabilities[0].action, 2);
  BOOST_CHECK_CLOSE(label3.outcome->probabilities[0].score, .75f, .0001f);
  BOOST_CHECK_EQUAL(label3.outcome->probabilities[1].action, 1);
  BOOST_CHECK_CLOSE(label3.outcome->probabilities[1].score, .25f, .0001f);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_cb_as_ccb)
{
  std::string json_text = R"(
    {
      "s_": "1",
      "s_": "2",
      "_labelIndex": 0,
      "_label_Action": 1,
      "_label_Cost": 1,
      "_label_Probability": 0.5,
      "_multi": [
        {
          "a_": "1",
          "b_": "1",
          "c_": "1"
        },
        {
          "a_": "2",
          "b_": "2",
          "c_": "2"
        },
        {
          "a_": "3",
          "b_": "3",
          "c_": "3"
        }
      ]
    })";

  auto vw =
      VW::initialize("--ccb_explore_adf --json --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  auto examples = parse_json(*vw, json_text);

  BOOST_CHECK_EQUAL(examples.size(), 5);
  BOOST_CHECK_EQUAL(examples[0]->l.conditional_contextual_bandit.type, CCB::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[4]->l.conditional_contextual_bandit.type, CCB::example_type::slot);

  auto& label1 = examples[4]->l.conditional_contextual_bandit;
  BOOST_CHECK_EQUAL(label1.explicit_included_actions.size(), 0);
  BOOST_CHECK_CLOSE(label1.outcome->cost, 1.f, .0001f);
  BOOST_CHECK_EQUAL(label1.outcome->probabilities.size(), 1);
  BOOST_CHECK_EQUAL(label1.outcome->probabilities[0].action, 0);
  BOOST_CHECK_CLOSE(label1.outcome->probabilities[0].score, .5f, .0001f);
  VW::finish_example(*vw, examples);
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_slates_dom_parser)
{
  std::string json_text = R"(
{
    "GUser": {
        "id": "mk",
        "major": "psychology",
        "hobby": "kids",
        "favorite_character": "7of9"
    },
    "_multi": [
        {
            "_slot_id": 0,
            "TAction": {
                "topic": "SkiConditions-VT"
            }
        },
        {
            "_slot_id": 0,
            "TAction": {
                "topic": "HerbGarden"
            }
        },
        {
            "_slot_id": 1,
            "TAction": {
                "topic": "BeyBlades"
            }
        },
        {
            "_slot_id": 1,
            "TAction": {
                "topic": "NYCLiving"
            }
        },
        {
            "_slot_id": 1,
            "TAction": {
                "topic": "MachineLearning"
            }
        }
    ],
    "_slots": [
        {
            "slot_id": "__0"
        },
        {
            "slot_id": "__2"
        }
    ]
}
)";

  // Assert parsed values against what they should be
  auto slates_vw =
      VW::initialize("--slates --dsjson --chain_hash --no_stdin --quiet", nullptr, false, nullptr, nullptr);
  auto examples = parse_json(*slates_vw, json_text);

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
  BOOST_CHECK_EQUAL(label0.labeled, false);
  BOOST_CHECK_EQUAL(examples[1]->l.slates.slot_id, 0);
  BOOST_CHECK_EQUAL(examples[2]->l.slates.slot_id, 0);
  BOOST_CHECK_EQUAL(examples[3]->l.slates.slot_id, 1);
  BOOST_CHECK_EQUAL(examples[4]->l.slates.slot_id, 1);
  BOOST_CHECK_EQUAL(examples[5]->l.slates.slot_id, 1);

  BOOST_CHECK_EQUAL(examples[0]->feature_space.indices()[0], 'G');
  BOOST_CHECK_EQUAL(examples[0]->feature_space.get('G', VW::hash_space(*slates_vw, "GUser")).indicies.size(), 4);

  VW::finish_example(*slates_vw, examples);
  VW::finish(*slates_vw);
}

// The json parser does insitu parsing, this test ensures that the string does not change. It internally must do a copy.
BOOST_AUTO_TEST_CASE(parse_json_text_does_not_change_input)
{
  std::string json_text =
      R"({"Version":"1","c":{"TShared":{"a=1":1,"b=0":1,"c=1":1},"_multi":[{"TAction":{"value=0.000000":1}},{"TAction":{"value=1.000000":1}},{"TAction":{"value=2.000000":1}},{"TAction":{"value=3.000000":1}},{"TAction":{"value=0.000000":1}},{"TAction":{"value=1.000000":1}},{"TAction":{"value=2.000000":1}},{"TAction":{"value=0.000000":1}},{"TAction":{"value=1.000000":1}}],"_slots":[{"Slate":{"c":1},"_inc":[0,1,2,3]},{"Slate":{"c":1},"_inc":[4,5,6]},{"Slate":{"c":1},"_inc":[7,8]}]},"_outcomes":[{"_id":"ac32c0fc-f895-429d-9063-01c996432f791249622271","_label_cost":0,"_a":[0,1,2,3],"_p":[0.25,0.25,0.25,0.25],"_o":[0]},{"_id":"b64a5e7d-6e76-4d66-98fe-dc214e675ff81249622271","_label_cost":0,"_a":[4,5,6],"_p":[0.333333,0.333333,0.333333],"_o":[0]},{"_id":"a3a29e41-d903-4fbe-b624-11632733cf6f1249622271","_label_cost":0,"_a":[7,8],"_p":[0.5,0.5],"_o":[0]}],"VWState":{"m":"N/A"}})";
  std::string json_text_copy = json_text;

  auto* ccb_vw = VW::initialize("--ccb_explore_adf --dsjson --quiet", nullptr, false, nullptr, nullptr);

  v_array<example*> examples;
  examples.push_back(&VW::get_unused_example(ccb_vw));
  ccb_vw->example_parser->text_reader(ccb_vw, json_text.c_str(), strlen(json_text.c_str()), examples);

  BOOST_CHECK_EQUAL(json_text, json_text_copy);

  multi_ex vec;
  for (const auto& ex : examples) { vec.push_back(ex); }
  VW::finish_example(*ccb_vw, vec);
  VW::finish(*ccb_vw);
}

BOOST_AUTO_TEST_CASE(parse_json_dedup_cb)
{
  const std::string json_deduped_text = R"(
{
  "GUser":{"id":"a","major":"eng","hobby":"hiking"},"_multi":[{"__aid":848539518},{"__aid":3407057455}]
}
  )";

  const std::string action_1 = R"({"TAction":{"a1":"f1"}})";
  const std::string action_2 = R"({"TAction":{"a2":"f2"}})";
  uint64_t dedup_id_1 = 848539518;
  uint64_t dedup_id_2 = 3407057455;

  auto vw = VW::initialize("--json --chain_hash --cb_explore_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  std::unordered_map<uint64_t, example*> dedup_examples;
  v_array<example*> examples;

  // parse first dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_1.c_str(), action_1.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_1, examples[0]);

  examples.clear();

  // parse second dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_2.c_str(), action_2.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_2, examples[0]);

  examples.clear();

  // parse json that includes dedup id's and re-use the examples from the dedup map instead of creating new ones
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)json_deduped_text.c_str(), json_deduped_text.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw, &dedup_examples);

  BOOST_CHECK_EQUAL(examples.size(), 3);                    // shared example + 2 multi examples
  BOOST_CHECK_NE(examples[1], dedup_examples[dedup_id_1]);  // checking pointers
  BOOST_CHECK_NE(examples[2], dedup_examples[dedup_id_2]);  // checking pointers

  // check internals

  // check namespaces
  BOOST_CHECK_EQUAL(examples[1]->indices.size(), 1);
  BOOST_CHECK(contains(examples[1]->feature_space.indices().begin(), examples[1]->feature_space.indices().end(), 'T'));
  BOOST_CHECK_EQUAL(
      examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].first, "TAction");
  BOOST_CHECK_EQUAL(examples[2]->indices.size(), 1);
  BOOST_CHECK(contains(examples[2]->feature_space.indices().begin(), examples[2]->feature_space.indices().end(), 'T'));
  BOOST_CHECK_EQUAL(
      examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].first, "TAction");

  // check features
  BOOST_CHECK_EQUAL(examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names.size(), 1);
  BOOST_CHECK_EQUAL(examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].second, "a1^f1");
  BOOST_CHECK_EQUAL(examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names.size(), 1);
  BOOST_CHECK_EQUAL(examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].second, "a2^f2");

  for (auto* example : examples) { VW::finish_example(*vw, *example); }
  for (auto& dedup : dedup_examples) { VW::finish_example(*vw, *dedup.second); }
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_dedup_cb_missing_dedup_id)
{
  const std::string json_deduped_text = R"(
{
  "GUser":{"id":"a","major":"eng","hobby":"hiking"},"_multi":[{"__aid":848539518},{"__aid":3407057455}]
}
  )";

  const std::string action_1 = R"({"TAction":{"a1":"f1"}})";
  const std::string action_2 = R"({"TAction":{"a2":"f2"}})";
  uint64_t dedup_id_1 = 848539518;
  uint64_t dedup_id_2 = 4407057455;  // dedup id doesn't match the one given in the payload

  auto vw = VW::initialize("--json --chain_hash --cb_explore_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  std::unordered_map<uint64_t, example*> dedup_examples;
  v_array<example*> examples;

  // parse first dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_1.c_str(), action_1.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_1, examples[0]);

  examples.clear();

  // parse second dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_2.c_str(), action_2.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_2, examples[0]);

  examples.clear();

  // parse json that includes dedup id's and re-use the examples from the dedup map instead of creating new ones
  examples.push_back(&VW::get_unused_example(vw));
  BOOST_REQUIRE_THROW(
      VW::read_line_json_s<true>(*vw, examples, (char*)json_deduped_text.c_str(), json_deduped_text.length(),
          (VW::example_factory_t)&VW::get_unused_example, (void*)vw, &dedup_examples),
      VW::vw_exception);

  for (auto* example : examples) { VW::finish_example(*vw, *example); }
  for (auto& dedup : dedup_examples) { VW::finish_example(*vw, *dedup.second); }
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_dedup_ccb)
{
  const std::string json_deduped_text = R"(
{
  "GUser":{"id":"a","major":"eng","hobby":"hiking"},"_multi":[{"__aid":848539518},{"__aid":3407057455}],
  "_slots": [
        {
          "_id": "00eef1eb-2205-4f47",
          "_inc": [1,2],
          "test": 4,
          "_label_cost": 2,
          "_o": [],
          "_a": 1,
          "_p": 0.25
        },
        {
          "other_feature": 3
        },
        {
          "_id": "set_id",
          "other": 6,
          "_label_cost": 4,
          "_o": [],
          "_a": [2,1],
          "_p": [0.75,0.25]
        }
      ]
}
  )";

  const std::string action_1 = R"({"TAction":{"a1":"f1"}})";
  const std::string action_2 = R"({"TAction":{"a2":"f2"}})";
  uint64_t dedup_id_1 = 848539518;
  uint64_t dedup_id_2 = 3407057455;

  auto vw =
      VW::initialize("--json --chain_hash --ccb_explore_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  std::unordered_map<uint64_t, example*> dedup_examples;
  v_array<example*> examples;

  // parse first dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_1.c_str(), action_1.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_1, examples[0]);

  examples.clear();

  // parse second dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_2.c_str(), action_2.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_2, examples[0]);

  examples.clear();

  // parse json that includes dedup id's and re-use the examples from the dedup map instead of creating new ones
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)json_deduped_text.c_str(), json_deduped_text.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw, &dedup_examples);

  BOOST_CHECK_EQUAL(examples.size(), 6);                    // shared example + 2 multi examples + 3 slots
  BOOST_CHECK_NE(examples[1], dedup_examples[dedup_id_1]);  // checking pointers
  BOOST_CHECK_NE(examples[2], dedup_examples[dedup_id_2]);  // checking pointers

  // check internals

  // check namespaces
  BOOST_CHECK_EQUAL(examples[1]->indices.size(), 1);
  BOOST_CHECK(contains(examples[1]->feature_space.indices().begin(), examples[1]->feature_space.indices().end(), 'T'));
  BOOST_CHECK_EQUAL(
      examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].first, "TAction");
  BOOST_CHECK_EQUAL(examples[2]->indices.size(), 1);
  BOOST_CHECK(contains(examples[2]->feature_space.indices().begin(), examples[2]->feature_space.indices().end(), 'T'));
  BOOST_CHECK_EQUAL(
      examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].first, "TAction");

  // check features
  BOOST_CHECK_EQUAL(examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names.size(), 1);
  BOOST_CHECK_EQUAL(examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].second, "a1^f1");
  BOOST_CHECK_EQUAL(examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names.size(), 1);
  BOOST_CHECK_EQUAL(examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].second, "a2^f2");

  // check ccb

  BOOST_CHECK_EQUAL(examples[0]->l.conditional_contextual_bandit.type, CCB::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.conditional_contextual_bandit.type, CCB::example_type::slot);
  BOOST_CHECK_EQUAL(examples[4]->l.conditional_contextual_bandit.type, CCB::example_type::slot);
  BOOST_CHECK_EQUAL(examples[5]->l.conditional_contextual_bandit.type, CCB::example_type::slot);

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
  BOOST_CHECK(label2.outcome == nullptr);

  auto label3 = examples[5]->l.conditional_contextual_bandit;
  BOOST_CHECK_EQUAL(label3.explicit_included_actions.size(), 0);
  BOOST_CHECK_CLOSE(label3.outcome->cost, 4.f, .0001f);
  BOOST_CHECK_EQUAL(label3.outcome->probabilities.size(), 2);
  BOOST_CHECK_EQUAL(label3.outcome->probabilities[0].action, 2);
  BOOST_CHECK_CLOSE(label3.outcome->probabilities[0].score, .75f, .0001f);
  BOOST_CHECK_EQUAL(label3.outcome->probabilities[1].action, 1);
  BOOST_CHECK_CLOSE(label3.outcome->probabilities[1].score, .25f, .0001f);

  for (auto* example : examples) { VW::finish_example(*vw, *example); }
  for (auto& dedup : dedup_examples) { VW::finish_example(*vw, *dedup.second); }
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_dedup_ccb_dedup_id_missing)
{
  const std::string json_deduped_text = R"(
{
  "GUser":{"id":"a","major":"eng","hobby":"hiking"},"_multi":[{"__aid":848539518},{"__aid":3407057455}],
  "_slots": [
        {
          "_id": "00eef1eb-2205-4f47",
          "_inc": [1,2],
          "test": 4,
          "_label_cost": 2,
          "_o": [],
          "_a": 1,
          "_p": 0.25
        },
        {
          "other_feature": 3
        },
        {
          "_id": "set_id",
          "other": 6,
          "_label_cost": 4,
          "_o": [],
          "_a": [2,1],
          "_p": [0.75,0.25]
        }
      ]
}
  )";

  const std::string action_1 = R"({"TAction":{"a1":"f1"}})";
  const std::string action_2 = R"({"TAction":{"a2":"f2"}})";
  uint64_t dedup_id_1 = 848539518;
  uint64_t dedup_id_2 = 4407057455;  // dedup id different then the one in payload

  auto vw =
      VW::initialize("--json --chain_hash --ccb_explore_adf --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  std::unordered_map<uint64_t, example*> dedup_examples;
  v_array<example*> examples;

  // parse first dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_1.c_str(), action_1.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_1, examples[0]);

  examples.clear();

  // parse second dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_2.c_str(), action_2.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_2, examples[0]);

  examples.clear();

  // parse json that includes dedup id's and re-use the examples from the dedup map instead of creating new ones
  examples.push_back(&VW::get_unused_example(vw));
  BOOST_REQUIRE_THROW(
      VW::read_line_json_s<true>(*vw, examples, (char*)json_deduped_text.c_str(), json_deduped_text.length(),
          (VW::example_factory_t)&VW::get_unused_example, (void*)vw, &dedup_examples),
      VW::vw_exception);

  for (auto* example : examples) { VW::finish_example(*vw, *example); }
  for (auto& dedup : dedup_examples) { VW::finish_example(*vw, *dedup.second); }
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_dedup_slates)
{
  const std::string json_deduped_text = R"(
{
  "GUser":{"id":"a","major":"eng","hobby":"hiking"},
  "_multi":[{"__aid":4282062864},{"__aid":4199675127}],
  "_slots":[{"Slot":{"a1":"f1"}},{"Slot":{"a2":"f2"}}]
}
  )";

  const std::string action_1 = R"({"TAction":{"a1":"f1"},"_slot_id":0})";
  const std::string action_2 = R"({"TAction":{"a2":"f2"},"_slot_id":1})";
  uint64_t dedup_id_1 = 4282062864;
  uint64_t dedup_id_2 = 4199675127;

  auto vw = VW::initialize("--json --chain_hash --slates --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  std::unordered_map<uint64_t, example*> dedup_examples;
  v_array<example*> examples;

  // parse first dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_1.c_str(), action_1.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_1, examples[0]);

  examples.clear();

  // parse second dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_2.c_str(), action_2.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_2, examples[0]);

  examples.clear();

  // parse json that includes dedup id's and re-use the examples from the dedup map instead of creating new ones
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)json_deduped_text.c_str(), json_deduped_text.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw, &dedup_examples);

  BOOST_CHECK_EQUAL(examples.size(), 5);                    // shared example + 2 multi examples + 2 slots
  BOOST_CHECK_NE(examples[1], dedup_examples[dedup_id_1]);  // checking pointers
  BOOST_CHECK_NE(examples[2], dedup_examples[dedup_id_2]);  // checking pointers

  // check internals

  // check namespaces
  BOOST_CHECK_EQUAL(examples[1]->indices.size(), 1);
  BOOST_CHECK(contains(examples[1]->feature_space.indices().begin(), examples[1]->feature_space.indices().end(), 'T'));
  BOOST_CHECK_EQUAL(
      examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].first, "TAction");
  BOOST_CHECK_EQUAL(examples[2]->indices.size(), 1);
  BOOST_CHECK(contains(examples[2]->feature_space.indices().begin(), examples[2]->feature_space.indices().end(), 'T'));
  BOOST_CHECK_EQUAL(
      examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].first, "TAction");

  // check features
  BOOST_CHECK_EQUAL(examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names.size(), 1);
  BOOST_CHECK_EQUAL(examples[1]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].second, "a1^f1");
  BOOST_CHECK_EQUAL(examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names.size(), 1);
  BOOST_CHECK_EQUAL(examples[2]->feature_space.get('T', VW::hash_space(*vw, "TAction")).space_names[0].second, "a2^f2");

  // check slates
  BOOST_CHECK_EQUAL(examples[0]->l.slates.type, VW::slates::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.slates.type, VW::slates::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.slates.type, VW::slates::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.slates.type, VW::slates::example_type::slot);
  BOOST_CHECK_EQUAL(examples[4]->l.slates.type, VW::slates::example_type::slot);

  BOOST_CHECK_EQUAL(examples[1]->l.slates.slot_id, 0);
  BOOST_CHECK_EQUAL(examples[2]->l.slates.slot_id, 1);

  for (auto* example : examples) { VW::finish_example(*vw, *example); }
  for (auto& dedup : dedup_examples) { VW::finish_example(*vw, *dedup.second); }
  VW::finish(*vw);
}

BOOST_AUTO_TEST_CASE(parse_json_dedup_slates_dedup_id_missing)
{
  const std::string json_deduped_text = R"(
{
  "GUser":{"id":"a","major":"eng","hobby":"hiking"},
  "_multi":[{"__aid":4282062864},{"__aid":4199675127}],
  "_slots":[{"Slot":{"a1":"f1"}},{"Slot":{"a2":"f2"}}]
}
  )";

  const std::string action_1 = R"({"TAction":{"a1":"f1"},"_slot_id":0})";
  const std::string action_2 = R"({"TAction":{"a2":"f2"},"_slot_id":1})";
  uint64_t dedup_id_1 = 4282062864;
  uint64_t dedup_id_2 = 5199675127;  // dedup id different then the one in the payload

  auto vw = VW::initialize("--json --chain_hash --slates --no_stdin --quiet", nullptr, false, nullptr, nullptr);

  std::unordered_map<uint64_t, example*> dedup_examples;
  v_array<example*> examples;

  // parse first dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_1.c_str(), action_1.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_1, examples[0]);

  examples.clear();

  // parse second dedup example and store it in dedup_examples map
  examples.push_back(&VW::get_unused_example(vw));
  VW::read_line_json_s<true>(*vw, examples, (char*)action_2.c_str(), action_2.length(),
      (VW::example_factory_t)&VW::get_unused_example, (void*)vw);
  dedup_examples.emplace(dedup_id_2, examples[0]);

  examples.clear();

  // parse json that includes dedup id's and re-use the examples from the dedup map instead of creating new ones
  examples.push_back(&VW::get_unused_example(vw));

  BOOST_REQUIRE_THROW(
      VW::read_line_json_s<true>(*vw, examples, (char*)json_deduped_text.c_str(), json_deduped_text.length(),
          (VW::example_factory_t)&VW::get_unused_example, (void*)vw, &dedup_examples),
      VW::vw_exception);

  for (auto* example : examples) { VW::finish_example(*vw, *example); }
  for (auto& dedup : dedup_examples) { VW::finish_example(*vw, *dedup.second); }
  VW::finish(*vw);
}
