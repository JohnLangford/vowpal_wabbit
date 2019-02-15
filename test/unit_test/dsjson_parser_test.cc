#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "test_common.h"

#include <vector>
#include "conditional_contextual_bandit.h"
#include "parse_example_json.h"

v_array<example*> parse_dsjson(vw& all, std::string line)
{
  v_array<example*> examples = v_init<example*>();
  examples.push_back(&VW::get_unused_example(&all));
  DecisionServiceInteraction interaction;

  VW::read_line_decision_service_json<true>(all, examples, (char*)line.c_str(), line.size(), false,
      (VW::example_factory_t)&VW::get_unused_example, (void*)&all, &interaction);

  return examples;
}

BOOST_AUTO_TEST_CASE(parse_dsjson_cb)
{
  auto vw = VW::initialize("--dsjson --cb_adf --no_stdin", nullptr, false, nullptr, nullptr);
  // Remove once ccb is available.
  auto jsonp = static_cast<json_parser<false>*>(vw->p->jsonp);
  jsonp->mode = json_parser_mode::cb;
  std::string text = R"({
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

  auto examples = parse_dsjson(*vw, text);
  BOOST_CHECK_EQUAL(examples.size(), 5);
  BOOST_CHECK_EQUAL(examples[0]->l.conditional_contextual_bandit.type, CCB::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.conditional_contextual_bandit.type, CCB::example_type::decision);
  BOOST_CHECK_EQUAL(examples[4]->l.conditional_contextual_bandit.type, CCB::example_type::decision);

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
  BOOST_CHECK_EQUAL(label2.outcome->probabilities[0].action, 1);
  BOOST_CHECK_CLOSE(label2.outcome->probabilities[0].score, .25f, .0001f);

  // TODO: Make unit test dig out and verify features.
}


BOOST_AUTO_TEST_CASE(parse_dsjson_ccb)
{
  auto vw = VW::initialize("--dsjson --no_stdin", nullptr, false, nullptr, nullptr);
  // Remove once ccb is available.
  auto jsonp = static_cast<json_parser<false>*>(vw->p->jsonp);
  jsonp->mode = json_parser_mode::ccb;
  std::string text = R"({
    "Timestamp":"timestamp_utc",
    "Version": "1",
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
        "_df":[
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
    "_decisions":[{
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

  auto examples = parse_dsjson(*vw, text);
  BOOST_CHECK_EQUAL(examples.size(), 5);
  BOOST_CHECK_EQUAL(examples[0]->l.conditional_contextual_bandit.type, CCB::example_type::shared);
  BOOST_CHECK_EQUAL(examples[1]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[2]->l.conditional_contextual_bandit.type, CCB::example_type::action);
  BOOST_CHECK_EQUAL(examples[3]->l.conditional_contextual_bandit.type, CCB::example_type::decision);
  BOOST_CHECK_EQUAL(examples[4]->l.conditional_contextual_bandit.type, CCB::example_type::decision);

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
  BOOST_CHECK_EQUAL(label2.outcome->probabilities[0].action, 1);
  BOOST_CHECK_CLOSE(label2.outcome->probabilities[0].score, .25f, .0001f);

  // TODO: Make unit test dig out and verify features.
}
