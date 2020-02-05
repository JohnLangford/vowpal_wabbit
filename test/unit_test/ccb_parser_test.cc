#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "test_common.h"

#include <vector>
#include "conditional_contextual_bandit.h"
#include "parser.h"
#include "example.h"

void parse_label(label_parser& lp, parser* p, VW::string_view label, polylabel& l)
{
  tokenize(' ', label, p->words);
  lp.default_label(l);
  lp.parse_label(p, nullptr, l, p->words);
}

BOOST_AUTO_TEST_CASE(ccb_parse_label)
{
  auto lp = CCB::ccb_label_parser;
  parser p{8 /*ring_size*/, false /*strict parse*/};

  {
    auto label = scoped_calloc_or_throw<polylabel>();
    parse_label(lp, &p, "ccb shared", *label);
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions.size(), 0);
    BOOST_CHECK(label->ccb().outcome == nullptr);
    BOOST_CHECK_EQUAL(label->ccb().type, CCB::example_type::shared);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    parse_label(lp, &p, "ccb action", *label.get());
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions.size(), 0);
    BOOST_CHECK(label->ccb().outcome == nullptr);
    BOOST_CHECK_EQUAL(label->ccb().type, CCB::example_type::action);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    parse_label(lp, &p, "ccb slot", *label.get());
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions.size(), 0);
    BOOST_CHECK(label->ccb().outcome == nullptr);
    BOOST_CHECK_EQUAL(label->ccb().type, CCB::example_type::slot);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    parse_label(lp, &p, "ccb slot 1,3,4", *label.get());
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions.size(), 3);
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions[0], 1);
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions[1], 3);
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions[2], 4);
    BOOST_CHECK(label->ccb().outcome == nullptr);
    BOOST_CHECK_EQUAL(label->ccb().type, CCB::example_type::slot);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    parse_label(lp, &p, "ccb slot 1:1.0:0.5 3", *label.get());
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions.size(), 1);
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions[0], 3);
    BOOST_CHECK_CLOSE(label->ccb().outcome->cost, 1.0f, FLOAT_TOL);
    BOOST_CHECK_EQUAL(label->ccb().outcome->probabilities.size(), 1);
    BOOST_CHECK_EQUAL(label->ccb().outcome->probabilities[0].action, 1);
    BOOST_CHECK_CLOSE(label->ccb().outcome->probabilities[0].score, .5f, FLOAT_TOL);
    BOOST_CHECK_EQUAL(label->ccb().type, CCB::example_type::slot);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    parse_label(lp, &p, "ccb slot 1:-2.0:0.5,2:0.25,3:0.25 3,4", *label.get());
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions.size(), 2);
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions[0], 3);
    BOOST_CHECK_EQUAL(label->ccb().explicit_included_actions[1], 4);
    BOOST_CHECK_CLOSE(label->ccb().outcome->cost, -2.0f, FLOAT_TOL);
    BOOST_CHECK_EQUAL(label->ccb().outcome->probabilities.size(), 3);
    BOOST_CHECK_EQUAL(label->ccb().outcome->probabilities[0].action, 1);
    BOOST_CHECK_CLOSE(label->ccb().outcome->probabilities[0].score, .5f, FLOAT_TOL);
    BOOST_CHECK_EQUAL(label->ccb().outcome->probabilities[1].action, 2);
    BOOST_CHECK_CLOSE(label->ccb().outcome->probabilities[1].score, .25f, FLOAT_TOL);
    BOOST_CHECK_EQUAL(label->ccb().outcome->probabilities[2].action, 3);
    BOOST_CHECK_CLOSE(label->ccb().outcome->probabilities[2].score, .25f, FLOAT_TOL);
    BOOST_CHECK_EQUAL(label->ccb().type, CCB::example_type::slot);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    BOOST_REQUIRE_THROW(parse_label(lp, &p, "shared", *label.get()), VW::vw_exception);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    BOOST_REQUIRE_THROW(parse_label(lp, &p, "other shared", *label.get()), VW::vw_exception);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    BOOST_REQUIRE_THROW(parse_label(lp, &p, "other", *label.get()), VW::vw_exception);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    BOOST_REQUIRE_THROW(parse_label(lp, &p, "ccb unknown", *label.get()), VW::vw_exception);
  }
  {
    auto label = scoped_calloc_or_throw<polylabel>();
    BOOST_REQUIRE_THROW(parse_label(lp, &p, "ccb slot 1:1.0:0.5,4:0.7", *label.get()), VW::vw_exception);
  }
}

BOOST_AUTO_TEST_CASE(ccb_cache_label)
{
  io_buf io;
  io.space.resize(1000);
  io.space.end() = io.space.begin() + 1000;

  parser p{8 /*ring_size*/, false /*strict parse*/};

  auto lp = CCB::ccb_label_parser;
  auto label = scoped_calloc_or_throw<polylabel>();
  parse_label(lp, &p, "ccb slot 1:-2.0:0.5,2:0.25,3:0.25 3,4", *label.get());
  lp.cache_label(*label.get(), io);
  io.space.end() = io.head;
  io.head = io.space.begin();

  auto uncached_label = scoped_calloc_or_throw<polylabel>();
  lp.read_cached_label(nullptr, *uncached_label.get(), io);

  BOOST_CHECK_EQUAL(uncached_label->ccb().explicit_included_actions.size(), 2);
  BOOST_CHECK_EQUAL(uncached_label->ccb().explicit_included_actions[0], 3);
  BOOST_CHECK_EQUAL(uncached_label->ccb().explicit_included_actions[1], 4);
  BOOST_CHECK_CLOSE(uncached_label->ccb().outcome->cost, -2.0f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(uncached_label->ccb().outcome->probabilities.size(), 3);
  BOOST_CHECK_EQUAL(uncached_label->ccb().outcome->probabilities[0].action, 1);
  BOOST_CHECK_CLOSE(uncached_label->ccb().outcome->probabilities[0].score, .5f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(uncached_label->ccb().outcome->probabilities[1].action, 2);
  BOOST_CHECK_CLOSE(uncached_label->ccb().outcome->probabilities[1].score, .25f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(uncached_label->ccb().outcome->probabilities[2].action, 3);
  BOOST_CHECK_CLOSE(uncached_label->ccb().outcome->probabilities[2].score, .25f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(uncached_label->ccb().type, CCB::example_type::slot);
}

BOOST_AUTO_TEST_CASE(ccb_copy_label)
{
  parser p{8 /*ring_size*/, false /*strict parse*/};
  auto lp = CCB::ccb_label_parser;

  polylabel label;
  parse_label(lp, &p, "ccb slot 1:-2.0:0.5,2:0.25,3:0.25 3,4", label);

  polylabel copied_to = label;

  BOOST_CHECK_EQUAL(copied_to.ccb().explicit_included_actions.size(), 2);
  BOOST_CHECK_EQUAL(copied_to.ccb().explicit_included_actions[0], 3);
  BOOST_CHECK_EQUAL(copied_to.ccb().explicit_included_actions[1], 4);
  BOOST_CHECK_CLOSE(copied_to.ccb().outcome->cost, -2.0f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(copied_to.ccb().outcome->probabilities.size(), 3);
  BOOST_CHECK_EQUAL(copied_to.ccb().outcome->probabilities[0].action, 1);
  BOOST_CHECK_CLOSE(copied_to.ccb().outcome->probabilities[0].score, .5f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(copied_to.ccb().outcome->probabilities[1].action, 2);
  BOOST_CHECK_CLOSE(copied_to.ccb().outcome->probabilities[1].score, .25f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(copied_to.ccb().outcome->probabilities[2].action, 3);
  BOOST_CHECK_CLOSE(copied_to.ccb().outcome->probabilities[2].score, .25f, FLOAT_TOL);
  BOOST_CHECK_EQUAL(copied_to.ccb().type, CCB::example_type::slot);
}
