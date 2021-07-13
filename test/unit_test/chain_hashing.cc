#ifndef STATIC_LINK_VW
#define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "test_common.h"
#include "vw.h"

BOOST_AUTO_TEST_CASE(chain_hashing_between_formats)
{
  feature_index txt_idx;
  feature_index json_idx;

  std::string text("1 |f a:b");
  std::string json_text = R"(
    {
      "_label": 1,
      "f": {
        "a": "b"
      }
    })";

  auto vw = VW::initialize("--quiet --chain_hash", nullptr, false, nullptr, nullptr);
  {
    multi_ex examples;
    examples.push_back(&VW::get_unused_example(vw));
    auto example = examples[0];
    VW::read_line(*vw, example, text.c_str());
    auto& indices = example->feature_space.get('f', VW::hash_space(*vw, "f")).indicies;
    txt_idx = indices[0];
    VW::finish_example(*vw, examples);
  }
  {
    auto examples = parse_json(*vw, json_text);
    auto example = examples[0];

    auto& indices = example->feature_space.get('f', VW::hash_space(*vw, "f")).indicies;
    json_idx = indices[0];
    VW::finish_example(*vw, examples);
  }
  BOOST_CHECK_EQUAL(txt_idx, json_idx);
  VW::finish(*vw);
}
