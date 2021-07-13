// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "test_common.h"

#include <vector>
#include "parser/flatbuffer/parse_example_flatbuffer.h"
#include "constant.h"

flatbuffers::Offset<void> get_label(flatbuffers::FlatBufferBuilder& builder, VW::parsers::flatbuffer::Label label_type)
{
  flatbuffers::Offset<void> label;
  if (label_type == VW::parsers::flatbuffer::Label_SimpleLabel)
    label = VW::parsers::flatbuffer::CreateSimpleLabel(builder, 0.0, 1.0).Union();

  return label;
}

flatbuffers::Offset<VW::parsers::flatbuffer::ExampleRoot> sample_flatbuffer_collection(
    flatbuffers::FlatBufferBuilder& builder, VW::parsers::flatbuffer::Label label_type)
{
  std::vector<flatbuffers::Offset<VW::parsers::flatbuffer::Example>> examples;
  std::vector<flatbuffers::Offset<VW::parsers::flatbuffer::Namespace>> namespaces;
  std::vector<flatbuffers::Offset<VW::parsers::flatbuffer::Feature>> fts;

  auto label = get_label(builder, label_type);

  fts.push_back(VW::parsers::flatbuffer::CreateFeatureDirect(builder, "hello", 2.23f, constant));
  namespaces.push_back(
      VW::parsers::flatbuffer::CreateNamespaceDirect(builder, nullptr, constant_namespace, &fts, constant_namespace));
  examples.push_back(VW::parsers::flatbuffer::CreateExampleDirect(builder, &namespaces, label_type, label));

  auto eg_collection = VW::parsers::flatbuffer::CreateExampleCollectionDirect(builder, &examples);
  return CreateExampleRoot(builder, VW::parsers::flatbuffer::ExampleType_ExampleCollection, eg_collection.Union());
}

flatbuffers::Offset<VW::parsers::flatbuffer::ExampleRoot> sample_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, VW::parsers::flatbuffer::Label label_type)
{
  std::vector<flatbuffers::Offset<VW::parsers::flatbuffer::Namespace>> namespaces;
  std::vector<flatbuffers::Offset<VW::parsers::flatbuffer::Feature>> fts;

  auto label = get_label(builder, label_type);

  fts.push_back(VW::parsers::flatbuffer::CreateFeatureDirect(builder, "hello", 2.23f, constant));
  namespaces.push_back(
      VW::parsers::flatbuffer::CreateNamespaceDirect(builder, nullptr, constant_namespace, &fts, constant_namespace));
  auto example = VW::parsers::flatbuffer::CreateExampleDirect(builder, &namespaces, label_type, label);

  return CreateExampleRoot(builder, VW::parsers::flatbuffer::ExampleType_Example, example.Union());
}

BOOST_AUTO_TEST_CASE(test_flatbuffer_standalone_example)
{
  auto all = VW::initialize("--no_stdin --quiet --flatbuffer", nullptr, false, nullptr, nullptr);

  flatbuffers::FlatBufferBuilder builder;

  auto root = sample_flatbuffer(builder, VW::parsers::flatbuffer::Label_SimpleLabel);
  builder.FinishSizePrefixed(root);

  uint8_t* buf = builder.GetBufferPointer();
  int size = builder.GetSize();

  v_array<example*> examples;
  examples.push_back(&VW::get_unused_example(all));
  all->flat_converter->parse_examples(all, examples, buf);

  auto example = all->flat_converter->data()->example_obj_as_Example();
  BOOST_CHECK_EQUAL(example->namespaces()->Length(), 1);
  BOOST_CHECK_EQUAL(example->namespaces()->Get(0)->feats()->size(), 1);
  BOOST_CHECK_CLOSE(example->label_as_SimpleLabel()->label(), 0.0, FLOAT_TOL);
  BOOST_CHECK_CLOSE(example->label_as_SimpleLabel()->weight(), 1.0, FLOAT_TOL);
  BOOST_CHECK_EQUAL(example->namespaces()->Get(0)->hash(), constant_namespace);
  BOOST_CHECK_EQUAL(example->namespaces()->Get(0)->feats()->Get(0)->name()->c_str(), "hello");
  BOOST_CHECK_EQUAL(example->namespaces()->Get(0)->feats()->Get(0)->hash(), constant);
  BOOST_CHECK_CLOSE(example->namespaces()->Get(0)->feats()->Get(0)->value(), 2.23, FLOAT_TOL);

  // Check vw example
  BOOST_CHECK_EQUAL(examples.size(), 1);
  BOOST_CHECK_CLOSE(examples[0]->l.simple.label, 0.f, FLOAT_TOL);
  const auto& red_features = examples[0]->_reduction_features.template get<simple_label_reduction_features>();
  BOOST_CHECK_CLOSE(red_features.weight, 1.f, FLOAT_TOL);

  auto* constant_feat_group = examples[0]->feature_space.get_or_null(constant_namespace, constant_namespace);
  BOOST_CHECK(constant_feat_group != nullptr);
  BOOST_CHECK_CLOSE(constant_feat_group->values[0], 2.23f, FLOAT_TOL);

  VW::finish_example(*all, *examples[0]);
  VW::finish(*all);
}

BOOST_AUTO_TEST_CASE(test_flatbuffer_collection)
{
  auto all = VW::initialize("--no_stdin --quiet --flatbuffer", nullptr, false, nullptr, nullptr);

  flatbuffers::FlatBufferBuilder builder;

  auto root = sample_flatbuffer_collection(builder, VW::parsers::flatbuffer::Label_SimpleLabel);
  builder.FinishSizePrefixed(root);

  uint8_t* buf = builder.GetBufferPointer();
  int size = builder.GetSize();

  v_array<example*> examples;
  examples.push_back(&VW::get_unused_example(all));
  all->flat_converter->parse_examples(all, examples, buf);

  auto collection_examples = all->flat_converter->data()->example_obj_as_ExampleCollection()->examples();
  BOOST_CHECK_EQUAL(collection_examples->Length(), 1);
  BOOST_CHECK_EQUAL(collection_examples->Get(0)->namespaces()->Length(), 1);
  BOOST_CHECK_EQUAL(collection_examples->Get(0)->namespaces()->Get(0)->feats()->size(), 1);
  BOOST_CHECK_CLOSE(collection_examples->Get(0)->label_as_SimpleLabel()->label(), 0.0, FLOAT_TOL);
  BOOST_CHECK_CLOSE(collection_examples->Get(0)->label_as_SimpleLabel()->weight(), 1.0, FLOAT_TOL);
  BOOST_CHECK_EQUAL(collection_examples->Get(0)->namespaces()->Get(0)->hash(), constant_namespace);
  BOOST_CHECK_EQUAL(collection_examples->Get(0)->namespaces()->Get(0)->feats()->Get(0)->name()->c_str(), "hello");
  BOOST_CHECK_EQUAL(collection_examples->Get(0)->namespaces()->Get(0)->feats()->Get(0)->hash(), constant);
  BOOST_CHECK_CLOSE(collection_examples->Get(0)->namespaces()->Get(0)->feats()->Get(0)->value(), 2.23, FLOAT_TOL);

  // check vw example
  BOOST_CHECK_EQUAL(examples.size(), 1);
  BOOST_CHECK_CLOSE(examples[0]->l.simple.label, 0.f, FLOAT_TOL);
  const auto& red_features = examples[0]->_reduction_features.template get<simple_label_reduction_features>();
  BOOST_CHECK_CLOSE(red_features.weight, 1.f, FLOAT_TOL);

  auto* constant_feat_group = examples[0]->feature_space.get_or_null(constant_namespace, constant_namespace);
  BOOST_CHECK(constant_feat_group != nullptr);
  BOOST_CHECK_CLOSE(constant_feat_group->values[0], 2.23f, FLOAT_TOL);

  VW::finish_example(*all, *examples[0]);
  VW::finish(*all);
}
