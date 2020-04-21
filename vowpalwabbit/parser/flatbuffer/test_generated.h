// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_TEST_H_
#define FLATBUFFERS_GENERATED_TEST_H_

#include "flatbuffers/flatbuffers.h"

struct Feature;
struct FeatureBuilder;

struct Datapoint;
struct DatapointBuilder;

struct Data;
struct DataBuilder;

struct Feature FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FeatureBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_VALUE = 6
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  bool KeyCompareLessThan(const Feature *o) const {
    return *name() < *o->name();
  }
  int KeyCompareWithValue(const char *val) const {
    return strcmp(name()->c_str(), val);
  }
  float value() const {
    return GetField<float>(VT_VALUE, 0.0f);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffsetRequired(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyField<float>(verifier, VT_VALUE) &&
           verifier.EndTable();
  }
};

struct FeatureBuilder {
  typedef Feature Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(Feature::VT_NAME, name);
  }
  void add_value(float value) {
    fbb_.AddElement<float>(Feature::VT_VALUE, value, 0.0f);
  }
  explicit FeatureBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FeatureBuilder &operator=(const FeatureBuilder &);
  flatbuffers::Offset<Feature> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Feature>(end);
    fbb_.Required(o, Feature::VT_NAME);
    return o;
  }
};

inline flatbuffers::Offset<Feature> CreateFeature(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    float value = 0.0f) {
  FeatureBuilder builder_(_fbb);
  builder_.add_value(value);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<Feature> CreateFeatureDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    float value = 0.0f) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  return CreateFeature(
      _fbb,
      name__,
      value);
}

struct Datapoint FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef DatapointBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_LABEL = 4,
    VT_FEATURES = 6
  };
  int32_t label() const {
    return GetField<int32_t>(VT_LABEL, 0);
  }
  const flatbuffers::Vector<flatbuffers::Offset<Feature>> *features() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<Feature>> *>(VT_FEATURES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_LABEL) &&
           VerifyOffset(verifier, VT_FEATURES) &&
           verifier.VerifyVector(features()) &&
           verifier.VerifyVectorOfTables(features()) &&
           verifier.EndTable();
  }
};

struct DatapointBuilder {
  typedef Datapoint Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_label(int32_t label) {
    fbb_.AddElement<int32_t>(Datapoint::VT_LABEL, label, 0);
  }
  void add_features(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Feature>>> features) {
    fbb_.AddOffset(Datapoint::VT_FEATURES, features);
  }
  explicit DatapointBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DatapointBuilder &operator=(const DatapointBuilder &);
  flatbuffers::Offset<Datapoint> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Datapoint>(end);
    return o;
  }
};

inline flatbuffers::Offset<Datapoint> CreateDatapoint(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t label = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Feature>>> features = 0) {
  DatapointBuilder builder_(_fbb);
  builder_.add_features(features);
  builder_.add_label(label);
  return builder_.Finish();
}

inline flatbuffers::Offset<Datapoint> CreateDatapointDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t label = 0,
    std::vector<flatbuffers::Offset<Feature>> *features = nullptr) {
  auto features__ = features ? _fbb.CreateVectorOfSortedTables<Feature>(features) : 0;
  return CreateDatapoint(
      _fbb,
      label,
      features__);
}

struct Data FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef DataBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_EXAMPLES = 4
  };
  const flatbuffers::Vector<flatbuffers::Offset<Datapoint>> *examples() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<Datapoint>> *>(VT_EXAMPLES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_EXAMPLES) &&
           verifier.VerifyVector(examples()) &&
           verifier.VerifyVectorOfTables(examples()) &&
           verifier.EndTable();
  }
};

struct DataBuilder {
  typedef Data Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_examples(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Datapoint>>> examples) {
    fbb_.AddOffset(Data::VT_EXAMPLES, examples);
  }
  explicit DataBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DataBuilder &operator=(const DataBuilder &);
  flatbuffers::Offset<Data> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Data>(end);
    return o;
  }
};

inline flatbuffers::Offset<Data> CreateData(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Datapoint>>> examples = 0) {
  DataBuilder builder_(_fbb);
  builder_.add_examples(examples);
  return builder_.Finish();
}

inline flatbuffers::Offset<Data> CreateDataDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<Datapoint>> *examples = nullptr) {
  auto examples__ = examples ? _fbb.CreateVector<flatbuffers::Offset<Datapoint>>(*examples) : 0;
  return CreateData(
      _fbb,
      examples__);
}

inline const Data *GetData(const void *buf) {
  return flatbuffers::GetRoot<Data>(buf);
}

inline const Data *GetSizePrefixedData(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<Data>(buf);
}

inline bool VerifyDataBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<Data>(nullptr);
}

inline bool VerifySizePrefixedDataBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<Data>(nullptr);
}

inline void FinishDataBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<Data> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedDataBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<Data> root) {
  fbb.FinishSizePrefixed(root);
}

#endif  // FLATBUFFERS_GENERATED_TEST_H_
