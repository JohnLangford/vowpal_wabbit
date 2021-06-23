#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "ut_util.h"

#include <vector>
#include <set>
#include <stdlib.h>
#include <streambuf>
#include <array>

#include <fstream>
#include "example_predict_builder.h"
#include "array_parameters.h"
#include "data.h"

using namespace ::testing;
using namespace vw_slim;
using namespace exploration;

// #define VW_SLIM_TEST_DEBUG "vwslim-debug.log"

// some gymnastics to re-use the float reading code
struct membuf : std::streambuf
{
  membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

std::vector<float> read_floats(std::istream& data)
{
  std::vector<float> floats;

  std::string line;
  while (std::getline(data, line)) floats.push_back((float)atof(line.c_str()));

  return floats;
}
std::vector<float> read_floats(unsigned char* data, unsigned int len)
{
  membuf mb((char*)data, (char*)(data + len));
  std::istream in(&mb);
  return read_floats(in);
}

std::vector<float> read_floats(const char* filename)
{
  std::ifstream data(filename);
  return read_floats(data);
}

std::pair<std::vector<float>, std::vector<float>> read_action_pdf_values(std::istream& data)
{
  std::vector<float> actions;
  std::vector<float> pdf_values;

  std::string line;
  while (std::getline(data, line))
  {
    std::string action = line.substr(0, line.find(",") + 1);
    std::string pdf_value = line.substr(line.find(",") + 1);
    actions.push_back((float)atof(action.c_str()));
    pdf_values.push_back((float)atof(pdf_value.c_str()));
  }

  return {actions, pdf_values};
}

std::pair<std::vector<float>, std::vector<float>> read_action_pdf_values(unsigned char* data, unsigned int len)
{
  membuf mb((char*)data, (char*)(data + len));
  std::istream in(&mb);
  return read_action_pdf_values(in);
}

struct test_data
{
  unsigned char* model;
  unsigned int model_len;
  unsigned char* pred;
  unsigned int pred_len;
};

#define TEST_DATA(input, filename)         \
  {                                        \
    if (!strcmp(input, #filename))         \
    {                                      \
      td.model = filename##_model;         \
      td.model_len = filename##_model_len; \
      td.pred = filename##_pred;           \
      td.pred_len = filename##_pred_len;   \
    }                                      \
  }

test_data get_test_data(const char* model_filename)
{
  test_data td;
  memset(&td, 0, sizeof(test_data));

  TEST_DATA(model_filename, regression_data_1);
  TEST_DATA(model_filename, regression_data_2);
  TEST_DATA(model_filename, regression_data_3);
  TEST_DATA(model_filename, regression_data_3);
  TEST_DATA(model_filename, regression_data_4);
  TEST_DATA(model_filename, regression_data_5);
  TEST_DATA(model_filename, regression_data_6);
  TEST_DATA(model_filename, regression_data_7);
  TEST_DATA(model_filename, regression_data_no_constant);
  TEST_DATA(model_filename, regression_data_ignore_linear);
  TEST_DATA(model_filename, multiclass_data_4);
  // TEST_DATA(model_filename, cb_data_epsilon_0_skype_jb);
  TEST_DATA(model_filename, cb_data_5);
  TEST_DATA(model_filename, cb_data_6);
  TEST_DATA(model_filename, cb_data_7);
  TEST_DATA(model_filename, cb_data_8);
  TEST_DATA(model_filename, cats);

  return td;
}

template <typename W>
void run_predict_in_memory(
    const char* model_filename, const char* data_filename, const char* prediction_reference_filename)
{
  std::vector<float> preds;
  std::vector<float> cats_actions;
  std::vector<float> cats_pdf_values;

  vw_predict<W> vw;
  // if files would be available
  test_data td = get_test_data(model_filename);
  ASSERT_EQ(S_VW_PREDICT_OK, vw.load((const char*)td.model, td.model_len));
  EXPECT_FALSE(vw.is_cb_explore_adf());

  float score;
  if (!strcmp(data_filename, "regression_data_1.txt"))
  {
    example_predict ex[2];
    // 1 |0 0:1
    example_predict_builder b0(&ex[0], (namespace_index)0);
    b0.push_feature(0, 1.f);
    // 1 |0 0:5
    example_predict_builder b1(&ex[1], (namespace_index)0);
    b1.push_feature(0, 5.f);

    for (auto& e : ex)
    {
      ASSERT_EQ(S_VW_PREDICT_OK, vw.predict(e, score));
      preds.push_back(score);
    }
  }
  else if (!strcmp(data_filename, "regression_data_2.txt"))
  {
    example_predict ex[2];
    // 1 |0 0:1 |a 0:2
    example_predict_builder b00(&ex[0], (namespace_index)0);
    b00.push_feature(0, 1.f);
    example_predict_builder b0a(&ex[0], "a");
    b0a.push_feature(0, 2.f);
    // 0 |c 0:3
    example_predict_builder b1(&ex[1], "c");
    b1.push_feature(0, 3.f);

    for (auto& e : ex)
    {
      ASSERT_EQ(S_VW_PREDICT_OK, vw.predict(e, score));
      preds.push_back(score);
    }
  }
  else if (!strcmp(data_filename, "regression_data_3.txt"))
  {
    example_predict ex[2];
    // 1 |a 0:1 |b 2:2
    example_predict_builder b0a(&ex[0], "a");
    b0a.push_feature(0, 1.f);
    example_predict_builder b0b(&ex[0], "b");
    b0b.push_feature(2, 2.f);
    // 0 |a 0:1 |b 2:4
    example_predict_builder b1a(&ex[1], "a");
    b1a.push_feature(0, 1.f);
    example_predict_builder b1b(&ex[1], "b");
    b1b.push_feature(2, 4.f);

    for (auto& e : ex)
    {
      ASSERT_EQ(S_VW_PREDICT_OK, vw.predict(e, score));
      preds.push_back(score);
    }
  }
  else if (!strcmp(data_filename, "regression_data_4.txt"))
  {
    example_predict ex[2];
    // 1 |a 0:1 |b 2:2 |c 3:3 |d 4:4
    example_predict_builder b0a(&ex[0], "a");
    b0a.push_feature(0, 1.f);
    example_predict_builder b0b(&ex[0], "b");
    b0b.push_feature(2, 2.f);
    example_predict_builder b0c(&ex[0], "c");
    b0c.push_feature(3, 3.f);
    example_predict_builder b0d(&ex[0], "d");
    b0d.push_feature(4, 4.f);
    // 0 |a 0:1 |b 2:4 |c 3:1 |d 1:2
    example_predict_builder b1a(&ex[1], "a");
    b1a.push_feature(0, 1.f);
    example_predict_builder b1b(&ex[1], "b");
    b1b.push_feature(2, 4.f);
    example_predict_builder b1c(&ex[1], "c");
    b1c.push_feature(3, 1.f);
    example_predict_builder b1d(&ex[1], "d");
    b1d.push_feature(1, 2.f);

    for (auto& e : ex)
    {
      ASSERT_EQ(S_VW_PREDICT_OK, vw.predict(e, score));
      preds.push_back(score);
    }
  }
  else if (!strcmp(data_filename, "regression_data_7.txt"))
  {
    example_predict ex[2];
    // 1 |a x:1 |b y:2
    example_predict_builder b0a(&ex[0], "a");
    b0a.push_feature_string("x", 1.f);
    example_predict_builder b0b(&ex[0], "b");
    b0b.push_feature_string("y", 2.f);
    // 0 |a x:1 |5 y:4
    example_predict_builder b1a(&ex[1], "a");
    b1a.push_feature_string("x", 1.f);
    example_predict_builder b1b(&ex[1], 5);
    b1b.push_feature_string("y", 4.f);

    for (auto& e : ex)
    {
      ASSERT_EQ(S_VW_PREDICT_OK, vw.predict(e, score));
      preds.push_back(score);
    }
  }
  else if (!strcmp(data_filename, "cats.txt"))
  {
    example ex[9];
    {
      // ca 185.121:0.657567:6.20426e-05 | 18-25 4 C 0 0 1 2 15 M
      example_predict_builder b0a(&ex[0], (char*)" ");
      b0a.push_feature_string((char*)"18-25", 1.f);
      b0a.push_feature(4, 1.f);
      b0a.push_feature_string((char*)"C", 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(2, 1.f);
      b0a.push_feature(15, 1.f);
      b0a.push_feature_string((char*)"M", 1.f);
    }

    {
      // ca 772.592:0.458316:6.20426e-05 | 18-25 4 C 4+ 0 1 2 15 M
      example_predict_builder b0a(&ex[1], (char*)" ");
      b0a.push_feature_string((char*)"18-25", 1.f);
      b0a.push_feature(4, 1.f);
      b0a.push_feature_string((char*)"C", 1.f);
      b0a.push_feature_string((char*)"4+", 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(2, 1.f);
      b0a.push_feature(15, 1.f);
      b0a.push_feature_string((char*)"M", 1.f);
    }

    {
      // ca 15140.6:0.31791:6.20426e-05 | 26-35 6 B 4+ 0 1 8 17 M
      example_predict_builder b0a(&ex[2], (char*)" ");
      b0a.push_feature_string((char*)"26-35", 1.f);
      b0a.push_feature(6, 1.f);
      b0a.push_feature_string((char*)"B", 1.f);
      b0a.push_feature_string((char*)"4+", 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(8, 1.f);
      b0a.push_feature(17, 1.f);
      b0a.push_feature_string((char*)"M", 1.f);
    }

    {
      // ca 14122:0.0199798:6.20426e-05 | 26-35 2 B 4+ 1 3 4 5 M
      example_predict_builder b0a(&ex[3], (char*)" ");
      b0a.push_feature_string((char*)"26-35", 1.f);
      b0a.push_feature(2, 1.f);
      b0a.push_feature_string((char*)"B", 1.f);
      b0a.push_feature_string((char*)"4+", 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(3, 1.f);
      b0a.push_feature(4, 1.f);
      b0a.push_feature(5, 1.f);
      b0a.push_feature_string((char*)"M", 1.f);
    }

    {
      // ca 11941.3:0.323547:6.20426e-05 | 55+ 17 C 1 0 1 6 8 M
      example_predict_builder b0a(&ex[4], (char*)" ");
      b0a.push_feature_string((char*)"55+", 1.f);
      b0a.push_feature(17, 1.f);
      b0a.push_feature_string((char*)"C", 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(6, 1.f);
      b0a.push_feature(8, 1.f);
      b0a.push_feature_string((char*)"M", 1.f);
    }

    { 
      // ca 15625.2:0.354757:6.20426e-05 | 26-35 7 A 1 0 5 8 12 M
      example_predict_builder b0a(&ex[5], (char*)" ");
      b0a.push_feature_string((char*)"26-35", 1.f);
      b0a.push_feature(7, 1.f);
      b0a.push_feature_string((char*)"A", 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(5, 1.f);
      b0a.push_feature(8, 1.f);
      b0a.push_feature(12, 1.f);
      b0a.push_feature_string((char*)"M", 1.f);
    }

    {
      // ca 13530:0.0975435:6.20426e-05 | 26-35 1 B 3 0 1 2 14 F
      example_predict_builder b0a(&ex[6], (char*)" ");
      b0a.push_feature_string((char*)"26-35", 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature_string((char*)"B", 1.f);
      b0a.push_feature(3, 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(2, 1.f);
      b0a.push_feature(14, 1.f);
      b0a.push_feature_string((char*)"F", 1.f);
    }

    {
      // ca 12715.1:0.168545:6.20426e-05 | 0-17 0 C 1 0 5 8 18 M
      example_predict_builder b0a(&ex[7], (char*)" ");
      b0a.push_feature_string((char*)"0-17", 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature_string((char*)"C", 1.f);
      b0a.push_feature(1, 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(5, 1.f);
      b0a.push_feature(8, 1.f);
      b0a.push_feature(18, 1.f);
      b0a.push_feature_string((char*)"M", 1.f);
    }

    {
      // ca 3111.9:0.320602:6.20426e-05 | 26-35 14 B 3 0 3 4 5 F
      example_predict_builder b0a(&ex[8], (char*)" ");
      b0a.push_feature_string((char*)"26-35", 1.f);
      b0a.push_feature(14, 1.f);
      b0a.push_feature_string((char*)"B", 1.f);
      b0a.push_feature(3, 1.f);
      b0a.push_feature(0, 1.f);
      b0a.push_feature(3, 1.f);
      b0a.push_feature(4, 1.f);
      b0a.push_feature(5, 1.f);
      b0a.push_feature_string((char*)"F", 1.f);
    }

    for (auto& e : ex)
    {
      float action = 0;
      float pdf_value = 0;
      ASSERT_EQ(S_VW_PREDICT_OK, vw.predict_cats(&e, action, pdf_value));
      cats_actions.push_back(action);
      cats_pdf_values.push_back(pdf_value);
    }
  }
  
  else
    FAIL() << "Unknown data file: " << data_filename;

  if (!strcmp(data_filename, "cats.txt"))
  {
    // compare output
    auto preds_expected = read_action_pdf_values(td.pred, td.pred_len);

    const float FLOAT_TOL = 0.1f;

    auto actual_actions = std::get<0>(preds_expected);
    auto actual_pdf_values = std::get<1>(preds_expected);

    EXPECT_THAT(cats_actions, Pointwise(FloatNearPointwise(FLOAT_TOL), actual_actions));
    EXPECT_THAT(cats_pdf_values, Pointwise(FloatNearPointwise(FLOAT_TOL), actual_pdf_values));
  }
  else
  {
    // compare output
    std::vector<float> preds_expected = read_floats(td.pred, td.pred_len);

    EXPECT_THAT(preds, Pointwise(FloatNearPointwise(1e-5f), preds_expected));
  }
}

enum PredictParamWeightType
{
  All,
  Sparse,
  Dense
};

struct PredictParam
{
  const char* model_filename;
  const char* data_filename;
  const char* prediction_reference_filename;
  PredictParamWeightType weight_type;
};

// nice rendering in unit tests
::std::ostream& operator<<(::std::ostream& os, const PredictParam& param)
{
  return os << param.model_filename << " " << param.data_filename << " "
            << (param.weight_type == PredictParamWeightType::Sparse ? "sparse" : "dense");
}

class PredictTest : public ::testing::TestWithParam<PredictParam>
{
};

TEST_P(PredictTest, Run)
{
  if (GetParam().weight_type == PredictParamWeightType::Sparse)
    run_predict_in_memory<sparse_parameters>(
        GetParam().model_filename, GetParam().data_filename, GetParam().prediction_reference_filename);
  else
    run_predict_in_memory<dense_parameters>(
        GetParam().model_filename, GetParam().data_filename, GetParam().prediction_reference_filename);
}

std::vector<PredictParam> GenerateTestParams()
{
  std::vector<PredictParam> fixtures;

  PredictParam predict_params[] = {
      {"regression_data_1", "regression_data_1.txt", "regression_data_1.pred", PredictParamWeightType::All},
      {"regression_data_2", "regression_data_2.txt", "regression_data_2.pred", PredictParamWeightType::All},
      {"regression_data_no_constant", "regression_data_1.txt", "regression_data_no-constant.pred",
          PredictParamWeightType::All},
      {"regression_data_ignore_linear", "regression_data_2.txt", "regression_data_ignore_linear.pred",
          PredictParamWeightType::All},
      {"regression_data_3", "regression_data_3.txt", "regression_data_3.pred", PredictParamWeightType::All},
      {"regression_data_4", "regression_data_4.txt", "regression_data_4.pred", PredictParamWeightType::All},
      {"regression_data_5", "regression_data_4.txt", "regression_data_5.pred", PredictParamWeightType::All},
      {"regression_data_6", "regression_data_3.txt", "regression_data_6.pred", PredictParamWeightType::Sparse},
      {"regression_data_7", "regression_data_7.txt", "regression_data_7.pred", PredictParamWeightType::All},
      {"cats", "cats.txt", "cats.pred", PredictParamWeightType::Dense}};

  for (int i = 0; i < sizeof(predict_params) / sizeof(PredictParam); i++)
  {
    PredictParam p = predict_params[i];
    if (p.weight_type != PredictParamWeightType::All)
      fixtures.push_back(p);
    else
    {
      for (int weight_type = PredictParamWeightType::Sparse; weight_type <= PredictParamWeightType::Dense;
           weight_type++)
      {
        p.weight_type = static_cast<PredictParamWeightType>(weight_type);
        fixtures.push_back(p);
      }
    }
  }

  return fixtures;
}

INSTANTIATE_TEST_SUITE_P(VowpalWabbitSlim, PredictTest, ::testing::ValuesIn(GenerateTestParams()));

struct InvalidModelParam
{
  const char* name;
  unsigned char* model;
  unsigned int model_len;
  std::set<size_t> undetectable_offsets;
  PredictParamWeightType weight_type;
};

// nice rendering in unit tests
::std::ostream& operator<<(::std::ostream& os, const InvalidModelParam& param) { return os << param.name; }

class InvalidModelTest : public ::testing::TestWithParam<InvalidModelParam>
{
};

TEST_P(InvalidModelTest, Run)
{
  const char* model_file = (const char*)GetParam().model;
  size_t model_file_size = (size_t)GetParam().model_len;

  auto& undetectable_offsets = GetParam().undetectable_offsets;

  for (size_t end = 0; end < model_file_size - 1; ++end)
  {
    // we're not able to detect if complete index:weight pairs are missing
    if (undetectable_offsets.find(end) != undetectable_offsets.end()) continue;

    // type parameterized and value parameterized test cases can't be combined:
    // https://stackoverflow.com/questions/8507385/google-test-is-there-a-way-to-combine-a-test-which-is-both-type-parameterized-a
    if (GetParam().weight_type == PredictParamWeightType::Sparse)
    {
      vw_predict<sparse_parameters> vw;
      EXPECT_NE(S_VW_PREDICT_OK, vw.load(&model_file[0], end)) << "partial model read until " << end << " didn't throw";
    }
    else
    {
      vw_predict<dense_parameters> vw;
      EXPECT_NE(S_VW_PREDICT_OK, vw.load(&model_file[0], end)) << "partial model read until " << end << " didn't throw";
    }
  }
}

InvalidModelParam invalid_model_param[] = {
    {"regression_data_1", regression_data_1_model, regression_data_1_model_len, {84, 92},
        PredictParamWeightType::Sparse},  // 2 weights
    {"regression_data_1", regression_data_1_model, regression_data_1_model_len, {84, 92},
        PredictParamWeightType::Dense},  // 2 weights
    {"regression_data_6", regression_data_6_model, regression_data_6_model_len, {99, 111, 123, 135},
        PredictParamWeightType::Sparse}  // 4 weights
};

INSTANTIATE_TEST_SUITE_P(VowpalWabbitSlim, InvalidModelTest, ::testing::ValuesIn(invalid_model_param));

TEST(VowpalWabbitSlim, multiclass_data_4)
{
  vw_predict<sparse_parameters> vw;
  test_data td = get_test_data("multiclass_data_4");
  ASSERT_EQ(0, vw.load((const char*)td.model, td.model_len));

  std::vector<float> out_scores;

  example_predict shared;
  // shared |a 0:1 5:12
  example_predict_builder bs(&shared, "a");
  bs.push_feature(0, 1.f);
  bs.push_feature(5, 12.f);

  example_predict ex[3];
  // 1:1.0 |b 0:1
  example_predict_builder b0(&ex[0], "b");
  b0.push_feature(0, 1.f);
  // 2:0.3 |b 0:2
  example_predict_builder b1(&ex[1], "b");
  b1.push_feature(0, 2.f);
  // 3:0.1 |b 0:3
  example_predict_builder b2(&ex[2], "b");
  b2.push_feature(0, 3.f);

  ASSERT_EQ(S_VW_PREDICT_OK, vw.predict(shared, ex, 3, out_scores));

  // 2:0.0386223,1:0.46983,0:0.901038
  std::vector<float> preds_expected = {0.901038f, 0.46983f, 0.0386223f};

  // compare output
  EXPECT_THAT(out_scores, Pointwise(FloatNearPointwise(1e-5f), preds_expected));
}

void cb_data_epsilon_0_skype_jb_test_runner(int call_type, int modality, int network_type, int platform,
    std::vector<int> ranking_expected, std::vector<float> pdf_expected)
{
  // load model.
  vw_predict<sparse_parameters> vw;
  test_data td = get_test_data("cb_data_epsilon_0_skype_jb");
  ASSERT_EQ(0, vw.load((const char*)td.model, td.model_len));

  // we have loaded the model and can push the features
  safe_example_predict features;
  vw_slim::example_predict_builder bOa(&features, "64");
  bOa.push_feature(static_cast<int>(call_type), 1.f);
  vw_slim::example_predict_builder bOb(&features, "16");
  bOb.push_feature(static_cast<int>(modality), 1.f);
  vw_slim::example_predict_builder bOc(&features, "32");
  bOc.push_feature(static_cast<int>(network_type), 1.f);
  vw_slim::example_predict_builder bOd(&features, "48");
  bOd.push_feature(static_cast<int>(platform), 1.f);

  // push actions
  const int min_delay_actions = 10;
  example_predict actions[min_delay_actions];
  for (int i = 0; i < min_delay_actions; i++)
  {
    vw_slim::example_predict_builder bOe(&actions[i], "80");
    bOe.push_feature(i, 1.f);
  }

  // predict CB value
  std::vector<float> pdfs;
  std::vector<int> rankings;
  int result = vw.predict("eid", features, actions, min_delay_actions, pdfs, rankings);

  // compare output with expected.
  EXPECT_EQ(result, 0);
  EXPECT_THAT(pdfs, Pointwise(FloatNearPointwise(1e-5f), pdf_expected));
  EXPECT_THAT(rankings, ranking_expected);
}

TEST(VowpalWabbitSlim, interaction_num_bits_bug)
{
  std::ifstream input("data/Delay_Margin_AudioNetworkPCR_all_cb_FF8.model", std::ios::in | std::ios::binary);
  input.seekg(0, std::ios::end);
  auto length = input.tellg();
  input.seekg(0, std::ios::beg);
  std::unique_ptr<char> buffer_ptr(new char[length]);
  input.read(buffer_ptr.get(),
      length);  // Extract how many bytes need to be decoded and resize the payload based on those bytes.

  vw_slim::vw_predict<sparse_parameters> vw;

  int result = vw.load(buffer_ptr.get(), length);
  EXPECT_EQ(result, 0);

  // we have loaded the model and can push the features
  example_predict features;

  // Test with the single namespace.
  vw_slim::example_predict_builder bOa(&features, "Features", vw.feature_index_num_bits());
  bOa.push_feature_string("Networkmobile", 1.f);
  bOa.push_feature_string("CallTypeP2P", 1.f);
  bOa.push_feature_string("PlatformAndroid", 1.f);
  bOa.push_feature_string("MediaTypeVideo", 1.f);

  const int MINDELAYACTIONS = 10;
  // push actions
  example_predict actions[MINDELAYACTIONS];
  for (int i = 0; i < MINDELAYACTIONS; i++)
  {
    vw_slim::example_predict_builder bOe(&actions[i], "80");
    bOe.push_feature(i, 1.f);
  }

  // generate UUID
  std::string uuidString("EventId_0");

  std::vector<float> pdfs;
  std::vector<int> rankings;

  result = vw.predict(uuidString.c_str(), features, actions, MINDELAYACTIONS, pdfs, rankings);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(rankings[0], 3);
}

// TEST(VowpalWabbitSlim, cb_data_epsilon_0_skype_jb)
// {
//   // Since the model is epsilon=0, the first entry should always be 0.
//   std::vector<float> pdf_expected = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

//   // {0, 1, 2, 0} => 1
//   std::vector<int> ranking_expected = {1, 0, 2, 3, 4, 5, 7, 6, 8, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(0, 1, 2, 0, ranking_expected, pdf_expected);

//   // {0, 1, 4, 0} => 1
//   ranking_expected = {1, 0, 2, 3, 4, 5, 6, 8, 7, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(0, 1, 4, 0, ranking_expected, pdf_expected);

//   // {0, 0, 2, 0} => 1
//   ranking_expected = {1, 2, 0, 3, 4, 5, 6, 7, 8, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(0, 0, 2, 0, ranking_expected, pdf_expected);

//   // {0, 0, 4, 0} => 1
//   ranking_expected = {1, 3, 2, 0, 4, 6, 5, 8, 7, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(0, 0, 4, 0, ranking_expected, pdf_expected);

//   // {2, 0, 4, 0} => 3
//   ranking_expected = {3, 1, 0, 2, 5, 4, 7, 6, 8, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(2, 0, 4, 0, ranking_expected, pdf_expected);

//   // {0, 1, 999, 0} => 2
//   ranking_expected = {2, 4, 6, 1, 0, 5, 3, 7, 8, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(0, 1, 999, 0, ranking_expected, pdf_expected);

//   // {0, 0, 999, 0} => 2
//   ranking_expected = {2, 4, 6, 1, 3, 5, 0, 7, 8, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(0, 0, 999, 0, ranking_expected, pdf_expected);

//   // {2, 0, 999, 0} => 2
//   ranking_expected = {2, 5, 4, 3, 6, 1, 0, 7, 8, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(2, 0, 999, 0, ranking_expected, pdf_expected);

//   // {999, 0, 4, 0} => 1
//   ranking_expected = {0, 1, 4, 2, 6, 3, 8, 7, 5, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(999, 0, 4, 0, ranking_expected, pdf_expected);

//   // {999, 0, 2, 0} => 2
//   ranking_expected = {0, 1, 4, 2, 7, 3, 6, 8, 5, 9};
//   cb_data_epsilon_0_skype_jb_test_runner(999, 0, 2, 0, ranking_expected, pdf_expected);
// }

void generate_cb_data_5(example_predict& shared, example_predict* ex)
{
  // shared |a 0:1 5:12
  example_predict_builder bs(&shared, "a");
  bs.push_feature(0, 1.f);
  bs.push_feature(5, 12.f);

  //  |b 0:1
  example_predict_builder b0(&ex[0], "b");
  b0.push_feature(0, 1.f);
  //  |b 0:2
  example_predict_builder b1(&ex[1], "b");
  b1.push_feature(0, 2.f);
  //  |b 0:3
  example_predict_builder b2(&ex[2], "b");
  b2.push_feature(0, 3.f);
}

std::string generate_string_seed(size_t i)
{
  std::stringstream s;
  s << "abcde" << i;
  return s.str();
}

struct CBPredictParam
{
  const char* description;
  const char* model_filename;

  int replications;

  std::vector<float> pdf_expected;
  std::vector<float> ranking_pdf_expected;
};

// nice rendering in unit tests
::std::ostream& operator<<(::std::ostream& os, const CBPredictParam& param)
{
  return os << param.description << " " << param.model_filename;
}

class CBPredictTest : public ::testing::TestWithParam<CBPredictParam>
{
};

TEST_P(CBPredictTest, CBRunPredict)
{
  vw_predict<sparse_parameters> vw;
  test_data td = get_test_data(GetParam().model_filename);
  ASSERT_EQ(S_VW_PREDICT_OK, vw.load((const char*)td.model, td.model_len));

  EXPECT_TRUE(vw.is_cb_explore_adf());

  std::vector<float> pdf_expected = GetParam().pdf_expected;
  std::vector<float> histogram(pdf_expected.size() * pdf_expected.size());

  example_predict shared;
  example_predict ex[3];

  std::vector<int> ranking;

  generate_cb_data_5(shared, ex);

  size_t rep = GetParam().replications;
  for (size_t i = 0; i < rep; i++)
  {
    std::vector<float> pdf;

    // invoke prediction
    ASSERT_EQ(S_VW_PREDICT_OK, vw.predict(generate_string_seed(i).c_str(), shared, ex, 3, pdf, ranking));

    ASSERT_EQ(pdf_expected.size(), ranking.size());
    for (size_t i = 0; i < ranking.size(); i++) histogram[i * ranking.size() + ranking[i]]++;
  }

  for (auto& d : histogram) d /= rep;

#ifdef VW_SLIM_TEST_DEBUG
    // std::fstream log(VW_SLIM_TEST_DEBUG, std::fstream::app);
    // for (size_t i = 0; i < 3; i++)
    //{
    //	log << "slot " << i << " ";
    //	for (size_t j = 0; j < 3; j++)
    //		log << histogram[i * 3 + j] << " ";
    //	log << std::endl;
    //}
#endif

  EXPECT_THAT(histogram, Pointwise(FloatNearPointwise(1e-2f), GetParam().ranking_pdf_expected));
}

CBPredictParam cb_predict_params[] = {
    {"CB Epsilon Greedy", "cb_data_5", 10000, {0.1f, 0.1f, 0.8f},
        {
            // see top action 2 w / 0.8
            0.1f, 0.09f, 0.80f,  // slot 0
                                 // most of the time we should see action 2
                                 // in 10% we should see the top action 0 swapped from top-slot to here
            0.0f, 0.90f, 0.09f,  // slot 1
                                 // most of the time we should see action 1
                                 // in 20% we should see the top action 2 swapped from top-slot to here
            0.89f, 0.0f, 0.10f,  // slot 2
        }},
    {"CB Softmax", "cb_data_6", 1000, {0.329f, 0.333f, 0.337f},
        {
            0.328f, 0.354f, 0.316f,  // slot 0
            0.000f, 0.644f, 0.354f,  // slot 1
            0.671f, 0.000f, 0.328f,  // slot 2
        }},
    {"CB Bag", "cb_data_7", 5, {0.0f, 0.0f, 1.0f},
        {
            0.0f, 0.0f, 1.0f,  // slot 0
            0.0f, 1.0f, 0.0f,  // slot 1
            1.0f, 0.0f, 0.0f,  // slot 2
        }},
    {"CB Bag Epsilon Greedy", "cb_data_8", 10000, {0.09f, 0.09f, 0.82f},
        {
            0.091f, 0.086f, 0.82f,  // slot 0
            0.00f, 0.91f, 0.086f,   // slot 1
            0.90f, 0.00f, 0.09f,    // slot 2
        }},
};

INSTANTIATE_TEST_SUITE_P(VowpalWabbitSlim, CBPredictTest, ::testing::ValuesIn(cb_predict_params));

// Test fixture to allow for both sparse and dense parameters
template <typename W>
class VwSlimTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(VwSlimTest);

typedef ::testing::Types<sparse_parameters, dense_parameters> WeightParameters;

TYPED_TEST_P(VwSlimTest, model_not_loaded)
{
  vw_predict<TypeParam> vw;
  example_predict ex;
  example_predict* actions = nullptr;
  std::vector<float> scores;
  std::vector<int> ranking;
  float score;

  EXPECT_EQ(E_VW_PREDICT_ERR_NO_MODEL_LOADED, vw.predict(ex, score));

  EXPECT_EQ(E_VW_PREDICT_ERR_NO_MODEL_LOADED, vw.predict(ex, actions, 0, scores));
  EXPECT_EQ(E_VW_PREDICT_ERR_NO_MODEL_LOADED, vw.predict("abc", ex, actions, 0, scores, ranking));
}

TYPED_TEST_P(VwSlimTest, model_reduction_mismatch)
{
  vw_predict<TypeParam> vw;
  example_predict ex;
  example_predict* actions = nullptr;
  std::vector<float> scores;
  std::vector<int> ranking;

  test_data td = get_test_data("regression_data_1");
  ASSERT_EQ(0, vw.load((const char*)td.model, td.model_len));

  EXPECT_EQ(E_VW_PREDICT_ERR_NO_A_CSOAA_MODEL, vw.predict(ex, actions, 0, scores));
  EXPECT_EQ(E_VW_PREDICT_ERR_NOT_A_CB_MODEL, vw.predict("abc", ex, actions, 0, scores, ranking));
}

TYPED_TEST_P(VwSlimTest, model_corrupted)
{
  test_data td = get_test_data("regression_data_1");

  vw_predict<TypeParam> vw;

  size_t num_bytes_to_corrupt = 10;
  float rand = 0;
  for (size_t i = 0; i < 100; i++)
  {
    std::vector<char> model_copy(td.model, td.model + td.model_len);
    for (size_t j = 0; j < num_bytes_to_corrupt; j++)
    {
      rand = uniform_random_merand48((uint64_t)rand);
      size_t random_idx = (size_t)(rand * model_copy.size());

      rand = uniform_random_merand48((uint64_t)rand);

      model_copy[random_idx] = (char)rand;

      ASSERT_NE(0, vw.load((const char*)&model_copy[0], model_copy.size()));
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(VwSlimTest, model_not_loaded, model_reduction_mismatch, model_corrupted);
INSTANTIATE_TYPED_TEST_SUITE_P(VowpalWabbitSlim, VwSlimTest, WeightParameters);

TEST(ColdStartModel, action_set_not_reordered)
{
  std::ifstream input("data/cold_start.model", std::ios::in | std::ios::binary);
  input.seekg(0, std::ios::end);
  auto length = input.tellg();
  input.seekg(0, std::ios::beg);
  std::unique_ptr<char> buffer_ptr(new char[length]);
  input.read(buffer_ptr.get(), length);

  vw_slim::vw_predict<sparse_parameters> vw;

  int result = vw.load(buffer_ptr.get(), length);
  EXPECT_EQ(result, 0);

  example_predict features;

  vw_slim::example_predict_builder bOa(&features, "Features", vw.feature_index_num_bits());
  bOa.push_feature_string("f1", 1.f);

  const int NUM_ACTIONS = 5;
  std::array<example_predict, NUM_ACTIONS> actions;
  for (int i = 0; i < actions.size(); i++)
  {
    vw_slim::example_predict_builder bOe(&actions[i], "ActionFeatures");
    bOe.push_feature(i, 1.f);
  }

  std::string uuidString("EventId_0");

  std::vector<float> pdfs;
  std::vector<int> rankings;

  result = vw.predict(uuidString.c_str(), features, actions.data(), NUM_ACTIONS, pdfs, rankings);

  EXPECT_GT(pdfs[0], 0.8);
  EXPECT_GT(pdfs[0], pdfs[1]);
  EXPECT_THAT(rankings, ElementsAre(0, 1, 2, 3, 4));
}


// TEST(NewTest, CATS)
// {
//   vw_predict<dense_parameters> vw;
//   test_data td = get_test_data("cats");
//   ASSERT_EQ(0, vw.load((const char*)td.model, td.model_len));

//   example ex[1];
//   // 1:1.0 |b 0:1
//   example_predict_builder b0(&ex[0], " ");
//   b0.push_feature_string("room=Living_Room", 1.f);
//   b0.push_feature_string("time_of_day=morning", 1.f);

//   float action;
//   float pdf_value;

//   ASSERT_EQ(S_VW_PREDICT_OK, vw.predict_cats(&ex[0], action, pdf_value));

//   // compare output
//   std::vector<float> output_a = {action};
//   std::vector<float> output_p = {pdf_value};
  
//   EXPECT_THAT(output_a, Pointwise(FloatNearPointwise(1e-5f), {0.0150219}));
//   EXPECT_THAT(output_p, Pointwise(FloatNearPointwise(1e-5f), {0.0005}));
// }