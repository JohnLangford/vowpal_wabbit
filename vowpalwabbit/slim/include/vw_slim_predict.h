#pragma once

#include <memory>
#include <vector>
#include <string>
#include <array>

// avoid mmap dependency
#define DISABLE_SHARED_WEIGHTS

#include "example_predict.h"
#include "explore.h"
#include "gd_predict.h"
#include "model_parser.h"
#include "interactions.h"
#include "opts.h"
#include "cats_tree.h"
#include "learner_no_throw.h"
#include "binary.h"
#include "get_pmf.h"
#include "pmf_to_pdf.h"
#include "cb_explore_pdf.h"
#include "cats_pdf.h"
#include "sample_pdf.h"
#include "cats.h"

// forward declarations
namespace GD
{
struct gd;
}  // namespace GD

namespace VW
{
namespace cats_tree
{
struct cats_tree;
void init(uint32_t num_actions, uint32_t bandwidth);
// what I will call from here with control
void predict(cats_tree& ot, VW::LEARNER::single_learner& base, example& ec);
}  // namespace cats_tree
namespace binary
{
template <bool is_learn>
void predict_or_learn(char&, VW::LEARNER::single_learner& base, example& ec);
}
}  // namespace VW

void VW::cats_tree::learn(VW::cats_tree::cats_tree& tree, VW::LEARNER::single_learner& base, example& ec) {}
void VW::continuous_action::learn(VW::continuous_action::get_pmf& reduction, VW::LEARNER::single_learner&, example& ec)
{
}
void VW::pmf_to_pdf::learn(VW::pmf_to_pdf::reduction& data, VW::LEARNER::single_learner&, example& ec) {}
void VW::continuous_action::learn(
    VW::continuous_action::cb_explore_pdf& reduction, VW::LEARNER::single_learner&, example& ec)
{
}
void VW::continuous_action::cats_pdf::learn(
    VW::continuous_action::cats_pdf::cats_pdf& reduction, VW::LEARNER::single_learner&, example& ec)
{
}
void VW::continuous_action::learn(
    VW::continuous_action::sample_pdf& reduction, VW::LEARNER::single_learner&, example& ec)
{
}

void VW::continuous_action::cats::learn(
    VW::continuous_action::cats::cats& reduction, VW::LEARNER::single_learner&, example& ec)
{
}

namespace vw_slim
{
namespace internal
{
template <typename It1, typename It2>
class location_reference;

template <typename It1, typename It2>
class location_value
{
public:
  using Val1 = typename std::iterator_traits<It1>::value_type;
  using Val2 = typename std::iterator_traits<It2>::value_type;

  Val1 _val1;
  Val2 _val2;

  location_value(const location_reference<It1, It2>& rhs) : _val1(*rhs._ptr1), _val2(*rhs._ptr2) {}
};

template <typename It1, typename It2>
class location_reference
{
public:
  It1 _ptr1;
  It2 _ptr2;
  using Ref = location_reference<It1, It2>;
  using Val = location_value<It1, It2>;

  location_reference() = delete;

  location_reference(const Ref& other) : _ptr1(other._ptr1), _ptr2(other._ptr2) {}

  location_reference(It1 first, It2 second) : _ptr1(first), _ptr2(second) {}

  location_reference& operator=(const Val& rhs)
  {
    *_ptr1 = rhs._val1;
    *_ptr2 = rhs._val2;
    return *this;
  }

  location_reference& operator=(const Ref& rhs)
  {
    *_ptr1 = *rhs._ptr1;
    *_ptr2 = *rhs._ptr2;
    return *this;
  }

  friend void swap(const location_reference& a, const location_reference& b)
  {
    std::iter_swap(a._ptr1, b._ptr1);
    std::iter_swap(a._ptr2, b._ptr2);
  }
};

template <typename It1, typename It2>
class collection_pair_iterator
    : public std::iterator<std::random_access_iterator_tag, location_value<It1, It2>,  // value type
          size_t,                                                                      // difference type
          location_reference<It1, It2>                                                 // reference_type
          >
{
  It1 _ptr1;
  It2 _ptr2;

public:
  using Iter = collection_pair_iterator<It1, It2>;
  using Loc = location_value<It1, It2>;
  using Ref = location_reference<It1, It2>;

  collection_pair_iterator(It1 first, It2 second) : _ptr1(first), _ptr2(second) {}

  // must support: a) default-construction, b) copy-construction, c) copy-assignment, d) destruction
  collection_pair_iterator() = default;
  collection_pair_iterator(const collection_pair_iterator& rhs) = default;
  collection_pair_iterator& operator=(const collection_pair_iterator& rhs) = default;

  // must support: a == b; a != b;
  bool operator==(const Iter& rhs) const { return (_ptr1 == rhs._ptr1 && _ptr2 == rhs._ptr2); }
  bool operator!=(const Iter& rhs) const { return !(operator==(rhs)); }

  // must support: b = *a; a->m ;
  // must support: *a = t;
  Ref operator*() { return Ref(_ptr1, _ptr2); }  // Non-conforming - normally returns loc&

  // VS library 14.25.28610 requires operator[] (since it's a random access iterator)
  Ref operator[](size_t n) { return Ref(_ptr1 + n, _ptr2 + n); }

  // must support: ++a; a++; *a++;
  Iter& operator++()
  {
    ++_ptr1;
    ++_ptr2;
    return *this;
  }

  Iter operator++(int)
  {
    Iter ret(*this);
    ++_ptr1;
    ++_ptr2;
    return ret;
  }

  // must support: --a; a--; *a--;
  Iter& operator--()
  {
    --_ptr1;
    --_ptr2;
    return *this;
  }

  Iter operator--(int)
  {
    Iter ret(*this);
    --_ptr1;
    --_ptr2;
    return ret;
  }

  // must support: a + n; n + a; a - n; a - b
  Iter operator+(const size_t n) const { return Iter(_ptr1 + n, _ptr2 + n); }
  // friend Iter operator+(const size_t, const Iter&);
  Iter operator-(const size_t n) const { return Iter(_ptr1 - n, _ptr2 - n); }
  size_t operator-(const Iter& rhs) const { return _ptr1 - rhs._ptr1; }

  // must support: a > b; a < b; a <= b; a >= b;
  bool operator<(const Iter& rhs) const { return _ptr1 < rhs._ptr1; }
  bool operator>(const Iter& rhs) const { return _ptr1 > rhs._ptr1; }
  bool operator<=(const Iter& rhs) const { return _ptr1 <= rhs._ptr1; }
  bool operator>=(const Iter& rhs) const { return _ptr1 >= rhs._ptr1; }

  // must support: a += n; a -= n;
  Iter& operator+=(size_t n)
  {
    _ptr1 += n;
    _ptr2 += n;
    return *this;
  }

  Iter& operator-=(size_t n)
  {
    _ptr1 -= n;
    _ptr2 -= n;
    return *this;
  }
};
}  // namespace internal
}  // namespace vw_slim

namespace vw_slim
{
/**
 * @brief Exploration algorithm specified by the model.
 */
enum vw_predict_exploration
{
  epsilon_greedy,
  softmax,
  bag
};

uint64_t ceil_log_2(uint64_t v);

// this guard assumes that namespaces are added in order
// the complete feature_space of the added namespace is cleared afterwards
class namespace_copy_guard
{
  example_predict& _ex;
  unsigned char _ns;
  bool _remove_ns;

public:
  namespace_copy_guard(example_predict& ex, unsigned char ns);
  ~namespace_copy_guard();

  void feature_push_back(feature_value v, feature_index idx);
};

class feature_offset_guard
{
  example_predict& _ex;
  uint64_t _old_ft_offset;

public:
  feature_offset_guard(example_predict& ex, uint64_t ft_offset);
  ~feature_offset_guard();
};

class stride_shift_guard
{
  example_predict& _ex;
  uint64_t _shift;

public:
  stride_shift_guard(example_predict& ex, uint64_t shift);
  ~stride_shift_guard();
};

template <typename W>
struct predict_info
{
  std::unique_ptr<W> _weights;
  std::vector<std::vector<namespace_index>> _interactions;
  INTERACTIONS::interactions_generator _generate_interactions;
  bool _contains_wildcard;
  std::array<bool, NUM_NAMESPACES> _ignore_linear;
};

/**
 * @brief Vowpal Wabbit slim predictor. Supports: regression, multi-class classification and contextual bandits.
 */
template <typename W>
class vw_predict
{
  std::string _id;
  std::string _version;
  std::string _command_line_arguments;
  bool _no_constant;
  int _cats;
  float _bandwidth;
  float _min_value;
  float _max_value;

  vw_predict_exploration _exploration;
  float _minimum_epsilon;
  float _epsilon;
  float _lambda;
  int _bag_size;
  uint32_t _num_bits;

  uint32_t _stride_shift;
  bool _model_loaded;
  uint64_t random_seed = 0;

  VW::LEARNER::learner<VW::continuous_action::cats::cats, example>* _cats_learner;
  std::unique_ptr<predict_info<W>> _predict_info;

public:
  vw_predict() : _model_loaded(false) {}

  static void predict_gd(predict_info<W>& vw_slim, VW::LEARNER::base_learner&, example& ec)
  {
    if (vw_slim._contains_wildcard)
    {
      vw_slim._generate_interactions.template update_interactions_if_new_namespace_seen<
          INTERACTIONS::generate_namespace_combinations_with_repetition, false>(vw_slim._interactions, ec.indices);
      ec.pred.scalar = GD::inline_predict<W>(*vw_slim._weights, false, vw_slim._ignore_linear,
          vw_slim._generate_interactions.generated_interactions,
          /* permutations */ false, ec);
    }
    else
    {
      ec.pred.scalar = GD::inline_predict<W>(
          *vw_slim._weights, false, vw_slim._ignore_linear, vw_slim._interactions, /* permutations */ false, ec);
    }
  }

  static void learn_gd(predict_info<W>& vw_slim, VW::LEARNER::base_learner&, example& ec)
  {
    std::cout << "hey" << std::endl;
  }

  /**
   * @brief Reads the Vowpal Wabbit model from the supplied buffer (produced using vw -f <modelname>)
   *
   * @param model The binary model.
   * @param length The length of the binary model.
   * @return int Returns 0 (S_VW_PREDICT_OK) if succesful, otherwise one of the error codes (see E_VW_PREDICT_ERR_*).
   */
  int load(const char* model, size_t length)
  {
    if (!model || length == 0) return E_VW_PREDICT_ERR_INVALID_MODEL;

    _model_loaded = false;

    _predict_info = VW::make_unique<predict_info<W>>();
    _predict_info->_contains_wildcard = false;

    // required for inline_predict
    _predict_info->_ignore_linear.fill(false);

    model_parser mp(model, length);

    // parser_regressor.cc: save_load_header
    RETURN_ON_FAIL(mp.read_string<false>("version", _version));

    // read model id
    RETURN_ON_FAIL(mp.read_string<true>("model_id", _id));

    RETURN_ON_FAIL(mp.skip(sizeof(char)));   // "model character"
    RETURN_ON_FAIL(mp.skip(sizeof(float)));  // "min_label"
    RETURN_ON_FAIL(mp.skip(sizeof(float)));  // "max_label"

    RETURN_ON_FAIL(mp.read("num_bits", _num_bits));

    RETURN_ON_FAIL(mp.skip(sizeof(uint32_t)));  // "lda"

    uint32_t ngram_len;
    RETURN_ON_FAIL(mp.read("ngram_len", ngram_len));
    mp.skip(3 * ngram_len);

    uint32_t skips_len;
    RETURN_ON_FAIL(mp.read("skips_len", skips_len));
    mp.skip(3 * skips_len);

    RETURN_ON_FAIL(mp.read_string<true>("file_options", _command_line_arguments));

    // command line arg parsing
    _no_constant = _command_line_arguments.find("--noconstant") != std::string::npos;

    // only 0-valued hash_seed supported
    int hash_seed;
    if (find_opt_int(_command_line_arguments, "--hash_seed", hash_seed) && hash_seed)
      return E_VW_PREDICT_ERR_HASH_SEED_NOT_SUPPORTED;

    find_opt_uint64_t(_command_line_arguments, "--random_seed", random_seed);

    _predict_info->_interactions.clear();
    find_opt(_command_line_arguments, "-q", _predict_info->_interactions);
    find_opt(_command_line_arguments, "--quadratic", _predict_info->_interactions);
    find_opt(_command_line_arguments, "--cubic", _predict_info->_interactions);
    find_opt(_command_line_arguments, "--interactions", _predict_info->_interactions);

    // VW performs the following transformation as a side-effect of looking for duplicates.
    // This affects how interaction hashes are generated.
    std::vector<std::vector<namespace_index>> vec_sorted;
    for (auto& interaction : _predict_info->_interactions)
    { std::sort(std::begin(interaction), std::end(interaction)); }

    for (const auto& inter : _predict_info->_interactions)
    {
      if (INTERACTIONS::contains_wildcard(inter))
      {
        _predict_info->_contains_wildcard = true;
        break;
      }
    }

    // TODO: take --cb_type dr into account
    uint64_t num_weights = 0;

    if (_command_line_arguments.find("--cb_explore_adf") != std::string::npos)
    {
      // parse exploration options
      if (find_opt_int(_command_line_arguments, "--bag", _bag_size))
      {
        _exploration = vw_predict_exploration::bag;
        num_weights = _bag_size;

        // check for additional minimum epsilon greedy
        _minimum_epsilon = 0.f;
        find_opt_float(_command_line_arguments, "--epsilon", _minimum_epsilon);
      }
      else if (_command_line_arguments.find("--softmax") != std::string::npos)
      {
        if (find_opt_float(_command_line_arguments, "--lambda", _lambda))
        {
          if (_lambda > 0)  // Lambda should always be negative because we are using a cost basis.
            _lambda = -_lambda;
          _exploration = vw_predict_exploration::softmax;
        }
      }
      else if (find_opt_float(_command_line_arguments, "--epsilon", _epsilon))
        _exploration = vw_predict_exploration::epsilon_greedy;
      else
        return E_VW_PREDICT_ERR_CB_EXPLORATION_MISSING;
    }

    std::unique_ptr<VW::pmf_to_pdf::reduction> pmf_to_pdf = nullptr;
    std::unique_ptr<VW::cats_tree::cats_tree> cats_tree = nullptr;
    if (_command_line_arguments.find("--cats") != std::string::npos)
    {
      find_opt_int(_command_line_arguments, "--cats", _cats);
      find_opt_float(_command_line_arguments, "--bandwidth", _bandwidth);
      find_opt_float(_command_line_arguments, "--min_value", _min_value);
      find_opt_float(_command_line_arguments, "--max_value", _max_value);
      if (!find_opt_float(_command_line_arguments, "--epsilon", _epsilon)) { _epsilon = 0.05f; }

      pmf_to_pdf = VW::make_unique<VW::pmf_to_pdf::reduction>();
      pmf_to_pdf->init(_cats, _bandwidth, _min_value, _max_value, /* first_only*/ false, /*bandwidth supplied*/ true);
      cats_tree = VW::make_unique<VW::cats_tree::cats_tree>();
      if (!cats_tree->init(_cats, pmf_to_pdf->tree_bandwidth)) { return E_VW_PREDICT_ERR_INVALID_MODEL; }
      num_weights = cats_tree->learner_count();
    }

    // VW style check_sum validation
    uint32_t check_sum_computed = mp.checksum();

    // perform check sum check
    uint32_t check_sum_len;
    RETURN_ON_FAIL((mp.read<uint32_t, false>("check_sum_len", check_sum_len)));
    if (check_sum_len != sizeof(uint32_t)) return E_VW_PREDICT_ERR_INVALID_MODEL;

    uint32_t check_sum;
    RETURN_ON_FAIL((mp.read<uint32_t, false>("check_sum", check_sum)));

    if (check_sum_computed != check_sum) return E_VW_PREDICT_ERR_INVALID_MODEL_CHECK_SUM;

    if (_command_line_arguments.find("--cb_adf") != std::string::npos)
    {
      RETURN_ON_FAIL(mp.skip(sizeof(uint64_t)));  // cb_adf.cc: event_sum
      RETURN_ON_FAIL(mp.skip(sizeof(uint64_t)));  // cb_adf.cc: action_sum
    }

    // gd.cc: save_load
    bool gd_resume;
    RETURN_ON_FAIL(mp.read("resume", gd_resume));
    if (gd_resume) return E_VW_PREDICT_ERR_GD_RESUME_NOT_SUPPORTED;

    // read sparse weights into dense
    uint64_t weight_length = (uint64_t)1 << _num_bits;
    _stride_shift = (uint32_t)ceil_log_2(num_weights);

    RETURN_ON_FAIL(mp.read_weights<W>(_predict_info->_weights, _num_bits, _stride_shift));

    // TODO return error if activated but these are not initialized
    if (pmf_to_pdf != nullptr && cats_tree != nullptr)
    {
      // TODO plug in interactions reduction and maybe metrics (or maybe figure out how to use setup functions)

      // cats_tree -> binary -> gd
      // create gd
      auto* ret = VW::LEARNER::make_base_learner(
          std::move(_predict_info), learn_gd, predict_gd, "setup::gd", prediction_type_t::scalar, label_type_t::simple)
                      .set_params_per_weight(1)
                      .build();

      bool success;
      auto* gd = VW::LEARNER::as_singleline_no_throw(VW::LEARNER::make_base(*ret), success);
      if (!success) { return E_VW_PREDICT_ERR_INVALID_MODEL; }
      // create binary learner

      auto* binary_reduction = VW::LEARNER::make_no_data_reduction_learner(
          gd, VW::binary::predict_or_learn<true>, VW::binary::predict_or_learn<false>, "setup::binary")
                                   .set_learn_returns_prediction(true)
                                   .build();

      auto* base = VW::LEARNER::as_singleline_no_throw(VW::LEARNER::make_base(*binary_reduction), success);
      if (!success) { return E_VW_PREDICT_ERR_INVALID_MODEL; }

      auto lc = cats_tree->learner_count();
      auto* cats_learner = VW::LEARNER::make_reduction_learner(
          std::move(cats_tree), base, VW::cats_tree::learn, VW::cats_tree::predict, "VW::cats::setup")
                               .set_params_per_weight((uint32_t)ceil_log_2(lc))  // depth of tree
                               .set_prediction_type(prediction_type_t::multiclass)
                               .set_label_type(label_type_t::simple)
                               .build();

      auto p_reduction = VW::make_unique<VW::continuous_action::get_pmf>();
      auto* c_base = VW::LEARNER::as_singleline_no_throw(VW::LEARNER::make_base(*cats_learner), success);
      if (!success) { return E_VW_PREDICT_ERR_INVALID_MODEL; }
      p_reduction->init(c_base, 0.0f);

      auto* get_pmf_red = VW::LEARNER::make_reduction_learner<VW::continuous_action::get_pmf, example>(
          std::move(p_reduction), c_base, VW::continuous_action::learn, VW::continuous_action::predict, "setup_get_pmf")
                              .set_params_per_weight(1)
                              .set_prediction_type(prediction_type_t::pdf)
                              .build();

      auto p_base = VW::LEARNER::as_singleline_no_throw(VW::LEARNER::make_base(*get_pmf_red), success);
      if (!success) { return E_VW_PREDICT_ERR_INVALID_MODEL; }
      pmf_to_pdf->_p_base = p_base;

      auto get_pmf_to_pdf_red = VW::LEARNER::make_reduction_learner(
          std::move(pmf_to_pdf), p_base, VW::pmf_to_pdf::learn, VW::pmf_to_pdf::predict, "setup_pmf_to_pdf")
                                    .set_params_per_weight(1)
                                    .set_prediction_type(prediction_type_t::pdf)
                                    .build();

      auto pmftopdf_base = VW::LEARNER::as_singleline_no_throw(VW::LEARNER::make_base(*get_pmf_to_pdf_red), success);
      if (!success) { return E_VW_PREDICT_ERR_INVALID_MODEL; }

      auto cb_explore_pdf = VW::make_unique<VW::continuous_action::cb_explore_pdf>();
      cb_explore_pdf->init(pmftopdf_base);
      cb_explore_pdf->epsilon = _epsilon;
      cb_explore_pdf->min_value = _min_value;
      cb_explore_pdf->max_value = _max_value;
      cb_explore_pdf->first_only = false;

      auto* cb_explore_pdf_learner =
          VW::LEARNER::make_reduction_learner<VW::continuous_action::cb_explore_pdf, example>(std::move(cb_explore_pdf),
              pmftopdf_base, VW::continuous_action::learn, VW::continuous_action::predict, "setup_cb_explore_pdf")
              .set_params_per_weight(1)
              .set_prediction_type(prediction_type_t::pdf)
              .build();

      auto* cb_explore_pdf_learner_base =
          VW::LEARNER::as_singleline_no_throw(VW::LEARNER::make_base(*cb_explore_pdf_learner), success);
      if (!success) { return E_VW_PREDICT_ERR_INVALID_MODEL; }
      auto cats_pdf = VW::make_unique<VW::continuous_action::cats_pdf::cats_pdf>(
          cb_explore_pdf_learner_base, /*always predict*/ true);

      auto* cats_pdf_learner = VW::LEARNER::make_reduction_learner(std::move(cats_pdf), cb_explore_pdf_learner_base,
          VW::continuous_action::cats_pdf::learn, VW::continuous_action::cats_pdf::predict, "setup_cats_pdf")
                                   .set_params_per_weight(1)
                                   .set_prediction_type(prediction_type_t::pdf)
                                   .set_learn_returns_prediction(true)
                                   .build();

      auto* cats_pdf_learner_base =
          VW::LEARNER::as_singleline_no_throw(VW::LEARNER::make_base(*cats_pdf_learner), success);
      if (!success) { return E_VW_PREDICT_ERR_INVALID_MODEL; }

      auto sample_pdf = VW::make_unique<VW::continuous_action::sample_pdf>();
      sample_pdf->init(cats_pdf_learner_base, &random_seed);

      auto* sample_pdf_learner =
          VW::LEARNER::make_reduction_learner<VW::continuous_action::sample_pdf, example>(std::move(sample_pdf),
              cats_pdf_learner_base, VW::continuous_action::learn, VW::continuous_action::predict, "setup_sample_pdf")
              .set_params_per_weight(1)
              .set_prediction_type(prediction_type_t::action_pdf_value)
              .build();

      auto* sample_pdf_learner_base =
          VW::LEARNER::as_singleline_no_throw(VW::LEARNER::make_base(*sample_pdf_learner), success);
      if (!success) { return E_VW_PREDICT_ERR_INVALID_MODEL; }
      auto cats = VW::make_unique<VW::continuous_action::cats::cats>(
          sample_pdf_learner_base, _cats, _bandwidth, _min_value, _max_value, /*bandwidth was supplied*/ true);

      _cats_learner = VW::LEARNER::make_reduction_learner(std::move(cats), sample_pdf_learner_base,
          VW::continuous_action::cats::learn, VW::continuous_action::cats::predict, "setup_cats")
                          .set_params_per_weight(1)
                          .set_prediction_type(prediction_type_t::action_pdf_value)
                          .build();
    }

    // TODO: check that permutations is not enabled (or parse it)

    _model_loaded = true;

    return S_VW_PREDICT_OK;
  }

  /**
   * @brief True if the model describes a contextual bandit (cb) model using action dependent features (afd)
   *
   * @return true True if contextual bandit predict method can be used.
   * @return false False if contextual bandit predict method cannot be used.
   */
  bool is_cb_explore_adf() { return _command_line_arguments.find("--cb_explore_adf") != std::string::npos; }

  /**
   * @brief True if the model describes a cost sensitive one-against-all (csoaa). This is also true for cb_explore_adf
   * models, as they are reduced to csoaa.
   *
   * @return true True if csoaa predict method can be used.
   * @return false False if csoaa predict method cannot be used.
   */
  bool is_csoaa_ldf() { return _command_line_arguments.find("--csoaa_ldf") != std::string::npos; }

  /**
   * @brief Predicts a score (as in regression) for the provided example.
   *
   * Regular regression with support for constant feature (bias term) and interactions
   *
   * @param ex The example to get the prediction for.
   * @param score The output score produced by the model.
   * @return int Returns 0 (S_VW_PREDICT_OK) if succesful, otherwise one of the error codes (see E_VW_PREDICT_ERR_*).
   */
  int predict(example_predict& ex, float& score)
  {
    if (!_model_loaded) return E_VW_PREDICT_ERR_NO_MODEL_LOADED;

    std::unique_ptr<namespace_copy_guard> ns_copy_guard;

    if (!_no_constant)
    {
      // add constant feature
      ns_copy_guard = std::unique_ptr<namespace_copy_guard>(new namespace_copy_guard(ex, constant_namespace));
      ns_copy_guard->feature_push_back(1.f, (constant << _stride_shift) + ex.ft_offset);
    }

    if (_predict_info->_contains_wildcard)
    {
      // permutations is not supported by slim so we can just use combinations!
      _predict_info->_generate_interactions.template update_interactions_if_new_namespace_seen<
          INTERACTIONS::generate_namespace_combinations_with_repetition, false>(
          _predict_info->_interactions, ex.indices);
      score = GD::inline_predict<W>(*_predict_info->_weights, false, _predict_info->_ignore_linear,
          _predict_info->_generate_interactions.generated_interactions,
          /* permutations */ false, ex);
    }
    else
    {
      score = GD::inline_predict<W>(*_predict_info->_weights, false, _predict_info->_ignore_linear,
          _predict_info->_interactions, /* permutations */ false, ex);
    }
    return S_VW_PREDICT_OK;
  }

  int predict_cats(example* ex, float& action, float& pdf_value)
  {
    if (!_model_loaded) return E_VW_PREDICT_ERR_NO_MODEL_LOADED;

    std::unique_ptr<namespace_copy_guard> ns_copy_guard;

    // apply manual stride shifts
    std::vector<std::unique_ptr<stride_shift_guard>> stride_shift_guards;
    stride_shift_guards.push_back(VW::make_unique<stride_shift_guard>(*ex, _stride_shift));

    if (!_no_constant)
    {
      ns_copy_guard = std::unique_ptr<namespace_copy_guard>(new namespace_copy_guard(*ex, constant_namespace));
      ns_copy_guard->feature_push_back(1.f, (constant << _stride_shift) + ex->ft_offset);
    }

    _cats_learner->predict(*ex);
    action = ex->pred.pdf_value.action;
    pdf_value = ex->pred.pdf_value.pdf_value;

    return S_VW_PREDICT_OK;
  }

  // multiclass classification
  int predict(example_predict& shared, example_predict* actions, size_t num_actions, std::vector<float>& out_scores)
  {
    if (!_model_loaded) return E_VW_PREDICT_ERR_NO_MODEL_LOADED;

    if (!is_csoaa_ldf()) return E_VW_PREDICT_ERR_NO_A_CSOAA_MODEL;

    out_scores.resize(num_actions);

    example_predict* action = actions;
    for (size_t i = 0; i < num_actions; i++, action++)
    {
      std::vector<std::unique_ptr<namespace_copy_guard>> ns_copy_guards;

      // shared feature copying
      for (auto ns : shared.indices)
      {
        // insert namespace
        auto ns_copy_guard = std::unique_ptr<namespace_copy_guard>(new namespace_copy_guard(*action, ns));

        // copy features
        for (auto fs : shared.feature_space[ns]) ns_copy_guard->feature_push_back(fs.value(), fs.index());

        // keep guard around
        ns_copy_guards.push_back(std::move(ns_copy_guard));
      }

      RETURN_ON_FAIL(predict(*action, out_scores[i]));
    }

    return S_VW_PREDICT_OK;
  }

  int predict(const char* event_id, example_predict& shared, example_predict* actions, size_t num_actions,
      std::vector<float>& pdf, std::vector<int>& ranking)
  {
    if (!_model_loaded) return E_VW_PREDICT_ERR_NO_MODEL_LOADED;

    if (!is_cb_explore_adf()) return E_VW_PREDICT_ERR_NOT_A_CB_MODEL;

    std::vector<float> scores;

    // add exploration
    pdf.resize(num_actions);
    ranking.resize(num_actions);

    switch (_exploration)
    {
      case vw_predict_exploration::epsilon_greedy:
      {
        // get the prediction
        RETURN_ON_FAIL(predict(shared, actions, num_actions, scores));

        // generate exploration distribution
        // model is trained against cost -> minimum is better
        auto top_action_iterator = std::min_element(std::begin(scores), std::end(scores));
        uint32_t top_action = (uint32_t)(top_action_iterator - std::begin(scores));

        RETURN_EXPLORATION_ON_FAIL(
            exploration::generate_epsilon_greedy(_epsilon, top_action, std::begin(pdf), std::end(pdf)));
        break;
      }
      case vw_predict_exploration::softmax:
      {
        // get the prediction
        RETURN_ON_FAIL(predict(shared, actions, num_actions, scores));

        // generate exploration distribution
        RETURN_EXPLORATION_ON_FAIL(exploration::generate_softmax(
            _lambda, std::begin(scores), std::end(scores), std::begin(pdf), std::end(pdf)));
        break;
      }
      case vw_predict_exploration::bag:
      {
        std::vector<uint32_t> top_actions(num_actions);

        // apply stride shifts
        std::vector<std::unique_ptr<stride_shift_guard>> stride_shift_guards;
        stride_shift_guards.push_back(
            std::unique_ptr<stride_shift_guard>(new stride_shift_guard(shared, _stride_shift)));
        example_predict* actions_end = actions + num_actions;
        for (example_predict* action = actions; action != actions_end; ++action)
          stride_shift_guards.push_back(
              std::unique_ptr<stride_shift_guard>(new stride_shift_guard(*action, _stride_shift)));

        for (size_t i = 0; i < _bag_size; i++)
        {
          std::vector<std::unique_ptr<feature_offset_guard>> feature_offset_guards;
          for (example_predict* action = actions; action != actions_end; ++action)
            feature_offset_guards.push_back(
                std::unique_ptr<feature_offset_guard>(new feature_offset_guard(*action, i)));

          RETURN_ON_FAIL(predict(shared, actions, num_actions, scores));

          auto top_action_iterator = std::min_element(std::begin(scores), std::end(scores));
          uint32_t top_action = (uint32_t)(top_action_iterator - std::begin(scores));

          top_actions[top_action]++;
        }

        // generate exploration distribution
        RETURN_EXPLORATION_ON_FAIL(
            exploration::generate_bag(std::begin(top_actions), std::end(top_actions), std::begin(pdf), std::end(pdf)));

        if (_minimum_epsilon > 0)
          RETURN_EXPLORATION_ON_FAIL(
              exploration::enforce_minimum_probability(_minimum_epsilon, true, std::begin(pdf), std::end(pdf)));

        break;
      }
      default:
        return E_VW_PREDICT_ERR_NOT_A_CB_MODEL;
    }

    RETURN_EXPLORATION_ON_FAIL(sort_by_scores(
        std::begin(pdf), std::end(pdf), std::begin(scores), std::end(scores), std::begin(ranking), std::end(ranking)));

    // Sample from the pdf
    uint32_t chosen_action_idx;
    RETURN_EXPLORATION_ON_FAIL(
        exploration::sample_after_normalizing(event_id, std::begin(pdf), std::end(pdf), chosen_action_idx));

    // Swap top element with chosen one (unless chosen is the top)
    if (chosen_action_idx != 0)
    {
      std::iter_swap(std::begin(ranking), std::begin(ranking) + chosen_action_idx);
      std::iter_swap(std::begin(pdf), std::begin(pdf) + chosen_action_idx);
    }

    return S_VW_PREDICT_OK;
  }

  template <typename PdfIt, typename InputScoreIt, typename OutputIt>
  static int sort_by_scores(PdfIt pdf_first, PdfIt pdf_last, InputScoreIt scores_first, InputScoreIt scores_last,
      OutputIt ranking_begin, OutputIt ranking_last)
  {
    const size_t pdf_size = pdf_last - pdf_first;
    const size_t ranking_size = ranking_last - ranking_begin;

    if (pdf_size != ranking_size) return E_EXPLORATION_PMF_RANKING_SIZE_MISMATCH;

    // Initialize ranking with actions 0,1,2,3 ...
    std::iota(ranking_begin, ranking_last, 0);

    // Pdf starts out in the same order as ranking.  Ranking and pdf should been sorted
    // in the order specified by scores.
    using CP = internal::collection_pair_iterator<OutputIt, PdfIt>;
    using Iter = typename CP::Iter;
    using Loc = typename CP::Loc;
    const Iter begin_coll(ranking_begin, pdf_first);
    const Iter end_coll(ranking_last, pdf_last);
    std::sort(begin_coll, end_coll, [&scores_first](const Loc& l, const Loc& r) {
      return scores_first[size_t(l._val1)] < scores_first[size_t(r._val1)];
    });

    return S_EXPLORATION_OK;
  }

  uint32_t feature_index_num_bits() { return _num_bits; }
};
}  // namespace vw_slim
