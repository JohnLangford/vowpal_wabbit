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
#include "opts.h"
#include "interactions.h"

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
  uint64_t _ns;

public:
  namespace_copy_guard(example_predict& ex, uint64_t ns, unsigned const char index);
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

/**
 * @brief Vowpal Wabbit slim predictor. Supports: regression, multi-class classification and contextual bandits.
 */
template <typename W>
class vw_predict
{
  std::unique_ptr<W> _weights;
  std::string _id;
  std::string _version;
  std::string _command_line_arguments;
  std::vector<std::vector<namespace_index>> _interactions;
  INTERACTIONS::interactions_generator _generate_interactions;
  bool _contains_wildcard;
  std::array<bool, NUM_NAMESPACES> _ignore_linear;
  bool _no_constant;

  vw_predict_exploration _exploration;
  float _minimum_epsilon;
  float _epsilon;
  float _lambda;
  int _bag_size;
  uint32_t _num_bits;

  uint32_t _stride_shift;
  bool _model_loaded;

public:
  vw_predict() : _model_loaded(false), _contains_wildcard(false) {}

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

    // required for inline_predict
    _ignore_linear.fill(false);

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

    _interactions.clear();
    find_opt(_command_line_arguments, "-q", _interactions);
    find_opt(_command_line_arguments, "--quadratic", _interactions);
    find_opt(_command_line_arguments, "--cubic", _interactions);
    find_opt(_command_line_arguments, "--interactions", _interactions);

    // VW performs the following transformation as a side-effect of looking for duplicates.
    // This affects how interaction hashes are generated.
    std::vector<std::vector<namespace_index>> vec_sorted;
    for (auto& interaction : _interactions) { std::sort(std::begin(interaction), std::end(interaction)); }

    for (const auto& inter : _interactions)
    {
      if (INTERACTIONS::contains_wildcard(inter))
      {
        _contains_wildcard = true;
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

    RETURN_ON_FAIL(mp.read_weights<W>(_weights, _num_bits, _stride_shift));

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
      ns_copy_guard =
          std::unique_ptr<namespace_copy_guard>(new namespace_copy_guard(ex, constant_namespace, constant_namespace));
      ns_copy_guard->feature_push_back(1.f, (constant << _stride_shift) + ex.ft_offset);
    }

    if (_contains_wildcard)
    {
      // permutations is not supported by slim so we can just use combinations!
      _generate_interactions.update_interactions_if_new_namespace_seen<
          INTERACTIONS::generate_namespace_combinations_with_repetition, false>(
          _interactions, ex.feature_space.get_indices());
      score = GD::inline_predict<W>(*_weights, false, _ignore_linear, _generate_interactions.generated_interactions,
          /* permutations */ false, ex);
    }
    else
    {
      score = GD::inline_predict<W>(*_weights, false, _ignore_linear, _interactions, /* permutations */ false, ex);
    }
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
      for (auto it = shared.feature_space.begin(); it != shared.feature_space.end(); ++it)
      {
        // insert namespace
        auto ns_copy_guard =
            std::unique_ptr<namespace_copy_guard>(new namespace_copy_guard(*action, it->hash, it->index));

        // copy features
        for (auto fs : *shared.feature_space.get_or_null(it->hash))
          ns_copy_guard->feature_push_back(fs.value(), fs->index);

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
