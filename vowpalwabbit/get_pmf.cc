#include "get_pmf.h"
#include "err_constants.h"
#include "api_status.h"
#include "debug_log.h"
#include "parse_args.h"

// Aliases
using LEARNER::single_learner;
using std::endl;
using VW::cb_continuous::continuous_label;
using VW::cb_continuous::continuous_label_elm;
using VW::config::make_option;
using VW::config::option_group_definition;
using VW::config::options_i;

// Enable/Disable indented debug statements
VW_DEBUG_ENABLE(false)

namespace VW
{
namespace continuous_action
{
////////////////////////////////////////////////////
// BEGIN sample_pdf reduction and reduction methods
struct get_pmf
{
  int learn(example& ec, api_status* status);
  int predict(example& ec, api_status* status);

  void init(single_learner* p_base, float epsilon);

 private:
  single_learner* _base = nullptr;
  float _epsilon;
};

int get_pmf::learn(example& ec, api_status* status)
{
  _base->learn(ec);
  return error_code::success;
}

int get_pmf::predict(example& ec, api_status* status)
{
  uint32_t base_prediction;

  {  // predict & restore prediction
    restore_prediction restore(ec);
    _base->predict(ec);
    base_prediction = ec.pred.multiclass - 1;
  }

  // Assume ec.pred.a_s allocated by the caller (probably pmf_to_pdf);
  ec.pred.a_s.clear();
  ec.pred.a_s.push_back({base_prediction, 1.0f});

  return error_code::success;
}

void get_pmf::init(single_learner* p_base, float epsilon) {
  _base = p_base;
  _epsilon = epsilon;
}

// Free function to tie function pointers to reduction class methods
template <bool is_learn>
void predict_or_learn(get_pmf& reduction, single_learner&, example& ec)
{
  api_status status;
  if (is_learn)
    reduction.learn(ec, &status);
  else
    reduction.predict(ec, &status);

  if (status.get_error_code() != error_code::success)
  {
    VW_DBG(ec) << status.get_error_msg() << endl;
  }
}

// END sample_pdf reduction and reduction methods
////////////////////////////////////////////////////

// Setup reduction in stack
LEARNER::base_learner* get_pdf_setup(config::options_i& options, vw& all)
{
  option_group_definition new_options("Continuous actions");
  bool invoked = false;
  float epsilon = 0.0f;
  new_options.add(make_option("get_pmf", invoked).keep().help("Convert a single multiclass prediction to a pmf"));

  options.add_and_parse(new_options);

  // If reduction was not invoked, don't add anything
  // to the reduction stack;
  if (!options.was_supplied("get_pmf"))
    return nullptr;

  LEARNER::base_learner* p_base = setup_base(options, all);
  auto p_reduction = scoped_calloc_or_throw<get_pmf>();
  p_reduction->init(as_singleline(p_base), epsilon);

  LEARNER::learner<get_pmf, example>& l = init_learner(
      p_reduction, as_singleline(p_base), predict_or_learn<true>, predict_or_learn<false>, 1, prediction_type::pdf);

  return make_base(l);
}
}  // namespace continuous_action
}  // namespace VW