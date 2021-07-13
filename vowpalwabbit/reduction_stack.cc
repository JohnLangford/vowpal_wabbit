#include "reduction_stack.h"

#include "global_data.h"  // to get vw struct
#include "cached_learner.h"
#include "learner.h"
#include "options.h"
#include "reductions_fwd.h"

// reductions / setup functions
#include "gd.h"
#include "sender.h"
#include "nn.h"
#include "cbify.h"
#include "oaa.h"
#include "boosting.h"
#include "multilabel_oaa.h"
#include "bs.h"
#include "topk.h"
#include "ect.h"
#include "csoaa.h"
#include "cb_algs.h"
#include "cb_adf.h"
#include "cb_to_cb_adf.h"
#include "cb_dro.h"
#include "cb_explore.h"
#include "cb_explore_adf_bag.h"
#include "cb_explore_adf_cover.h"
#include "cb_explore_adf_first.h"
#include "cb_explore_adf_greedy.h"
#include "cb_explore_adf_regcb.h"
#include "cb_explore_adf_squarecb.h"
#include "cb_explore_adf_synthcover.h"
#include "cb_explore_adf_rnd.h"
#include "cb_explore_adf_softmax.h"
#include "slates.h"
#include "generate_interactions.h"
#include "mwt.h"
#include "confidence.h"
#include "scorer.h"
#include "expreplay.h"
#include "search.h"
#include "bfgs.h"
#include "lda_core.h"
#include "noop.h"
#include "print.h"
#include "gd_mf.h"
#include "mf.h"
#include "ftrl.h"
#include "svrg.h"
#include "binary.h"
#include "lrq.h"
#include "lrqfa.h"
#include "autolink.h"
#include "log_multi.h"
#include "recall_tree.h"
#include "memory_tree.h"
#include "plt.h"
#include "stagewise_poly.h"
#include "active.h"
#include "active_cover.h"
#include "cs_active.h"
#include "kernel_svm.h"
#include "interact.h"
#include "OjaNewton.h"
#include "audit_regressor.h"
#include "marginal.h"
#include "metrics.h"
#include "explore_eval.h"
#include "baseline.h"
#include "classweight.h"
#include "cb_sample.h"
#include "warm_cb.h"
#include "shared_feature_merger.h"
#include "cbzo.h"
// #include "cntk.h"
#include "cats.h"
#include "cats_pdf.h"
#include "cb_explore_pdf.h"
#include "offset_tree.h"
#include "cats_tree.h"
#include "get_pmf.h"
#include "pmf_to_pdf.h"
#include "sample_pdf.h"
#include "kskip_ngram_transformer.h"

void register_reductions(std::vector<reduction_setup_fn>& reductions,
    std::vector<std::tuple<std::string, reduction_setup_fn>>& reduction_stack)
{
  VW::cached_learner null_ptr_learner;

  std::map<reduction_setup_fn, std::string> allowlist = {{GD::setup, "gd"}, {ftrl_setup, "ftrl"},
      {scorer_setup, "scorer"}, {CSOAA::csldf_setup, "csoaa_ldf"},
      {VW::cb_explore_adf::greedy::setup, "cb_explore_adf_greedy"},
      {VW::cb_explore_adf::regcb::setup, "cb_explore_adf_regcb"},
      {VW::shared_feature_merger::shared_feature_merger_setup, "shared_feature_merger"},
      {generate_interactions_setup, "generate_interactions"}};

  auto name_extractor = VW::config::options_name_extractor();
  vw dummy_all;

  for (auto setup_fn : reductions)
  {
    if (allowlist.count(setup_fn)) { reduction_stack.push_back(std::make_tuple(allowlist[setup_fn], setup_fn)); }
    else
    {
      auto base = setup_fn(null_ptr_learner, name_extractor, dummy_all);

      if (base == nullptr)
        reduction_stack.push_back(std::make_tuple(name_extractor.generated_name, setup_fn));
      else
        THROW("fatal: under register_reduction() all setup functions must return nullptr");
    }
  }
}

void prepare_reductions(std::vector<std::tuple<std::string, reduction_setup_fn>>& reduction_stack)
{
  std::vector<reduction_setup_fn> reductions;

  // Base algorithms
  reductions.push_back(GD::setup);
  reductions.push_back(kernel_svm_setup);
  reductions.push_back(ftrl_setup);
  reductions.push_back(svrg_setup);
  reductions.push_back(sender_setup);
  reductions.push_back(gd_mf_setup);
  reductions.push_back(print_setup);
  reductions.push_back(noop_setup);
  reductions.push_back(lda_setup);
  reductions.push_back(bfgs_setup);
  reductions.push_back(OjaNewton_setup);
  // reductions.push_back(VW_CNTK::setup);

  reductions.push_back(mf_setup);

  reductions.push_back(generate_interactions_setup);

  // Score Users
  reductions.push_back(baseline_setup);
  reductions.push_back(ExpReplay::expreplay_setup<'b', simple_label_parser>);
  reductions.push_back(active_setup);
  reductions.push_back(active_cover_setup);
  reductions.push_back(confidence_setup);
  reductions.push_back(nn_setup);
  reductions.push_back(marginal_setup);
  reductions.push_back(autolink_setup);
  reductions.push_back(lrq_setup);
  reductions.push_back(lrqfa_setup);
  reductions.push_back(stagewise_poly_setup);
  reductions.push_back(scorer_setup);
  reductions.push_back(VW::cbzo::setup);

  // Reductions
  reductions.push_back(bs_setup);
  reductions.push_back(VW::binary::binary_setup);

  reductions.push_back(ExpReplay::expreplay_setup<'m', MULTICLASS::mc_label>);
  reductions.push_back(topk_setup);
  reductions.push_back(oaa_setup);
  reductions.push_back(boosting_setup);
  reductions.push_back(ect_setup);
  reductions.push_back(log_multi_setup);
  reductions.push_back(recall_tree_setup);
  reductions.push_back(memory_tree_setup);
  reductions.push_back(classweight_setup);
  reductions.push_back(multilabel_oaa_setup);
  reductions.push_back(plt_setup);

  reductions.push_back(cs_active_setup);
  reductions.push_back(CSOAA::csoaa_setup);
  reductions.push_back(interact_setup);
  reductions.push_back(CSOAA::csldf_setup);
  reductions.push_back(cb_algs_setup);
  reductions.push_back(cb_adf_setup);
  reductions.push_back(mwt_setup);
  reductions.push_back(VW::cats_tree::setup);
  reductions.push_back(cb_explore_setup);
  reductions.push_back(VW::cb_explore_adf::greedy::setup);
  reductions.push_back(VW::cb_explore_adf::softmax::setup);
  reductions.push_back(VW::cb_explore_adf::rnd::setup);
  reductions.push_back(VW::cb_explore_adf::regcb::setup);
  reductions.push_back(VW::cb_explore_adf::squarecb::setup);
  reductions.push_back(VW::cb_explore_adf::synthcover::setup);
  reductions.push_back(VW::cb_explore_adf::first::setup);
  reductions.push_back(VW::cb_explore_adf::cover::setup);
  reductions.push_back(VW::cb_explore_adf::bag::setup);
  reductions.push_back(cb_dro_setup);
  reductions.push_back(cb_sample_setup);
  reductions.push_back(explore_eval_setup);
  reductions.push_back(VW::shared_feature_merger::shared_feature_merger_setup);
  reductions.push_back(CCB::ccb_explore_adf_setup);
  reductions.push_back(VW::slates::slates_setup);
  // cbify/warm_cb can generate multi-examples. Merge shared features after them
  reductions.push_back(warm_cb_setup);
  reductions.push_back(VW::continuous_action::get_pmf_setup);
  reductions.push_back(VW::pmf_to_pdf::setup);
  reductions.push_back(VW::continuous_action::cb_explore_pdf_setup);
  reductions.push_back(VW::continuous_action::cats_pdf::setup);
  reductions.push_back(VW::continuous_action::sample_pdf_setup);
  reductions.push_back(VW::continuous_action::cats::setup);
  reductions.push_back(cbify_setup);
  reductions.push_back(cbifyldf_setup);
  reductions.push_back(cb_to_cb_adf_setup);
  reductions.push_back(VW::offset_tree::setup);
  reductions.push_back(ExpReplay::expreplay_setup<'c', COST_SENSITIVE::cs_label>);
  reductions.push_back(Search::setup);
  reductions.push_back(audit_regressor_setup);
  reductions.push_back(VW::metrics::metrics_setup);

  register_reductions(reductions, reduction_stack);
}

namespace VW
{
default_reduction_stack_setup::default_reduction_stack_setup(vw& all)
{
  // push all reduction functions into the stack
  prepare_reductions(reduction_stack);
  // populate setup_fn -> name map to be used to lookup names in setup_base
  all.build_setupfn_name_dict(reduction_stack);
}

void default_reduction_stack_setup::print_enabled_reductions(vw& all)
{
  *(all.trace_message) << "Enabled reductions: ";

  if (!enabled_reductions.empty())
  {
    const char* const delim = ", ";
    std::ostringstream imploded;
    std::copy(
        enabled_reductions.begin(), enabled_reductions.end() - 1, std::ostream_iterator<std::string>(imploded, delim));

    *(all.trace_message) << imploded.str() << enabled_reductions.back();
  }

  *(all.trace_message) << std::endl;
}

// this function consumes all the reduction_stack until it's able to construct a base_learner
// same signature/code as the old setup_base(...) from parse_args.cc
VW::LEARNER::base_learner* default_reduction_stack_setup::operator()(VW::config::options_i& options, vw& all)
{
  if (!reduction_stack.empty())
  {
    auto func_map = reduction_stack.back();
    reduction_setup_fn setup_func = std::get<1>(func_map);
    std::string setup_func_name = std::get<0>(func_map);
    reduction_stack.pop_back();

    // 'hacky' way of keeping track of the option group created by the setup_func about to be created
    options.tint(setup_func_name);
    auto base = setup_func(*this, options, all);
    options.reset_tint();

    // returning nullptr means that setup_func (any reduction) was not 'enabled' but
    // only added their respective command args and did not add itself into the
    // chain of learners, therefore we call into setup_base again
    if (base == nullptr) { return this->operator()(options, all); }
    else
    {
      enabled_reductions.push_back(setup_func_name);
      reduction_stack.clear();
      return base;
    }
  }

  return nullptr;
}
}  // namespace VW
