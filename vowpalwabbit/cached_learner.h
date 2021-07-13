#pragma once

#include "global_data.h"  // to get vw struct
#include "options.h"      // to get options_i

namespace VW
{
struct cached_learner : public setup_base_fn
{
  VW::LEARNER::base_learner* operator()(VW::config::options_i&, vw&) override { return _cached; }

  operator bool() const { return !(_cached == nullptr); }

  cached_learner(VW::LEARNER::base_learner* learner = nullptr) : _cached(learner) {}

private:
  VW::LEARNER::base_learner* _cached = nullptr;
};
}  // namespace VW
