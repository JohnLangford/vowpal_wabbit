// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "learner_no_throw.h"
#include <sstream>

// re-implement for slim TODO: fix
void return_simple_example(vw& all, void*, example& ec) {}
namespace VW
{
namespace LEARNER
{
float recur_sensitivity(void*, base_learner& base, example& ec) { return base.sensitivity(ec); }
}
std::string debug_depth_indent_string(const int32_t depth)
{
  constexpr const char* indent_str = "- ";
  constexpr const char* space_str = "  ";

  if (depth == 0) return indent_str;

  std::stringstream str_stream;
  for (int32_t i = 0; i < depth - 1; i++) { str_stream << space_str; }
  str_stream << indent_str;
  return str_stream.str();
}
std::string debug_depth_indent_string(example& ec)
{
  return debug_depth_indent_string(ec._debug_current_reduction_depth);
}
std::string debug_depth_indent_string(const multi_ex& ec) { return debug_depth_indent_string(*ec[0]); }
}