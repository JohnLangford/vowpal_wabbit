// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "example_predict.h"

#include <sstream>

VW_WARNING_STATE_PUSH
VW_WARNING_DISABLE_DEPRECATED_USAGE
safe_example_predict::safe_example_predict()
{
  ft_offset = 0;
}

VW_WARNING_STATE_POP
void safe_example_predict::clear() { feature_space.clear(); }

std::string features_to_string(const example_predict& ec)
{
  std::stringstream strstream;
  strstream << "[off=" << ec.ft_offset << "]";
  // TODO dont const cast
  for (auto& bucket : const_cast<example_predict&>(ec))
  {
    for (auto& f : bucket)
    {
      auto ind_iter = f.feats.indicies.cbegin();
      auto val_iter = f.feats.values.cbegin();
      for (; ind_iter != f.feats.indicies.cend(); ++ind_iter, ++val_iter)
      {
        strstream << "[h=" << *ind_iter << ","
                  << "v=" << *val_iter << "]";
      }
    }
  }

  return strstream.str();
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
