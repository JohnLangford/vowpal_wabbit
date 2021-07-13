// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include "example.h"
#include "unique_sort.h"
#include <algorithm>

void unique_features(features& fs, int max)
{
  if (fs.indicies.empty()) return;

  auto range = fs.audit_range();
  auto last_index = range.begin();
  auto end = max > 0 ? range.begin() + std::min(fs.size(), static_cast<size_t>(max)) : range.end();

  for (auto i = ++range.begin(); i != end; ++i)
    if (i.index() != last_index.index())
      if (i != ++last_index)
      {
        last_index.value() = i.value();
        last_index.index() = i.index();
        if (!fs.space_names.empty()) *last_index.audit() = *i.audit();
      }

  ++last_index;
  fs.truncate_to(last_index);
}

void unique_sort_features(uint64_t parse_mask, example* ae)
{
  for (auto& bucket : *ae)
  {
    for (auto& fs : bucket)
    {
      if (fs.feats.sort(parse_mask)) { unique_features(fs.feats); }
    }
  }

  ae->sorted = true;
}
