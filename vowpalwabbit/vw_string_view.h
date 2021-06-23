#pragma once

#include "hashstring.h"

#include <boost/version.hpp>

#ifdef VW_SLIM_ON

namsepace VW { using string_view = basic_string_view<char>; }
#else
#  if BOOST_VERSION < 106100
#    include <boost/utility/string_ref.hpp>
namespace VW
{
using string_view = boost::string_ref;
}
#  else
#    include <boost/utility/string_view.hpp>
namespace VW
{
using string_view = boost::string_view;
}
#  endif
#endif

namespace std
{
// boost VW::string_view hashing isn't available until 1.69. Implement our own for now
template <>
struct hash<VW::string_view>
{
  size_t operator()(const VW::string_view& s) const { return hashstring(s.begin(), s.length(), 0); }
};
}  // namespace std
