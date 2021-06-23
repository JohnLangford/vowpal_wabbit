#include "opts.h"

#include <cstdlib>
#include <cctype>

namespace vw_slim
{
template <class T>
void find_opt(std::string const& command_line_args, std::string arg_name, std::vector<T>& out_values)
{
  // append space to search for '--quadratic '
  arg_name += ' ';

  // support -q ab -v -q cd
  for (size_t start = 0; start < command_line_args.size();)
  {
    auto idx = command_line_args.find(arg_name, start);
    if (idx == std::string::npos) return;  // no more occurences found, exit

    auto idx_after_arg = idx + arg_name.size();

    // skip all white space
    for (; idx_after_arg < command_line_args.size() && std::isspace(command_line_args[idx_after_arg]); ++idx_after_arg)
      ;

    if (idx_after_arg == command_line_args.size()) return;

    if (command_line_args[idx_after_arg] == '-' &&
        // make sure we allow -5.2 (negative numbers)
        !(idx_after_arg + 1 < command_line_args.size() &&
            (command_line_args[idx_after_arg + 1] >= '0' && command_line_args[idx_after_arg + 1] <= '9')))
    {
      start = idx_after_arg;
      continue;  // next option found
    }

    // find next non-white space character
    auto idx_after_value = idx_after_arg;
    for (; idx_after_value < command_line_args.size() && !std::isspace(command_line_args[idx_after_value]);
         ++idx_after_value)
      ;

    auto value_size = idx_after_value - idx_after_arg;
    if (value_size > 0)
    {
      std::string args = command_line_args.substr(idx_after_arg, value_size);
      out_values.emplace_back(args.begin(), args.end());
    }
    start = idx_after_arg + 1;
  }
}

template void find_opt<std::string>(
    std::string const& command_line_args, std::string arg_name, std::vector<std::string>& out_values);

template void find_opt<std::vector<namespace_index>>(
    std::string const& command_line_args, std::string arg_name, std::vector<std::vector<namespace_index>>& out_values);

std::vector<std::string> find_opt(std::string const& command_line_args, std::string arg_name)
{
  std::vector<std::string> values;
  find_opt(command_line_args, arg_name, values);
  return values;
}

template <typename T, typename S, S (*F)(const char*)>
bool find_opt_parse(std::string const& command_line_args, std::string arg_name, T& value)
{
  std::vector<std::string> opts = find_opt(command_line_args, arg_name);

  if (opts.size() != 1) return false;

  value = (T)F(opts[0].c_str());

  return true;
}

bool find_opt_float(std::string const& command_line_args, std::string arg_name, float& value)
{
  return find_opt_parse<float, double, atof>(command_line_args, arg_name, value);
}

bool find_opt_int(std::string const& command_line_args, std::string arg_name, int& value)
{
  return find_opt_parse<int, int, atoi>(command_line_args, arg_name, value);
}

bool find_opt_uint64_t(std::string const& command_line_args, std::string arg_name, uint64_t& value)
{
  return find_opt_parse<uint64_t, int, atoi>(command_line_args, arg_name, value);
}

}  // namespace vw_slim