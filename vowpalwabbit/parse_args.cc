// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cstdio>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>

#include "parse_regressor.h"
#include "parser.h"
#include "parse_primitives.h"
#include "vw.h"
#include "interactions.h"

#include "parse_args.h"
#include "reduction_stack.h"

#include "rand48.h"
#include "learner.h"
#include "parse_example.h"
#include "best_constant.h"
#include "vw_exception.h"
#include "accumulate.h"
#include "vw_validate.h"
#include "vw_allreduce.h"
#include "metrics.h"

#include "options.h"
#include "options_boost_po.h"
#include "options_serializer_boost_po.h"
#include "named_labels.h"

#include "io/io_adapter.h"
#include "io/custom_streambuf.h"
#include "io/owning_stream.h"
#include "io/logger.h"

#ifdef BUILD_EXTERNAL_PARSER
#  include "parse_example_binary.h"
#endif

using std::cout;
using std::endl;
using namespace VW::config;

namespace logger = VW::io::logger;

//
// Does std::string end with a certain substring?
//
bool ends_with(std::string const& fullString, std::string const& ending)
{
  if (fullString.length() > ending.length())
  { return (fullString.compare(fullString.length() - ending.length(), ending.length(), ending) == 0); }
  else
  {
    return false;
  }
}

uint64_t hash_file_contents(VW::io::reader* f)
{
  uint64_t v = 5289374183516789128;
  char buf[1024];
  while (true)
  {
    ssize_t n = f->read(buf, 1024);
    if (n <= 0) break;
    for (ssize_t i = 0; i < n; i++)
    {
      v *= 341789041;
      v += static_cast<uint64_t>(buf[i]);
    }
  }
  return v;
}

bool directory_exists(std::string path)
{
  struct stat info;
  if (stat(path.c_str(), &info) != 0)
    return false;
  else
    return (info.st_mode & S_IFDIR) > 0;
  //  boost::filesystem::path p(path);
  //  return boost::filesystem::exists(p) && boost::filesystem::is_directory(p);
}

std::string find_in_path(std::vector<std::string> paths, std::string fname)
{
#ifdef _WIN32
  std::string delimiter = "\\";
#else
  std::string delimiter = "/";
#endif
  for (std::string path : paths)
  {
    std::string full = ends_with(path, delimiter) ? (path + fname) : (path + delimiter + fname);
    std::ifstream f(full.c_str());
    if (f.good()) return full;
  }
  return "";
}

void parse_dictionary_argument(vw& all, const std::string& str)
{
  if (str.length() == 0) return;
  // expecting 'namespace:file', for instance 'w:foo.txt'
  // in the case of just 'foo.txt' it's applied to the default namespace

  char ns = ' ';
  VW::string_view s(str);
  if ((str.length() > 2) && (str[1] == ':'))
  {
    ns = str[0];
    s.remove_prefix(2);
  }

  std::string fname = find_in_path(all.dictionary_path, std::string(s));
  if (fname == "") THROW("error: cannot find dictionary '" << s << "' in path; try adding --dictionary_path");

  bool is_gzip = ends_with(fname, ".gz");
  std::unique_ptr<VW::io::reader> file_adapter;
  try
  {
    file_adapter = is_gzip ? VW::io::open_compressed_file_reader(fname) : VW::io::open_file_reader(fname);
  }
  catch (...)
  {
    THROW("error: cannot read dictionary from file '" << fname << "'"
                                                      << ", opening failed");
  }

  uint64_t fd_hash = hash_file_contents(file_adapter.get());

  if (!all.logger.quiet)
    *(all.trace_message) << "scanned dictionary '" << s << "' from '" << fname << "', hash=" << std::hex << fd_hash
                         << std::dec << endl;

  // see if we've already read this dictionary
  for (size_t id = 0; id < all.loaded_dictionaries.size(); id++)
  {
    if (all.loaded_dictionaries[id].file_hash == fd_hash)
    {
      all.namespace_dictionaries[static_cast<size_t>(ns)].push_back(all.loaded_dictionaries[id].dict);
      return;
    }
  }

  std::unique_ptr<VW::io::reader> fd;
  try
  {
    fd = VW::io::open_file_reader(fname);
  }
  catch (...)
  {
    THROW("error: cannot re-read dictionary from file '" << fname << "', opening failed");
  }
  auto map = std::make_shared<feature_dict>();
  // mimicing old v_hashmap behavior for load factor.
  // A smaller factor will generally use more memory but have faster access
  map->max_load_factor(0.25);
  example* ec = VW::alloc_examples(1);

  size_t def = static_cast<size_t>(' ');

  ssize_t size = 2048, pos, nread;
  char rc;
  char* buffer = calloc_or_throw<char>(size);
  do
  {
    pos = 0;
    do
    {
      nread = fd->read(&rc, 1);
      if ((rc != EOF) && (nread > 0)) buffer[pos++] = rc;
      if (pos >= size - 1)
      {
        size *= 2;
        const auto new_buffer = static_cast<char*>(realloc(buffer, size));
        if (new_buffer == nullptr)
        {
          free(buffer);
          VW::dealloc_examples(ec, 1);
          THROW("error: memory allocation failed in reading dictionary");
        }
        else
          buffer = new_buffer;
      }
    } while ((rc != EOF) && (rc != '\n') && (nread > 0));
    buffer[pos] = 0;

    // we now have a line in buffer
    char* c = buffer;
    while (*c == ' ' || *c == '\t') ++c;  // skip initial whitespace
    char* d = c;
    while (*d != ' ' && *d != '\t' && *d != '\n' && *d != '\0') ++d;  // gobble up initial word
    if (d == c) continue;                                             // no word
    if (*d != ' ' && *d != '\t') continue;                            // reached end of line
    std::string word(c, d - c);
    if (map->find(word) != map->end())  // don't overwrite old values!
    { continue; }
    d--;
    *d = '|';  // set up for parser::read_line
    VW::read_line(all, ec, d);
    // now we just need to grab stuff from the default namespace of ec!
    if (ec->feature_space[def].size() == 0) { continue; }
    map->emplace(word, VW::make_unique<features>(ec->feature_space[def]));

    // clear up ec
    ec->tag.clear();
    ec->indices.clear();
    for (size_t i = 0; i < 256; i++) { ec->feature_space[i].clear(); }
  } while ((rc != EOF) && (nread > 0));
  free(buffer);
  VW::dealloc_examples(ec, 1);

  if (!all.logger.quiet)
    *(all.trace_message) << "dictionary " << s << " contains " << map->size() << " item"
                         << (map->size() == 1 ? "" : "s") << endl;

  all.namespace_dictionaries[static_cast<size_t>(ns)].push_back(map);
  dictionary_info info = {s.to_string(), fd_hash, map};
  all.loaded_dictionaries.push_back(info);
}

void parse_affix_argument(vw& all, std::string str)
{
  if (str.length() == 0) return;
  char* cstr = calloc_or_throw<char>(str.length() + 1);
  VW::string_cpy(cstr, (str.length() + 1), str.c_str());

  char* next_token;
  char* p = strtok_s(cstr, ",", &next_token);

  try
  {
    while (p)
    {
      char* q = p;
      uint16_t prefix = 1;
      if (q[0] == '+') { q++; }
      else if (q[0] == '-')
      {
        prefix = 0;
        q++;
      }
      if ((q[0] < '1') || (q[0] > '7')) THROW("malformed affix argument (length must be 1..7): " << p);

      uint16_t len = static_cast<uint16_t>(q[0] - '0');
      uint16_t ns = static_cast<uint16_t>(' ');  // default namespace
      if (q[1] != 0)
      {
        if (valid_ns(q[1]))
          ns = static_cast<uint16_t>(q[1]);
        else
          THROW("malformed affix argument (invalid namespace): " << p);

        if (q[2] != 0) THROW("malformed affix argument (too long): " << p);
      }

      uint16_t afx = (len << 1) | (prefix & 0x1);
      all.affix_features[ns] <<= 4;
      all.affix_features[ns] |= afx;

      p = strtok_s(nullptr, ",", &next_token);
    }
  }
  catch (...)
  {
    free(cstr);
    throw;
  }

  free(cstr);
}

void parse_diagnostics(options_i& options, vw& all)
{
  bool version_arg = false;
  bool help = false;
  bool skip_driver = false;
  std::string progress_arg;
  option_group_definition diagnostic_group("Diagnostic options");
  diagnostic_group.add(make_option("version", version_arg).help("Version information"))
      .add(make_option("audit", all.audit).short_name("a").help("print weights of features"))
      .add(make_option("progress", progress_arg)
               .short_name("P")
               .help("Progress update frequency. int: additive, float: multiplicative"))
      .add(make_option("quiet", all.logger.quiet).help("Don't output disgnostics and progress updates"))
      .add(make_option("limit_output", all.logger.upper_limit).help("Avoid chatty output. Limit total printed lines."))
      .add(make_option("dry_run", skip_driver)
               .help("Parse arguments and print corresponding metadata. Will not execute driver."))
      .add(make_option("help", help)
               .short_name("h")
               .help("More information on vowpal wabbit can be found here https://vowpalwabbit.org."));

  options.add_and_parse(diagnostic_group);

  if (help) { all.logger.quiet = true; }

  if (all.logger.quiet)
  {
    logger::log_set_level(logger::log_level::off);
    // This is valid:
    // https://stackoverflow.com/questions/25690636/is-it-valid-to-construct-an-stdostream-from-a-null-buffer This
    // results in the ostream not outputting anything.
    all.trace_message = VW::make_unique<std::ostream>(nullptr);
  }

  if (options.was_supplied("limit_output")) logger::set_max_output(all.logger.upper_limit);

  // pass all.logger.quiet around
  if (all.all_reduce) all.all_reduce->quiet = all.logger.quiet;

  // Upon direct query for version -- spit it out directly to stdout
  if (version_arg)
  {
    std::cout << VW::version.to_string() << " (git commit: " << VW::git_commit << ")\n";
    exit(0);
  }

  if (options.was_supplied("progress") && !all.logger.quiet)
  {
    all.progress_arg = static_cast<float>(::atof(progress_arg.c_str()));
    // --progress interval is dual: either integer or floating-point
    if (progress_arg.find_first_of(".") == std::string::npos)
    {
      // No "." in arg: assume integer -> additive
      all.progress_add = true;
      if (all.progress_arg < 1)
      {
        *(all.trace_message) << "warning: additive --progress <int>"
                             << " can't be < 1: forcing to 1" << endl;
        all.progress_arg = 1;
      }
      all.sd->dump_interval = all.progress_arg;
    }
    else
    {
      // A "." in arg: assume floating-point -> multiplicative
      all.progress_add = false;

      if (all.progress_arg <= 1.0)
      {
        *(all.trace_message) << "warning: multiplicative --progress <float>: " << progress_arg
                             << " is <= 1.0: adding 1.0" << endl;
        all.progress_arg += 1.0;
      }
      else if (all.progress_arg > 9.0)
      {
        *(all.trace_message) << "warning: multiplicative --progress <float>"
                             << " is > 9.0: you probably meant to use an integer" << endl;
      }
      all.sd->dump_interval = 1.0;
    }
  }
}

input_options parse_source(vw& all, options_i& options)
{
  input_options parsed_options;

  option_group_definition input_options("Input options");
  input_options.add(make_option("data", all.data_filename).short_name("d").help("Example set"))
      .add(make_option("daemon", parsed_options.daemon).help("persistent daemon mode on port 26542"))
      .add(make_option("foreground", parsed_options.foreground)
               .help("in persistent daemon mode, do not run in the background"))
      .add(make_option("port", parsed_options.port).help("port to listen on; use 0 to pick unused port"))
      .add(make_option("num_children", all.num_children).help("number of children for persistent daemon mode"))
      .add(make_option("pid_file", parsed_options.pid_file).help("Write pid file in persistent daemon mode"))
      .add(make_option("port_file", parsed_options.port_file).help("Write port used in persistent daemon mode"))
      .add(make_option("cache", parsed_options.cache).short_name("c").help("Use a cache.  The default is <data>.cache"))
      .add(make_option("cache_file", parsed_options.cache_files).help("The location(s) of cache_file."))
      .add(make_option("json", parsed_options.json).help("Enable JSON parsing."))
      .add(make_option("dsjson", parsed_options.dsjson).help("Enable Decision Service JSON parsing."))
      .add(make_option("kill_cache", parsed_options.kill_cache)
               .short_name("k")
               .help("do not reuse existing cache: create a new one always"))
      .add(
          make_option("compressed", parsed_options.compressed)
              .help(
                  "use gzip format whenever possible. If a cache file is being created, this option creates a "
                  "compressed cache file. A mixture of raw-text & compressed inputs are supported with autodetection."))
      .add(make_option("no_stdin", all.stdin_off).help("do not default to reading from stdin"))
      .add(make_option("no_daemon", all.no_daemon)
               .help("Force a loaded daemon or active learning model to accept local input instead of starting in "
                     "daemon mode"))
      .add(make_option("chain_hash", parsed_options.chain_hash_json)
               .help("Enable chain hash in JSON for feature name and string feature value. e.g. {'A': {'B': 'C'}} is "
                     "hashed as "
                     "A^B^C. Note: this will become the default in a future version, so enabling this option will "
                     "migrate you to the new behavior and silence the warning."))
      .add(make_option("flatbuffer", parsed_options.flatbuffer)
               .help("data file will be interpreted as a flatbuffer file"));
#ifdef BUILD_EXTERNAL_PARSER
  VW::external::parser::set_parse_args(input_options, parsed_options);
#endif

  options.add_and_parse(input_options);

  // Check if the options provider has any positional args. Only really makes sense for command line, others just return
  // an empty list.
  const auto positional_tokens = options.get_positional_tokens();
  if (positional_tokens.size() == 1) { all.data_filename = positional_tokens[0]; }
  else if (positional_tokens.size() > 1)
  {
    *(all.trace_message) << "Warning: Multiple data files passed as positional parameters, only the first one will be "
                            "read and the rest will be ignored."
                         << endl;
  }

  if (parsed_options.daemon || options.was_supplied("pid_file") || (options.was_supplied("port") && !all.active))
  {
    all.daemon = true;
    // allow each child to process up to 1e5 connections
    all.numpasses = static_cast<size_t>(1e5);
  }

  // Add an implicit cache file based on the data filename.
  if (parsed_options.cache) { parsed_options.cache_files.push_back(all.data_filename + ".cache"); }

  if ((parsed_options.cache || options.was_supplied("cache_file")) && options.was_supplied("invert_hash"))
    THROW("invert_hash is incompatible with a cache file.  Use it in single pass mode only.");

  if (!all.holdout_set_off &&
      (options.was_supplied("output_feature_regularizer_binary") ||
          options.was_supplied("output_feature_regularizer_text")))
  {
    all.holdout_set_off = true;
    *(all.trace_message) << "Making holdout_set_off=true since output regularizer specified" << endl;
  }

  return parsed_options;
}

namespace VW
{
const char* are_features_compatible(vw& vw1, vw& vw2)
{
  if (vw1.example_parser->hasher != vw2.example_parser->hasher) return "hasher";

  if (!std::equal(vw1.spelling_features.begin(), vw1.spelling_features.end(), vw2.spelling_features.begin()))
    return "spelling_features";

  if (!std::equal(vw1.affix_features.begin(), vw1.affix_features.end(), vw2.affix_features.begin()))
    return "affix_features";

  if (vw1.skip_gram_transformer != nullptr && vw2.skip_gram_transformer != nullptr)
  {
    const auto& vw1_ngram_strings = vw1.skip_gram_transformer->get_initial_ngram_definitions();
    const auto& vw2_ngram_strings = vw2.skip_gram_transformer->get_initial_ngram_definitions();
    const auto& vw1_skips_strings = vw1.skip_gram_transformer->get_initial_skip_definitions();
    const auto& vw2_skips_strings = vw2.skip_gram_transformer->get_initial_skip_definitions();

    if (!std::equal(vw1_ngram_strings.begin(), vw1_ngram_strings.end(), vw2_ngram_strings.begin())) return "ngram";

    if (!std::equal(vw1_skips_strings.begin(), vw1_skips_strings.end(), vw2_skips_strings.begin())) return "skips";
  }
  else if (vw1.skip_gram_transformer != nullptr || vw2.skip_gram_transformer != nullptr)
  {
    // If one of them didn't define the ngram transformer then they differ by ngram (skips depends on ngram)
    return "ngram";
  }

  if (!std::equal(vw1.limit.begin(), vw1.limit.end(), vw2.limit.begin())) return "limit";

  if (vw1.num_bits != vw2.num_bits) return "num_bits";

  if (vw1.permutations != vw2.permutations) return "permutations";

  if (vw1.interactions.size() != vw2.interactions.size()) return "interactions size";

  if (vw1.ignore_some != vw2.ignore_some) return "ignore_some";

  if (vw1.ignore_some && !std::equal(vw1.ignore.begin(), vw1.ignore.end(), vw2.ignore.begin())) return "ignore";

  if (vw1.ignore_some_linear != vw2.ignore_some_linear) return "ignore_some_linear";

  if (vw1.ignore_some_linear &&
      !std::equal(vw1.ignore_linear.begin(), vw1.ignore_linear.end(), vw2.ignore_linear.begin()))
    return "ignore_linear";

  if (vw1.redefine_some != vw2.redefine_some) return "redefine_some";

  if (vw1.redefine_some && !std::equal(vw1.redefine.begin(), vw1.redefine.end(), vw2.redefine.begin()))
    return "redefine";

  if (vw1.add_constant != vw2.add_constant) return "add_constant";

  if (vw1.dictionary_path.size() != vw2.dictionary_path.size()) return "dictionary_path size";

  if (!std::equal(vw1.dictionary_path.begin(), vw1.dictionary_path.end(), vw2.dictionary_path.begin()))
    return "dictionary_path";

  for (auto i = std::begin(vw1.interactions), j = std::begin(vw2.interactions); i != std::end(vw1.interactions);
       ++i, ++j)
    if (*i != *j) return "interaction mismatch";

  return nullptr;
}

}  // namespace VW

// Return a copy of std::string replacing \x00 sequences in it
std::string spoof_hex_encoded_namespaces(const std::string& arg)
{
  constexpr size_t NUMBER_OF_HEX_CHARS = 2;
  // "\x" + hex chars
  constexpr size_t LENGTH_OF_HEX_TOKEN = 2 + NUMBER_OF_HEX_CHARS;
  constexpr size_t HEX_BASE = 16;

  // Too short to be hex encoded.
  if (arg.size() < LENGTH_OF_HEX_TOKEN) { return arg; }

  std::string res;
  size_t pos = 0;
  while (pos < arg.size() - (LENGTH_OF_HEX_TOKEN - 1))
  {
    if (arg[pos] == '\\' && arg[pos + 1] == 'x')
    {
      std::string substr = arg.substr(pos + NUMBER_OF_HEX_CHARS, NUMBER_OF_HEX_CHARS);
      char* p;
      auto c = static_cast<namespace_index>(std::strtoul(substr.c_str(), &p, HEX_BASE));
      if (*p == '\0')
      {
        res.push_back(c);
        pos += LENGTH_OF_HEX_TOKEN;
      }
      else
      {
        logger::errlog_warn("Possibly malformed hex representation of a namespace: '\\x{}'", substr);
        res.push_back(arg[pos++]);
      }
    }
    else
    {
      res.push_back(arg[pos++]);
    }
  }

  // Copy last 2 characters
  while (pos < arg.size()) { res.push_back(arg[pos++]); }

  return res;
}

void parse_feature_tweaks(
    options_i& options, vw& all, bool interactions_settings_duplicated, std::vector<std::string>& dictionary_nses)
{
  std::string hash_function("strings");
  uint32_t new_bits;
  std::vector<std::string> spelling_ns;
  std::vector<std::string> quadratics;
  std::vector<std::string> cubics;
  std::vector<std::string> interactions;
  std::vector<std::string> ignores;
  std::vector<std::string> ignore_linears;
  std::vector<std::string> keeps;
  std::vector<std::string> redefines;

  std::vector<std::string> ngram_strings;
  std::vector<std::string> skip_strings;

  std::vector<std::string> dictionary_path;

  bool noconstant;
  bool leave_duplicate_interactions;
  std::string affix;
  std::string q_colon;

  option_group_definition feature_options("Feature options");
  feature_options
      .add(make_option("hash", hash_function).keep().help("how to hash the features. Available options: strings, all"))
      .add(make_option("hash_seed", all.hash_seed).keep().default_value(0).help("seed for hash function"))
      .add(make_option("ignore", ignores).keep().help("ignore namespaces beginning with character <arg>"))
      .add(make_option("ignore_linear", ignore_linears)
               .keep()
               .help("ignore namespaces beginning with character <arg> for linear terms only"))
      .add(make_option("keep", keeps).keep().help("keep namespaces beginning with character <arg>"))
      .add(make_option("redefine", redefines)
               .keep()
               .help("redefine namespaces beginning with characters of std::string S as namespace N. <arg> shall be in "
                     "form "
                     "'N:=S' where := is operator. Empty N or S are treated as default namespace. Use ':' as a "
                     "wildcard in S.")
               .keep())
      .add(make_option("bit_precision", new_bits).short_name("b").help("number of bits in the feature table"))
      .add(make_option("noconstant", noconstant).help("Don't add a constant feature"))
      .add(make_option("constant", all.initial_constant).short_name("C").help("Set initial value of constant"))
      .add(make_option("ngram", ngram_strings)
               .help("Generate N grams. To generate N grams for a single namespace 'foo', arg should be fN."))
      .add(make_option("skips", skip_strings)
               .help("Generate skips in N grams. This in conjunction with the ngram tag can be used to generate "
                     "generalized n-skip-k-gram. To generate n-skips for a single namespace 'foo', arg should be fN."))
      .add(make_option("feature_limit", all.limit_strings)
               .help("limit to N features. To apply to a single namespace 'foo', arg should be fN"))
      .add(make_option("affix", affix)
               .keep()
               .help("generate prefixes/suffixes of features; argument '+2a,-3b,+1' means generate 2-char prefixes for "
                     "namespace a, 3-char suffixes for b and 1 char prefixes for default namespace"))
      .add(make_option("spelling", spelling_ns)
               .keep()
               .help("compute spelling features for a give namespace (use '_' for default namespace)"))
      .add(make_option("dictionary", dictionary_nses)
               .keep()
               .help("read a dictionary for additional features (arg either 'x:file' or just 'file')"))
      .add(make_option("dictionary_path", dictionary_path)
               .help("look in this directory for dictionaries; defaults to current directory or env{PATH}"))
      .add(make_option("interactions", interactions)
               .keep()
               .help("Create feature interactions of any level between namespaces."))
      .add(make_option("permutations", all.permutations)
               .help("Use permutations instead of combinations for feature interactions of same namespace."))
      .add(make_option("leave_duplicate_interactions", leave_duplicate_interactions)
               .help("Don't remove interactions with duplicate combinations of namespaces. For ex. this is a "
                     "duplicate: '-q ab -q ba' and a lot more in '-q ::'."))
      .add(make_option("quadratic", quadratics).short_name("q").keep().help("Create and use quadratic features"))
      // TODO this option is unused - remove?
      .add(make_option("q:", q_colon).help("DEPRECATED ':' corresponds to a wildcard for all printable characters"))
      .add(make_option("cubic", cubics).keep().help("Create and use cubic features"));
  options.add_and_parse(feature_options);

  // feature manipulation
  all.example_parser->hasher = getHasher(hash_function);

  if (options.was_supplied("spelling"))
  {
    for (size_t id = 0; id < spelling_ns.size(); id++)
    {
      spelling_ns[id] = spoof_hex_encoded_namespaces(spelling_ns[id]);
      if (spelling_ns[id][0] == '_')
        all.spelling_features[static_cast<unsigned char>(' ')] = true;
      else
        all.spelling_features[static_cast<size_t>(spelling_ns[id][0])] = true;
    }
  }

  if (options.was_supplied("q:"))
  {
    *(all.trace_message)
        << "WARNING: '--q:' is deprecated and not supported. You can use : as a wildcard in interactions." << endl;
  }

  if (options.was_supplied("affix")) parse_affix_argument(all, spoof_hex_encoded_namespaces(affix));

  // Process ngram and skips arguments
  if (options.was_supplied("skips"))
  {
    if (!options.was_supplied("ngram")) { THROW("You can not skip unless ngram is > 1"); }
  }

  if (options.was_supplied("ngram"))
  {
    if (options.was_supplied("sort_features")) { THROW("ngram is incompatible with sort_features."); }

    std::vector<std::string> hex_decoded_ngram_strings;
    hex_decoded_ngram_strings.reserve(ngram_strings.size());
    std::transform(ngram_strings.begin(), ngram_strings.end(), std::back_inserter(hex_decoded_ngram_strings),
        [](const std::string& arg) { return spoof_hex_encoded_namespaces(arg); });

    std::vector<std::string> hex_decoded_skip_strings;
    hex_decoded_skip_strings.reserve(skip_strings.size());
    std::transform(skip_strings.begin(), skip_strings.end(), std::back_inserter(hex_decoded_skip_strings),
        [](const std::string& arg) { return spoof_hex_encoded_namespaces(arg); });

    all.skip_gram_transformer = VW::make_unique<VW::kskip_ngram_transformer>(
        VW::kskip_ngram_transformer::build(hex_decoded_ngram_strings, hex_decoded_skip_strings, all.logger.quiet));
  }

  if (options.was_supplied("feature_limit")) compile_limits(all.limit_strings, all.limit, all.logger.quiet);

  if (options.was_supplied("bit_precision"))
  {
    if (all.default_bits == false && new_bits != all.num_bits)
      THROW("Number of bits is set to " << new_bits << " and " << all.num_bits
                                        << " by argument and model.  That does not work.");

    all.default_bits = false;
    all.num_bits = new_bits;

    VW::validate_num_bits(all);
  }

  // prepare namespace interactions
  std::vector<std::vector<namespace_index>> decoded_interactions;

  if ( ( (!all.interactions.empty() && /*data was restored from old model file directly to v_array and will be overriden automatically*/
          (options.was_supplied("quadratic") || options.was_supplied("cubic") || options.was_supplied("interactions")) ) )
       ||
       interactions_settings_duplicated /*settings were restored from model file to file_options and overriden by params from command line*/)
  {
    *(all.trace_message)
        << "WARNING: model file has set of {-q, --cubic, --interactions} settings stored, but they'll be "
           "OVERRIDEN by set of {-q, --cubic, --interactions} settings from command line."
        << endl;

    // in case arrays were already filled in with values from old model file - reset them
    if (!all.interactions.empty()) { all.interactions.clear(); }
  }

  if (options.was_supplied("quadratic"))
  {
    if (!all.logger.quiet) *(all.trace_message) << "creating quadratic features for pairs: ";

    for (auto& i : quadratics)
    {
      if (i.size() != 2) { THROW("error, quadratic features must involve two sets.)") }
      auto encoded = spoof_hex_encoded_namespaces(i);
      decoded_interactions.emplace_back(encoded.begin(), encoded.end());
      if (!all.logger.quiet) *(all.trace_message) << i << " ";
    }

    if (!all.logger.quiet && !options.was_supplied("leave_duplicate_interactions"))
    {
      bool contains_wildcard_quadratic =
          std::find_if(quadratics.begin(), quadratics.end(), [](const std::string& interaction) {
            return interaction.find(wildcard_namespace) != std::string::npos;
          }) != quadratics.end();
      if (contains_wildcard_quadratic)
      {
        *(all.trace_message) << "\n"
                             << "WARNING: any duplicate namespace interactions will be removed\n"
                             << "You can use --leave_duplicate_interactions to disable this behaviour.";
      }
    }


    if (!all.logger.quiet) *(all.trace_message) << endl;
  }

  if (options.was_supplied("cubic"))
  {
    if (!all.logger.quiet) *(all.trace_message) << "creating cubic features for triples: ";
    for (const auto& i : cubics)
    {
      if (i.size() != 3) { THROW("error, cubic features must involve three sets.") }
      auto encoded = spoof_hex_encoded_namespaces(i);
      decoded_interactions.emplace_back(encoded.begin(), encoded.end());
      if (!all.logger.quiet) *(all.trace_message) << i << " ";
    }
    if (!all.logger.quiet) *(all.trace_message) << endl;

    bool contains_wildcard_cubic = std::find_if(cubics.begin(), cubics.end(), [](const std::string& interaction) {
      return interaction.find(wildcard_namespace) != std::string::npos;
    }) != cubics.end();
    if (contains_wildcard_cubic)
    {
      *(all.trace_message) << "\n"
                           << "WARNING: any duplicate namespace interactions will be removed\n"
                           << "You can use --leave_duplicate_interactions to disable this behaviour.";
    }
  }

  if (options.was_supplied("interactions"))
  {
    if (!all.logger.quiet) *(all.trace_message) << "creating features for following interactions: ";

    for (const auto& i : interactions)
    {
      if (i.size() < 2) { THROW("error, feature interactions must involve at least two namespaces") }
      auto encoded = spoof_hex_encoded_namespaces(i);
      decoded_interactions.emplace_back(encoded.begin(), encoded.end());
      if (!all.logger.quiet) *(all.trace_message) << i << " ";
    }
    if (!all.logger.quiet) *(all.trace_message) << endl;
  }

  if (decoded_interactions.size() > 0)
  {
    // Sorts the overall list
    std::sort(decoded_interactions.begin(), decoded_interactions.end(), INTERACTIONS::sort_interactions_comparator);

    size_t removed_cnt = 0;
    size_t sorted_cnt = 0;
    // Sorts individual interactions
    INTERACTIONS::sort_and_filter_duplicate_interactions(
        decoded_interactions, !leave_duplicate_interactions, removed_cnt, sorted_cnt);

    if (removed_cnt > 0 && !all.logger.quiet)
    {
      *(all.trace_message) << "WARNING: duplicate namespace interactions were found. Removed: " << removed_cnt << '.'
                           << endl
                           << "You can use --leave_duplicate_interactions to disable this behaviour." << endl;
    }

    if (sorted_cnt > 0 && !all.logger.quiet)
    {
      *(all.trace_message) << "WARNING: some interactions contain duplicate characters and their characters order has "
                              "been changed. Interactions affected: "
                           << sorted_cnt << '.' << endl;
    }

    all.interactions = std::move(decoded_interactions);
  }

  for (size_t i = 0; i < 256; i++)
  {
    all.ignore[i] = false;
    all.ignore_linear[i] = false;
  }
  all.ignore_some = false;
  all.ignore_some_linear = false;

  if (options.was_supplied("ignore"))
  {
    all.ignore_some = true;

    for (auto& i : ignores)
    {
      i = spoof_hex_encoded_namespaces(i);
      for (auto j : i) all.ignore[static_cast<size_t>(static_cast<unsigned char>(j))] = true;
    }

    if (!all.logger.quiet)
    {
      *(all.trace_message) << "ignoring namespaces beginning with: ";
      for (auto const& ignore : ignores)
        for (auto const character : ignore) *(all.trace_message) << character << " ";

      *(all.trace_message) << endl;
    }
  }

  if (options.was_supplied("ignore_linear"))
  {
    all.ignore_some_linear = true;

    for (auto& i : ignore_linears)
    {
      i = spoof_hex_encoded_namespaces(i);
      for (auto j : i) all.ignore_linear[static_cast<size_t>(static_cast<unsigned char>(j))] = true;
    }

    if (!all.logger.quiet)
    {
      *(all.trace_message) << "ignoring linear terms for namespaces beginning with: ";
      for (auto const& ignore : ignore_linears)
        for (auto const character : ignore) *(all.trace_message) << character << " ";

      *(all.trace_message) << endl;
    }
  }

  if (options.was_supplied("keep"))
  {
    for (size_t i = 0; i < 256; i++) all.ignore[i] = true;

    all.ignore_some = true;

    for (auto& i : keeps)
    {
      i = spoof_hex_encoded_namespaces(i);
      for (const auto& j : i) all.ignore[static_cast<size_t>(static_cast<unsigned char>(j))] = false;
    }

    if (!all.logger.quiet)
    {
      *(all.trace_message) << "using namespaces beginning with: ";
      for (auto const& keep : keeps)
        for (auto const character : keep) *(all.trace_message) << character << " ";

      *(all.trace_message) << endl;
    }
  }

  // --redefine param code
  all.redefine_some = false;  // false by default

  if (options.was_supplied("redefine"))
  {
    // initail values: i-th namespace is redefined to i itself
    for (size_t i = 0; i < 256; i++) all.redefine[i] = static_cast<unsigned char>(i);

    // note: --redefine declaration order is matter
    // so --redefine :=L --redefine ab:=M  --ignore L  will ignore all except a and b under new M namspace

    for (const auto& arg : redefines)
    {
      const std::string& argument = spoof_hex_encoded_namespaces(arg);
      size_t arg_len = argument.length();

      size_t operator_pos = 0;  // keeps operator pos + 1 to stay unsigned type
      bool operator_found = false;
      unsigned char new_namespace = ' ';

      // let's find operator ':=' position in N:=S
      for (size_t i = 0; i < arg_len; i++)
      {
        if (operator_found)
        {
          if (i > 2) { new_namespace = argument[0]; }  // N is not empty
          break;
        }
        else if (argument[i] == ':')
          operator_pos = i + 1;
        else if ((argument[i] == '=') && (operator_pos == i))
          operator_found = true;
      }

      if (!operator_found) THROW("argument of --redefine is malformed. Valid format is N:=S, :=S or N:=");

      if (++operator_pos > 3)  // seek operator end
        *(all.trace_message)
            << "WARNING: multiple namespaces are used in target part of --redefine argument. Only first one ('"
            << new_namespace << "') will be used as target namespace." << endl;

      all.redefine_some = true;

      // case ':=S' doesn't require any additional code as new_namespace = ' ' by default

      if (operator_pos == arg_len)  // S is empty, default namespace shall be used
        all.redefine[static_cast<int>(' ')] = new_namespace;
      else
        for (size_t i = operator_pos; i < arg_len; i++)
        {
          // all namespaces from S are redefined to N
          unsigned char c = argument[i];
          if (c != ':')
            all.redefine[c] = new_namespace;
          else
          {
            // wildcard found: redefine all except default and break
            for (size_t j = 0; j < 256; j++) all.redefine[j] = new_namespace;
            break;  // break processing S
          }
        }
    }
  }

  if (options.was_supplied("dictionary"))
  {
    if (options.was_supplied("dictionary_path"))
      for (const std::string& path : dictionary_path)
        if (directory_exists(path)) all.dictionary_path.push_back(path);
    if (directory_exists(".")) all.dictionary_path.push_back(".");

#if _WIN32
    std::string PATH;
    char* buf;
    size_t buf_size;
    auto err = _dupenv_s(&buf, &buf_size, "PATH");
    if (!err && buf_size != 0)
    {
      PATH = std::string(buf, buf_size);
      free(buf);
    }
    const char delimiter = ';';
#else
    const std::string PATH = getenv("PATH");
    const char delimiter = ':';
#endif
    if (!PATH.empty())
    {
      size_t previous = 0;
      size_t index = PATH.find(delimiter);
      while (index != std::string::npos)
      {
        all.dictionary_path.push_back(PATH.substr(previous, index - previous));
        previous = index + 1;
        index = PATH.find(delimiter, previous);
      }
      all.dictionary_path.push_back(PATH.substr(previous));
    }
  }

  if (noconstant) all.add_constant = false;
}

void parse_example_tweaks(options_i& options, vw& all)
{
  std::string named_labels;
  std::string loss_function;
  float loss_parameter = 0.0;
  size_t early_terminate_passes;
  bool test_only = false;

  option_group_definition example_options("Example options");
  example_options.add(make_option("testonly", test_only).short_name("t").help("Ignore label information and just test"))
      .add(make_option("holdout_off", all.holdout_set_off).help("no holdout data in multiple passes"))
      .add(make_option("holdout_period", all.holdout_period).default_value(10).help("holdout period for test only"))
      .add(make_option("holdout_after", all.holdout_after)
               .help("holdout after n training examples, default off (disables holdout_period)"))
      .add(
          make_option("early_terminate", early_terminate_passes)
              .default_value(3)
              .help(
                  "Specify the number of passes tolerated when holdout loss doesn't decrease before early termination"))
      .add(make_option("passes", all.numpasses).help("Number of Training Passes"))
      .add(make_option("initial_pass_length", all.pass_length).help("initial number of examples per pass"))
      .add(make_option("examples", all.max_examples).help("number of examples to parse"))
      .add(make_option("min_prediction", all.sd->min_label).help("Smallest prediction to output"))
      .add(make_option("max_prediction", all.sd->max_label).help("Largest prediction to output"))
      .add(make_option("sort_features", all.example_parser->sort_features)
               .help("turn this on to disregard order in which features have been defined. This will lead to smaller "
                     "cache sizes"))
      .add(make_option("loss_function", loss_function)
               .default_value("squared")
               .help("Specify the loss function to be used, uses squared by default. Currently available ones are "
                     "squared, classic, hinge, logistic, quantile and poisson."))
      .add(make_option("quantile_tau", loss_parameter)
               .default_value(0.5f)
               .help("Parameter \\tau associated with Quantile loss. Defaults to 0.5"))
      .add(make_option("l1", all.l1_lambda).help("l_1 lambda"))
      .add(make_option("l2", all.l2_lambda).help("l_2 lambda"))
      .add(make_option("no_bias_regularization", all.no_bias).help("no bias in regularization"))
      .add(make_option("named_labels", named_labels)
               .keep()
               .help("use names for labels (multiclass, etc.) rather than integers, argument specified all possible "
                     "labels, comma-sep, eg \"--named_labels Noun,Verb,Adj,Punc\""));
  options.add_and_parse(example_options);

  if (test_only || all.eta == 0.)
  {
    if (!all.logger.quiet) *(all.trace_message) << "only testing" << endl;
    all.training = false;
    if (all.lda > 0) all.eta = 0;
  }
  else
    all.training = true;

  if ((all.numpasses > 1 || all.holdout_after > 0) && !all.holdout_set_off)
    all.holdout_set_off = false;  // holdout is on unless explicitly off
  else
    all.holdout_set_off = true;

  if (options.was_supplied("min_prediction") || options.was_supplied("max_prediction") || test_only)
    all.set_minmax = noop_mm;

  if (options.was_supplied("named_labels"))
  {
    all.sd->ldict = VW::make_unique<VW::named_labels>(named_labels);
    if (!all.logger.quiet) *(all.trace_message) << "parsed " << all.sd->ldict->getK() << " named labels" << endl;
  }

  all.loss = getLossFunction(all, loss_function, loss_parameter);

  if (all.l1_lambda < 0.)
  {
    *(all.trace_message) << "l1_lambda should be nonnegative: resetting from " << all.l1_lambda << " to 0" << endl;
    all.l1_lambda = 0.;
  }
  if (all.l2_lambda < 0.)
  {
    *(all.trace_message) << "l2_lambda should be nonnegative: resetting from " << all.l2_lambda << " to 0" << endl;
    all.l2_lambda = 0.;
  }
  all.reg_mode += (all.l1_lambda > 0.) ? 1 : 0;
  all.reg_mode += (all.l2_lambda > 0.) ? 2 : 0;
  if (!all.logger.quiet)
  {
    if (all.reg_mode % 2 && !options.was_supplied("bfgs"))
      *(all.trace_message) << "using l1 regularization = " << all.l1_lambda << endl;
    if (all.reg_mode > 1) *(all.trace_message) << "using l2 regularization = " << all.l2_lambda << endl;
  }
}

void parse_output_preds(options_i& options, vw& all)
{
  std::string predictions;
  std::string raw_predictions;

  option_group_definition output_options("Output options");
  output_options.add(make_option("predictions", predictions).short_name("p").help("File to output predictions to"))
      .add(make_option("raw_predictions", raw_predictions)
               .short_name("r")
               .help("File to output unnormalized predictions to"));
  options.add_and_parse(output_options);

  if (options.was_supplied("predictions"))
  {
    if (!all.logger.quiet) *(all.trace_message) << "predictions = " << predictions << endl;

    if (predictions == "stdout")
    {
      all.final_prediction_sink.push_back(VW::io::open_stdout());  // stdout
    }
    else
    {
      try
      {
        all.final_prediction_sink.push_back(VW::io::open_file_writer(predictions));
      }
      catch (...)
      {
        *(all.trace_message) << "Error opening the predictions file: " << predictions << endl;
      }
    }
  }

  if (options.was_supplied("raw_predictions"))
  {
    if (!all.logger.quiet)
    {
      *(all.trace_message) << "raw predictions = " << raw_predictions << endl;
      if (options.was_supplied("binary"))
        *(all.trace_message)
            << "Warning: --raw_predictions has no defined value when --binary specified, expect no output" << endl;
    }
    if (raw_predictions == "stdout") { all.raw_prediction = VW::io::open_stdout(); }
    else
    {
      all.raw_prediction = VW::io::open_file_writer(raw_predictions);
    }
  }
}

void parse_output_model(options_i& options, vw& all)
{
  option_group_definition output_model_options("Output model");
  output_model_options
      .add(make_option("final_regressor", all.final_regressor_name).short_name("f").help("Final regressor"))
      .add(make_option("readable_model", all.text_regressor_name)
               .help("Output human-readable final regressor with numeric features"))
      .add(make_option("invert_hash", all.inv_hash_regressor_name)
               .help("Output human-readable final regressor with feature names.  Computationally expensive."))
      .add(make_option("save_resume", all.save_resume)
               .help("save extra state so learning can be resumed later with new data"))
      .add(make_option("preserve_performance_counters", all.preserve_performance_counters)
               .help("reset performance counters when warmstarting"))
      .add(make_option("save_per_pass", all.save_per_pass).help("Save the model after every pass over data"))
      .add(make_option("output_feature_regularizer_binary", all.per_feature_regularizer_output)
               .help("Per feature regularization output file"))
      .add(make_option("output_feature_regularizer_text", all.per_feature_regularizer_text)
               .help("Per feature regularization output file, in text"))
      .add(make_option("id", all.id).help("User supplied ID embedded into the final regressor"));
  options.add_and_parse(output_model_options);

  if (all.final_regressor_name.compare("") && !all.logger.quiet)
    *(all.trace_message) << "final_regressor = " << all.final_regressor_name << endl;

  if (options.was_supplied("invert_hash")) all.hash_inv = true;

  // Question: This doesn't seem necessary
  // if (options.was_supplied("id") && find(arg.args.begin(), arg.args.end(), "--id") == arg.args.end())
  // {
  //   arg.args.push_back("--id");
  //   arg.args.push_back(arg.vm["id"].as<std::string>());
  // }
}

void load_input_model(vw& all, io_buf& io_temp)
{
  // Need to see if we have to load feature mask first or second.
  // -i and -mask are from same file, load -i file first so mask can use it
  if (!all.feature_mask.empty() && all.initial_regressors.size() > 0 && all.feature_mask == all.initial_regressors[0])
  {
    // load rest of regressor
    all.l->save_load(io_temp, true, false);
    io_temp.close_file();

    parse_mask_regressor_args(all, all.feature_mask, all.initial_regressors);
  }
  else
  {  // load mask first
    parse_mask_regressor_args(all, all.feature_mask, all.initial_regressors);

    // load rest of regressor
    all.l->save_load(io_temp, true, false);
    io_temp.close_file();
  }
}

ssize_t trace_message_wrapper_adapter(void* context, const char* buffer, size_t num_bytes)
{
  auto* wrapper_context = reinterpret_cast<trace_message_wrapper*>(context);
  std::string str(buffer, num_bytes);
  wrapper_context->_trace_message(wrapper_context->_inner_context, str);
  return static_cast<ssize_t>(num_bytes);
}

vw& parse_args(
    std::unique_ptr<options_i, options_deleter_type> options, trace_message_t trace_listener, void* trace_context)
{
  vw& all = *(new vw());
  all.options = std::move(options);

  if (trace_listener)
  {
    // Since the trace_message_t interface uses a string and the writer interface uses a buffer we unfortunately
    // need to adapt between them here.
    all.trace_message_wrapper_context = std::make_shared<trace_message_wrapper>(trace_context, trace_listener);
    all.trace_message = VW::make_unique<VW::io::owning_ostream>(VW::make_unique<VW::io::writer_stream_buf>(
        VW::io::create_custom_writer(all.trace_message_wrapper_context.get(), trace_message_wrapper_adapter)));
  }

  try
  {
    time(&all.init_time);

    bool strict_parse = false;
    int ring_size_tmp;
    option_group_definition vw_args("VW options");
    vw_args.add(make_option("ring_size", ring_size_tmp).default_value(256).help("size of example ring"))
        .add(make_option("strict_parse", strict_parse).help("throw on malformed examples"));
    all.options->add_and_parse(vw_args);

    if (ring_size_tmp <= 0) { THROW("ring_size should be positive"); }
    size_t ring_size = static_cast<size_t>(ring_size_tmp);

    all.example_parser = new parser{ring_size, strict_parse};
    all.example_parser->_shared_data = all.sd;

    option_group_definition update_args("Update options");
    update_args.add(make_option("learning_rate", all.eta).help("Set learning rate").short_name("l"))
        .add(make_option("power_t", all.power_t).help("t power value"))
        .add(make_option("decay_learning_rate", all.eta_decay_rate)
                 .help("Set Decay factor for learning_rate between passes"))
        .add(make_option("initial_t", all.sd->t).help("initial t value"))
        .add(make_option("feature_mask", all.feature_mask)
                 .help("Use existing regressor to determine which parameters may be updated.  If no initial_regressor "
                       "given, also used for initial weights."));
    all.options->add_and_parse(update_args);

    option_group_definition weight_args("Weight options");
    weight_args
        .add(make_option("initial_regressor", all.initial_regressors).help("Initial regressor(s)").short_name("i"))
        .add(make_option("initial_weight", all.initial_weight).help("Set all weights to an initial value of arg."))
        .add(make_option("random_weights", all.random_weights).help("make initial weights random"))
        .add(make_option("normal_weights", all.normal_weights).help("make initial weights normal"))
        .add(make_option("truncated_normal_weights", all.tnormal_weights).help("make initial weights truncated normal"))
        .add(make_option("sparse_weights", all.weights.sparse).help("Use a sparse datastructure for weights"))
        .add(make_option("input_feature_regularizer", all.per_feature_regularizer_input)
                 .help("Per feature regularization input file"));
    all.options->add_and_parse(weight_args);

    std::string span_server_arg;
    int span_server_port_arg;
    // bool threads_arg;
    size_t unique_id_arg;
    size_t total_arg;
    size_t node_arg;
    option_group_definition parallelization_args("Parallelization options");
    parallelization_args
        .add(make_option("span_server", span_server_arg).help("Location of server for setting up spanning tree"))
        //(make_option("threads", threads_arg).help("Enable multi-threading")) Unused option?
        .add(make_option("unique_id", unique_id_arg).default_value(0).help("unique id used for cluster parallel jobs"))
        .add(
            make_option("total", total_arg).default_value(1).help("total number of nodes used in cluster parallel job"))
        .add(make_option("node", node_arg).default_value(0).help("node number in cluster parallel job"))
        .add(make_option("span_server_port", span_server_port_arg)
                 .default_value(26543)
                 .help("Port of the server for setting up spanning tree"));
    all.options->add_and_parse(parallelization_args);

    // total, unique_id and node must be specified together.
    if ((all.options->was_supplied("total") || all.options->was_supplied("node") ||
            all.options->was_supplied("unique_id")) &&
        !(all.options->was_supplied("total") && all.options->was_supplied("node") &&
            all.options->was_supplied("unique_id")))
    { THROW("you must specificy unique_id, total, and node if you specify any"); }

    if (all.options->was_supplied("span_server"))
    {
      all.all_reduce_type = AllReduceType::Socket;
      all.all_reduce = new AllReduceSockets(
          span_server_arg, span_server_port_arg, unique_id_arg, total_arg, node_arg, all.logger.quiet);
    }

    parse_diagnostics(*all.options.get(), all);

    all.initial_t = static_cast<float>(all.sd->t);
    return all;
  }
  catch (...)
  {
    VW::finish(all);
    throw;
  }
}

bool check_interaction_settings_collision(options_i& options, std::string file_options)
{
  bool command_line_has_interaction = options.was_supplied("q") || options.was_supplied("quadratic") ||
      options.was_supplied("cubic") || options.was_supplied("interactions");

  if (!command_line_has_interaction) return false;

  // we don't use -q to save pairs in all.file_options, so only 3 options checked
  bool file_options_has_interaction = file_options.find("--quadratic") != std::string::npos;
  file_options_has_interaction = file_options_has_interaction || (file_options.find("--cubic") != std::string::npos);
  file_options_has_interaction =
      file_options_has_interaction || (file_options.find("--interactions") != std::string::npos);

  return file_options_has_interaction;
}

void merge_options_from_header_strings(const std::vector<std::string>& strings, bool skip_interactions,
    VW::config::options_i& options, bool& is_ccb_input_model)
{
  po::options_description desc("");

  // Get list of options in file options std::string
  po::parsed_options pos = po::command_line_parser(strings).options(desc).allow_unregistered().run();

  bool skipping = false;
  std::string saved_key = "";
  unsigned int count = 0;
  bool first_seen = false;

  for (auto opt : pos.options)
  {
    // If we previously encountered an option we want to skip, ignore tokens without --.
    if (skipping)
    {
      for (auto token : opt.original_tokens)
      {
        auto found = token.find("--");
        if (found != std::string::npos) { skipping = false; }
      }

      if (skipping)
      {
        saved_key = "";
        continue;
      }
    }

    bool treat_as_value = false;
    // If the key starts with a digit, this is a mis-interpretation of a value as a key. Pull it into the previous
    // option. This was found in the case of --lambda -1, misinterpreting -1 as an option key. The easy way to fix this
    // requires introducing "identifier-like" semantics for options keys, e.g. "does not begin with a digit". That does
    // not seem like an unreasonable restriction. The logical check here is: is "string_key" of the form {'-', <digit>,
    // <etc.>}.
    if (opt.string_key.length() > 1 && opt.string_key[0] == '-' && opt.string_key[1] >= '0' && opt.string_key[1] <= '9')
    { treat_as_value = true; }

    // File options should always use long form.

    // If the key is empty this must be a value, otherwise set the key.
    if (!treat_as_value && opt.string_key != "")
    {
      // If the new token is a new option and there were no values previously it was a bool option. Add it as a switch.
      if (count == 0 && first_seen) { options.insert(saved_key, ""); }

      count = 0;
      first_seen = true;

      // If the interaction settings are doubled, the copy in the model file is ignored.
      if (skip_interactions &&
          (opt.string_key == "quadratic" || opt.string_key == "cubic" || opt.string_key == "interactions"))
      {
        // skip this option.
        skipping = true;
        first_seen = false;
        continue;
      }
      saved_key = opt.string_key;
      is_ccb_input_model = is_ccb_input_model || (saved_key == "ccb_explore_adf");

      if (opt.value.size() > 0)
      {
        for (auto value : opt.value)
        {
          options.insert(saved_key, value);
          count++;
        }
      }
    }
    else
    {
      // If treat_as_value is set, boost incorrectly interpreted the token as containing an option key
      // In this case, what should have happened is all original_tokens items should be in value.
      auto source = treat_as_value ? opt.original_tokens : opt.value;
      for (auto value : source)
      {
        options.insert(saved_key, value);
        count++;
      }
    }
  }

  if (count == 0 && saved_key != "") { options.insert(saved_key, ""); }
}

options_i& load_header_merge_options(options_i& options, vw& all, io_buf& model, bool& interactions_settings_duplicated)
{
  std::string file_options;
  save_load_header(all, model, true, false, file_options, options);

  interactions_settings_duplicated = check_interaction_settings_collision(options, file_options);

  // Convert file_options into  vector.
  std::istringstream ss{file_options};
  std::vector<std::string> container{std::istream_iterator<std::string>{ss}, std::istream_iterator<std::string>{}};

  merge_options_from_header_strings(container, interactions_settings_duplicated, options, all.is_ccb_input_model);

  return options;
}

void parse_modules(
    options_i& options, vw& all, bool interactions_settings_duplicated, std::vector<std::string>& dictionary_nses)
{
  option_group_definition rand_options("Randomization options");
  rand_options.add(make_option("random_seed", all.random_seed).help("seed random number generator"));
  options.add_and_parse(rand_options);
  all.get_random_state()->set_random_state(all.random_seed);

  parse_feature_tweaks(options, all, interactions_settings_duplicated, dictionary_nses);  // feature tweaks

  parse_example_tweaks(options, all);  // example manipulation

  parse_output_model(options, all);

  parse_output_preds(options, all);

  // create reduction stack builder instance
  all.learner_builder = VW::make_unique<VW::default_reduction_stack_setup>(all);
  // kick-off reduction setup functions
  all.l = all.learner_builder->operator()(options, all);

  if (!all.logger.quiet)
  {
    *(all.trace_message) << "Num weight bits = " << all.num_bits << endl;
    *(all.trace_message) << "learning rate = " << all.eta << endl;
    *(all.trace_message) << "initial_t = " << all.sd->t << endl;
    *(all.trace_message) << "power_t = " << all.power_t << endl;
    if (all.numpasses > 1) *(all.trace_message) << "decay_learning_rate = " << all.eta_decay_rate << endl;
  }
}

void parse_sources(options_i& options, vw& all, io_buf& model, bool skipModelLoad)
{
  if (!skipModelLoad)
    load_input_model(all, model);
  else
    model.close_file();

  auto parsed_source_options = parse_source(all, options);
  enable_sources(all, all.logger.quiet, all.numpasses, parsed_source_options);

  // force wpp to be a power of 2 to avoid 32-bit overflow
  uint32_t i = 0;
  size_t params_per_problem = all.l->increment;
  while (params_per_problem > (static_cast<uint64_t>(1) << i)) i++;
  all.wpp = (1 << i) >> all.weights.stride_shift();
}

namespace VW
{
void cmd_string_replace_value(std::stringstream*& ss, std::string flag_to_replace, std::string new_value)
{
  flag_to_replace.append(
      " ");  // add a space to make sure we obtain the right flag in case 2 flags start with the same set of characters
  std::string cmd = ss->str();
  size_t pos = cmd.find(flag_to_replace);
  if (pos == std::string::npos)
    // flag currently not present in command string, so just append it to command string
    *ss << " " << flag_to_replace << new_value;
  else
  {
    // flag is present, need to replace old value with new value

    // compute position after flag_to_replace
    pos += flag_to_replace.size();

    // now pos is position where value starts
    // find position of next space
    size_t pos_after_value = cmd.find(" ", pos);
    if (pos_after_value == std::string::npos)
    {
      // we reach the end of the std::string, so replace the all characters after pos by new_value
      cmd.replace(pos, cmd.size() - pos, new_value);
    }
    else
    {
      // replace characters between pos and pos_after_value by new_value
      cmd.replace(pos, pos_after_value - pos, new_value);
    }

    ss->str(cmd);
  }
}

char** to_argv_escaped(std::string const& s, int& argc)
{
  std::vector<std::string> tokens = escaped_tokenize(' ', s);
  char** argv = calloc_or_throw<char*>(tokens.size() + 1);
  argv[0] = calloc_or_throw<char>(2);
  argv[0][0] = 'b';
  argv[0][1] = '\0';

  for (size_t i = 0; i < tokens.size(); i++)
  {
    argv[i + 1] = calloc_or_throw<char>(tokens[i].length() + 1);
    sprintf_s(argv[i + 1], (tokens[i].length() + 1), "%s", tokens[i].data());
  }

  argc = static_cast<int>(tokens.size() + 1);
  return argv;
}

char** to_argv(std::string const& s, int& argc)
{
  VW::string_view strview(s);
  std::vector<VW::string_view> foo;
  tokenize(' ', strview, foo);

  char** argv = calloc_or_throw<char*>(foo.size() + 1);
  // small optimization to avoid a string copy before tokenizing
  argv[0] = calloc_or_throw<char>(2);
  argv[0][0] = 'b';
  argv[0][1] = '\0';
  for (size_t i = 0; i < foo.size(); i++)
  {
    size_t len = foo[i].length();
    argv[i + 1] = calloc_or_throw<char>(len + 1);
    memcpy(argv[i + 1], foo[i].data(), len);
    // copy() is supported with boost::string_view, not with string_ref
    // foo[i].copy(argv[i], len);
    // unnecessary because of the calloc, but needed if we change stuff in the future
    // argv[i][len] = '\0';
  }

  argc = static_cast<int>(foo.size()) + 1;
  return argv;
}

char** get_argv_from_string(std::string s, int& argc) { return to_argv(s, argc); }

void free_args(int argc, char* argv[])
{
  for (int i = 0; i < argc; i++) free(argv[i]);
  free(argv);
}

vw* initialize(
    config::options_i& options, io_buf* model, bool skipModelLoad, trace_message_t trace_listener, void* trace_context)
{
  std::unique_ptr<options_i, options_deleter_type> opts(&options, [](VW::config::options_i*) {});

  return initialize(std::move(opts), model, skipModelLoad, trace_listener, trace_context);
}

vw* initialize(std::unique_ptr<options_i, options_deleter_type> options, io_buf* model, bool skipModelLoad,
    trace_message_t trace_listener, void* trace_context)
{
  // Set up logger as early as possible
  logger::initialize_logger();
  vw& all = parse_args(std::move(options), trace_listener, trace_context);

  try
  {
    // if user doesn't pass in a model, read from options
    io_buf localModel;
    if (!model)
    {
      std::vector<std::string> all_initial_regressor_files(all.initial_regressors);
      if (all.options->was_supplied("input_feature_regularizer"))
      { all_initial_regressor_files.push_back(all.per_feature_regularizer_input); }
      read_regressor_file(all, all_initial_regressor_files, localModel);
      model = &localModel;
    }

    // Loads header of model files and loads the command line options into the options object.
    bool interactions_settings_duplicated;
    load_header_merge_options(*all.options.get(), all, *model, interactions_settings_duplicated);

    std::vector<std::string> dictionary_nses;
    parse_modules(*all.options.get(), all, interactions_settings_duplicated, dictionary_nses);
    parse_sources(*all.options.get(), all, *model, skipModelLoad);

    // we must delay so parse_mask is fully defined.
    for (size_t id = 0; id < dictionary_nses.size(); id++) parse_dictionary_argument(all, dictionary_nses[id]);

    all.options->check_unregistered();

    // upon direct query for help -- spit it out to stdout;
    if (all.options->get_typed_option<bool>("help").value())
    {
      if (all.learner_builder) { cout << all.options->help(all.learner_builder->enabled_reductions); }
      else
      {
        cout << all.options->help({});
      }
      exit(0);
    }

    // output list of enabled reductions
    if (all.learner_builder)
    {
      if (!all.logger.quiet && !all.options->was_supplied("audit_regressor"))
      { all.learner_builder->print_enabled_reductions(all); }
    }

    if (!all.options->get_typed_option<bool>("dry_run").value())
    {
      if (!all.logger.quiet && !all.bfgs && !all.searchstr && !all.options->was_supplied("audit_regressor"))
      { all.sd->print_update_header(*all.trace_message); }
      all.l->init_driver();
    }

    return &all;
  }
  catch (std::exception& e)
  {
    *(all.trace_message) << "Error: " << e.what() << endl;
    finish(all);
    throw;
  }
  catch (...)
  {
    finish(all);
    throw;
  }
}

vw* initialize(std::string s, io_buf* model, bool skipModelLoad, trace_message_t trace_listener, void* trace_context)
{
  int argc = 0;
  char** argv = to_argv(s, argc);
  vw* ret = nullptr;

  try
  {
    ret = initialize(argc, argv, model, skipModelLoad, trace_listener, trace_context);
  }
  catch (...)
  {
    free_args(argc, argv);
    throw;
  }

  free_args(argc, argv);
  return ret;
}

vw* initialize_escaped(
    std::string const& s, io_buf* model, bool skipModelLoad, trace_message_t trace_listener, void* trace_context)
{
  int argc = 0;
  char** argv = to_argv_escaped(s, argc);
  vw* ret = nullptr;

  try
  {
    ret = initialize(argc, argv, model, skipModelLoad, trace_listener, trace_context);
  }
  catch (...)
  {
    free_args(argc, argv);
    throw;
  }

  free_args(argc, argv);
  return ret;
}

vw* initialize(
    int argc, char* argv[], io_buf* model, bool skipModelLoad, trace_message_t trace_listener, void* trace_context)
{
  std::unique_ptr<options_i, options_deleter_type> options(
      new config::options_boost_po(argc, argv), [](VW::config::options_i* ptr) { delete ptr; });
  return initialize(std::move(options), model, skipModelLoad, trace_listener, trace_context);
}

// Create a new VW instance while sharing the model with another instance
// The extra arguments will be appended to those of the other VW instance
vw* seed_vw_model(vw* vw_model, const std::string extra_args, trace_message_t trace_listener, void* trace_context)
{
  options_serializer_boost_po serializer;
  for (auto const& option : vw_model->options->get_all_options())
  {
    if (vw_model->options->was_supplied(option->m_name))
    {
      // ignore no_stdin since it will be added by vw::initialize, and ignore -i since we don't want to reload the
      // model.
      if (option->m_name == "no_stdin" || option->m_name == "initial_regressor") { continue; }

      serializer.add(*option);
    }
  }

  auto serialized_options = serializer.str();
  serialized_options = serialized_options + " " + extra_args;

  vw* new_model =
      VW::initialize(serialized_options.c_str(), nullptr, true /* skipModelLoad */, trace_listener, trace_context);
  free_it(new_model->sd);

  // reference model states stored in the specified VW instance
  new_model->weights.shallow_copy(vw_model->weights);  // regressor
  new_model->sd = vw_model->sd;                        // shared data
  new_model->example_parser->_shared_data = new_model->sd;

  return new_model;
}

void sync_stats(vw& all)
{
  if (all.all_reduce != nullptr)
  {
    float loss = static_cast<float>(all.sd->sum_loss);
    all.sd->sum_loss = static_cast<double>(accumulate_scalar(all, loss));
    float weighted_labeled_examples = static_cast<float>(all.sd->weighted_labeled_examples);
    all.sd->weighted_labeled_examples = static_cast<double>(accumulate_scalar(all, weighted_labeled_examples));
    float weighted_labels = static_cast<float>(all.sd->weighted_labels);
    all.sd->weighted_labels = static_cast<double>(accumulate_scalar(all, weighted_labels));
    float weighted_unlabeled_examples = static_cast<float>(all.sd->weighted_unlabeled_examples);
    all.sd->weighted_unlabeled_examples = static_cast<double>(accumulate_scalar(all, weighted_unlabeled_examples));
    float example_number = static_cast<float>(all.sd->example_number);
    all.sd->example_number = static_cast<uint64_t>(accumulate_scalar(all, example_number));
    float total_features = static_cast<float>(all.sd->total_features);
    all.sd->total_features = static_cast<uint64_t>(accumulate_scalar(all, total_features));
  }
}

void finish(vw& all, bool delete_all)
{
  // also update VowpalWabbit::PerformanceStatistics::get() (vowpalwabbit.cpp)
  if (!all.logger.quiet && !all.options->was_supplied("audit_regressor"))
  {
    all.trace_message->precision(6);
    *(all.trace_message) << std::fixed;
    *(all.trace_message) << endl << "finished run";
    if (all.current_pass == 0 || all.current_pass == 1)
      *(all.trace_message) << endl << "number of examples = " << all.sd->example_number;
    else
    {
      *(all.trace_message) << endl << "number of examples per pass = " << all.sd->example_number / all.current_pass;
      *(all.trace_message) << endl << "passes used = " << all.current_pass;
    }
    *(all.trace_message) << endl << "weighted example sum = " << all.sd->weighted_examples();
    *(all.trace_message) << endl << "weighted label sum = " << all.sd->weighted_labels;
    *(all.trace_message) << endl << "average loss = ";
    if (all.holdout_set_off)
      if (all.sd->weighted_labeled_examples > 0)
        *(all.trace_message) << all.sd->sum_loss / all.sd->weighted_labeled_examples;
      else
        *(all.trace_message) << "n.a.";
    else if ((all.sd->holdout_best_loss == FLT_MAX) || (all.sd->holdout_best_loss == FLT_MAX * 0.5))
      *(all.trace_message) << "undefined (no holdout)";
    else
      *(all.trace_message) << all.sd->holdout_best_loss << " h";
    if (all.sd->report_multiclass_log_loss)
    {
      if (all.holdout_set_off)
        *(all.trace_message) << endl
                             << "average multiclass log loss = "
                             << all.sd->multiclass_log_loss / all.sd->weighted_labeled_examples;
      else
        *(all.trace_message) << endl
                             << "average multiclass log loss = "
                             << all.sd->holdout_multiclass_log_loss / all.sd->weighted_labeled_examples << " h";
    }

    float best_constant;
    float best_constant_loss;
    if (get_best_constant(all, best_constant, best_constant_loss))
    {
      *(all.trace_message) << endl << "best constant = " << best_constant;
      if (best_constant_loss != FLT_MIN)
        *(all.trace_message) << endl << "best constant's loss = " << best_constant_loss;
    }

    *(all.trace_message) << endl << "total feature number = " << all.sd->total_features;
    if (all.sd->queries > 0) *(all.trace_message) << endl << "total queries = " << all.sd->queries;
    *(all.trace_message) << endl;
  }

  // implement finally.
  // finalize_regressor can throw if it can't write the file.
  // we still want to free up all the memory.
  vw_exception finalize_regressor_exception(__FILE__, __LINE__, "empty");
  bool finalize_regressor_exception_thrown = false;
  try
  {
    finalize_regressor(all, all.final_regressor_name);
  }
  catch (vw_exception& e)
  {
    finalize_regressor_exception = e;
    finalize_regressor_exception_thrown = true;
  }

  metrics::output_metrics(all);
  logger::log_summary();

  if (delete_all) delete &all;

  if (finalize_regressor_exception_thrown) throw finalize_regressor_exception;
}
}  // namespace VW
