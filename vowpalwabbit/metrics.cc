// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "debug_log.h"
#include "reductions.h"
#include "learner.h"
#ifdef BUILD_EXTERNAL_PARSER
#  include "parse_example_external.h"
#endif
#include <rapidjson/filewritestream.h>
#include <rapidjson/writer.h>
#include <cfloat>

#include "io/logger.h"

using namespace VW::config;
using namespace VW::LEARNER;
using namespace rapidjson;

namespace logger = VW::io::logger;

namespace VW
{
namespace metrics
{
struct metrics_data
{
  std::string out_file;
  size_t learn_count = 0;
  size_t predict_count = 0;
};

void list_to_json_file(dsjson_metrics* ds_metrics, std::string filename, metric_sink& metrics)
{
  FILE* fp;

  if (VW::file_open(&fp, filename.c_str(), "wt") == 0)
  {
    char writeBuffer[1024];
    FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);

    writer.StartObject();
    for (const auto& m : metrics.int_metrics_list)
    {
      writer.Key(m.first.c_str());
      writer.Uint64(m.second);
    }
    for (const auto& m : metrics.float_metrics_list)
    {
      writer.Key(m.first.c_str());
      writer.Double(static_cast<double>(m.second));
    }

    // ds_metrics is nullptr when --dsjson is disabled
    if (ds_metrics)
    {
      writer.Key("number_skipped_events");
      writer.Int64(ds_metrics->NumberOfSkippedEvents);
      writer.Key("number_events_zero_actions");
      writer.Int64(ds_metrics->NumberOfEventsZeroActions);
      writer.Key("line_parse_error");
      writer.Int64(ds_metrics->LineParseError);
      writer.Key("first_event_id");
      writer.String(ds_metrics->FirstEventId.c_str());
      writer.Key("first_event_time");
      writer.String(ds_metrics->FirstEventTime.c_str());
      writer.Key("last_event_id");
      writer.String(ds_metrics->LastEventId.c_str());
      writer.Key("last_event_time");
      writer.String(ds_metrics->LastEventTime.c_str());
    }

    writer.EndObject();

    fclose(fp);
  }
  else
  {
    logger::errlog_warn("skipping metrics. could not open file for metrics: {}", filename);
  }
}

void output_metrics(vw& all)
{
  if (all.options->was_supplied("extra_metrics"))
  {
    std::string filename = all.options->get_typed_option<std::string>("extra_metrics").value();
    metric_sink list_metrics;

    all.l->persist_metrics(list_metrics);

#ifdef BUILD_EXTERNAL_PARSER
    if (all.external_parser) { all.external_parser->persist_metrics(list_metrics.int_metrics_list); }
#endif

    list_metrics.int_metrics_list.emplace_back("total_log_calls", logger::get_log_count());

    list_to_json_file(all.example_parser->metrics.get(), filename, list_metrics);
  }
}

template <bool is_learn, typename T, typename E>
void predict_or_learn(metrics_data& data, T& base, E& ec)
{
  if (is_learn)
  {
    data.learn_count++;
    base.learn(ec);
  }
  else
  {
    data.predict_count++;
    base.predict(ec);
  }
}

void persist(metrics_data& data, metric_sink& metrics)
{
  metrics.int_metrics_list.emplace_back("total_predict_calls", data.predict_count);
  metrics.int_metrics_list.emplace_back("total_learn_calls", data.learn_count);
}

VW::LEARNER::base_learner* metrics_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<metrics_data>();

  option_group_definition new_options("Debug: Metrics");
  new_options.add(make_option("extra_metrics", data->out_file)
                      .necessary()
                      .help("Specify filename to write metrics to. Note: There is no fixed schema."));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (data->out_file.empty()) THROW("extra_metrics argument (output filename) is missing.");

  auto* base_learner = setup_base(options, all);

  if (base_learner->is_multiline)
  {
    learner<metrics_data, multi_ex>* l = &init_learner(data, as_multiline(base_learner),
        predict_or_learn<true, multi_learner, multi_ex>, predict_or_learn<false, multi_learner, multi_ex>, 1,
        base_learner->pred_type, all.get_setupfn_name(metrics_setup), base_learner->learn_returns_prediction);
    l->set_persist_metrics(persist);
    return make_base(*l);
  }
  else
  {
    learner<metrics_data, example>* l = &init_learner(data, as_singleline(base_learner),
        predict_or_learn<true, single_learner, example>, predict_or_learn<false, single_learner, example>, 1,
        base_learner->pred_type, all.get_setupfn_name(metrics_setup), base_learner->learn_returns_prediction);
    l->set_persist_metrics(persist);
    return make_base(*l);
  }
}

}  // namespace metrics
}  // namespace VW
