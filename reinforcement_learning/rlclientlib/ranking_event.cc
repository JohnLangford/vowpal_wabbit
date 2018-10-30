#include "action_flags.h"
#include "ranking_event.h"
#include "utility/data_buffer.h"

#include "explore_internal.h"
#include "hash.h"

#include <sstream>
#include <iomanip>

using namespace std;

namespace reinforcement_learning {
  namespace u = utility;

  event::event()
  {}

  event::event(const char* event_id, float pass_prob)
    : _event_id(event_id)
    , _pass_prob(pass_prob)
  {}

  event::event(event&& other)
    : _event_id(std::move(other._event_id))
    , _pass_prob(other._pass_prob)
  {}

  event& event::operator=(event&& other) {
    if (&other != this) {
      _event_id = std::move(other._event_id);
      _pass_prob = other._pass_prob;
    }
    return *this;
  }

  event::~event() {}

  bool event::try_drop(float pass_prob, int drop_pass) {
    _pass_prob *= pass_prob;
    return prg(drop_pass) > pass_prob;
  }

  float event::prg(int drop_pass) const {
    const auto seed_str = _event_id + std::to_string(drop_pass);
    const auto seed = uniform_hash(seed_str.c_str(), seed_str.length(), 0);
    return exploration::uniform_random_merand48(seed);
  }

  ranking_event::ranking_event()
  { }

  ranking_event::ranking_event(const char* event_id, bool deferred_action, float pass_prob, const char* context, const ranking_response& response)
    : event(event_id, pass_prob)
    , _deferred_action(deferred_action)
    , _context(context)
    , _model_id(response.get_model_id())
  {
    for (auto const &r : response) {
      _a_vector.push_back(r.action_id + 1);
      _p_vector.push_back(r.probability);
    }
  }

  ranking_event::ranking_event(ranking_event&& other)
    : event(std::move(other))
    , _deferred_action(other._deferred_action)
    , _context(std::move(other._context))
    , _a_vector(std::move(other._a_vector))
    , _p_vector(std::move(other._p_vector))
    , _model_id(std::move(other._model_id))
  { }

  ranking_event& ranking_event::operator=(ranking_event&& other) {
    if (&other != this) {
      event::operator=(std::move(other));
      _deferred_action = std::move(other._deferred_action);
      _context = std::move(other._context);
      _a_vector = std::move(other._a_vector);
      _p_vector = std::move(other._p_vector);
      _model_id = std::move(other._model_id);
    }
    return *this;
  }

  flatbuffers::Offset<RankingEvent> ranking_event::serialize_eventhub_message(flatbuffers::FlatBufferBuilder& builder) {
    short version = 1;
    auto event_id_offset = builder.CreateString(_event_id);

    auto a_vector_offset = builder.CreateVector(_a_vector);
    auto p_vector_offset = builder.CreateVector(_p_vector);

    auto context_offset = builder.CreateString(_context);

    auto vw_state_offset = VW::Events::CreateVWStateType(builder, builder.CreateString(_model_id));

    return VW::Events::CreateRankingEvent(builder, version, event_id_offset, _deferred_action, a_vector_offset, context_offset, p_vector_offset, vw_state_offset);
  }

  ranking_event ranking_event::choose_rank(const char* event_id, const char* context,
    unsigned int flags, const ranking_response& resp, float pass_prob) {
    return ranking_event(event_id, flags & action_flags::DEFERRED, pass_prob, context, resp);
  }

  std::string ranking_event::str() {
    u::data_buffer oss;

    oss << R"({"Version":"1","EventId":")" << _event_id << R"(")";

    if (_deferred_action) {
      oss << R"(,"DeferredAction":true)";
    }

    //add action ids
    oss << R"(,"a":[)";
    for (auto id : _a_vector) {
      oss << id + 1 << ",";
    }
    //remove trailing ,
    oss.remove_last();

    //add probabilities
    oss << R"(],"c":)" << _context << R"(,"p":[)";
    for (auto prob : _p_vector) {
      oss << prob << ",";
    }
    //remove trailing ,
    oss.remove_last();

    //add model id
    oss << R"(],"VWState":{"m":")" << _model_id << R"("})";

    return oss.str();
  }

  outcome_event::outcome_event()
  { }

  outcome_event::outcome_event(const char* event_id, float pass_prob, const char* outcome)
    : event(event_id, pass_prob)
    , _outcome(outcome)
  {
  }

  outcome_event::outcome_event(const char* event_id, float pass_prob, float outcome)
    : event(event_id, pass_prob)
  {
    _outcome = std::to_string(outcome);
  }

  outcome_event::outcome_event(outcome_event&& other)
    : event(std::move(other))
    , _outcome(std::move(other._outcome))
  { }

  outcome_event& outcome_event::operator=(outcome_event&& other) {
    if (&other != this) {
      event::operator=(std::move(other));
      _outcome = std::move(other._outcome);
    }
    return *this;
  }

  std::string outcome_event::str() {
    u::data_buffer oss;
    oss << R"({"EventId":")" << _event_id << R"(","v":)" << _outcome << R"(})";
    return oss.str();
  }

  flatbuffers::Offset<OutcomeEvent> outcome_event::serialize_eventhub_message(flatbuffers::FlatBufferBuilder& builder) {
    auto event_id_offset = builder.CreateString(_event_id);
    auto outcome_offset = builder.CreateString(_outcome);
    return VW::Events::CreateOutcomeEvent(builder, event_id_offset, outcome_offset);
  }

  outcome_event outcome_event::report_outcome(const char* event_id, const char* outcome, float pass_prob) {
    return outcome_event(event_id, pass_prob, outcome);
  }

  outcome_event outcome_event::report_outcome(const char* event_id, float outcome, float pass_prob) {
    return outcome_event(event_id, pass_prob, outcome);
  }

  outcome_event outcome_event::report_action_taken(const char* event_id, float pass_prob) {
    return outcome_event(event_id, pass_prob, "");
  }
}
