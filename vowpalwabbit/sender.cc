// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <vector>
#ifdef _WIN32
#  define NOMINMAX
#  include <WinSock2.h>
#  ifndef SHUT_RD
#    define SHUT_RD SD_RECEIVE
#  endif

#  ifndef SHUT_WR
#    define SHUT_WR SD_SEND
#  endif

#  ifndef SHUT_RDWR
#    define SHUT_RDWR SD_BOTH
#  endif
#else
#  include <netdb.h>
#endif

#include "io_buf.h"
#include "cache.h"
#include "network.h"
#include "reductions.h"

using namespace VW::config;

struct sender
{
  io_buf* buf;
  std::unique_ptr<VW::io::socket> _socket;
  std::unique_ptr<VW::io::reader> _socket_reader;
  vw* all;  // loss ring_size others
  example** delay_ring;
  size_t sent_index;
  size_t received_index;

  ~sender()
  {
    free(delay_ring);
    delete buf;
  }
};

void open_sockets(sender& s, std::string host)
{
  s._socket = VW::io::wrap_socket_descriptor(open_socket(host.c_str()));
  s._socket_reader = s._socket->get_reader();
  s.buf = new io_buf();
  s.buf->add_file(s._socket->get_writer());
}

void send_features(io_buf* b, example& ec, uint32_t mask)
{
  // note: subtracting 1 b/c not sending constant
  output_byte(*b, static_cast<uint64_t>(ec.feature_space.size() - 1));

  for (auto& bucket : ec.feature_space)
  {
    for (auto it = bucket.begin(); it != bucket.end(); ++it)
    { output_features(*b, it->index, it->hash, it->feats, mask); }
  }

  b->flush();
}

void receive_result(sender& s)
{
  float res, weight;

  get_prediction(s._socket_reader.get(), res, weight);
  example& ec = *s.delay_ring[s.received_index++ % s.all->example_parser->ring_size];
  ec.pred.scalar = res;

  label_data& ld = ec.l.simple;
  ec.loss = s.all->loss->getLoss(s.all->sd, ec.pred.scalar, ld.label) * ec.weight;

  return_simple_example(*(s.all), nullptr, ec);
}

void learn(sender& s, VW::LEARNER::single_learner&, example& ec)
{
  if (s.received_index + s.all->example_parser->ring_size / 2 - 1 == s.sent_index) receive_result(s);

  s.all->set_minmax(s.all->sd, ec.l.simple.label);
  s.all->example_parser->lbl_parser.cache_label(&ec.l, ec._reduction_features, *s.buf);  // send label information.
  cache_tag(*s.buf, ec.tag);
  send_features(s.buf, ec, static_cast<uint32_t>(s.all->parse_mask));
  s.delay_ring[s.sent_index++ % s.all->example_parser->ring_size] = &ec;
}

void finish_example(vw&, sender&, example&) {}

void end_examples(sender& s)
{
  // close our outputs to signal finishing.
  while (s.received_index != s.sent_index) receive_result(s);
  s.buf->close_files();
}

VW::LEARNER::base_learner* sender_setup(options_i& options, vw& all)
{
  std::string host;

  option_group_definition sender_options("Network sending");
  sender_options.add(make_option("sendto", host).keep().necessary().help("send examples to <host>"));

  if (!options.add_parse_and_check_necessary(sender_options)) { return nullptr; }

  auto s = scoped_calloc_or_throw<sender>();
  open_sockets(*s.get(), host);

  s->all = &all;
  s->delay_ring = calloc_or_throw<example*>(all.example_parser->ring_size);

  VW::LEARNER::learner<sender, example>& l = init_learner(s, learn, learn, 1, all.get_setupfn_name(sender_setup));
  l.set_finish_example(finish_example);
  l.set_end_examples(end_examples);
  return make_base(l);
}
