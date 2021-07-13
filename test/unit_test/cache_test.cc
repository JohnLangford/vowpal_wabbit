#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "cache.h"
#include "vw.h"
#include "test_common.h"

BOOST_AUTO_TEST_CASE(write_and_read_features_from_cache)
{
  auto& vw = *VW::initialize("--quiet");
  example src_ex;
  VW::read_line(vw, &src_ex, "|ns1 example value test |ss2 ex:0.5");

  auto backing_vector = std::make_shared<std::vector<char>>();
  io_buf io_writer;
  io_writer.add_file(VW::io::create_vector_writer(backing_vector));

  VW::write_example_to_cache(io_writer, &src_ex, vw.example_parser->lbl_parser, vw.parse_mask);

  io_writer.flush();

  io_buf io_reader;
  io_reader.add_file(VW::io::create_buffer_view(backing_vector->data(), backing_vector->size()));

  example dest_ex;
  VW::read_example_from_cache(io_reader, &dest_ex, vw.example_parser->lbl_parser, vw.example_parser->sorted_cache,
      vw.example_parser->_shared_data);

  BOOST_CHECK_EQUAL(dest_ex.indices.size(), 2);
  BOOST_CHECK_EQUAL(dest_ex.feature_space.get('n', VW::hash_space(vw, "ns1")).size(), 3);
  BOOST_CHECK_EQUAL(dest_ex.feature_space.get('s', VW::hash_space(vw, "ss2")).size(), 1);

  check_collections_with_float_tolerance(src_ex.feature_space.get('s', VW::hash_space(vw, "ss2")).values,
      dest_ex.feature_space.get('s', VW::hash_space(vw, "ss2")).values, FLOAT_TOL);
  check_collections_exact(src_ex.feature_space.get('s', VW::hash_space(vw, "ss2")).indicies,
      dest_ex.feature_space.get('s', VW::hash_space(vw, "ss2")).indicies);
  check_collections_with_float_tolerance(src_ex.feature_space.get('n', VW::hash_space(vw, "ns1")).values,
      dest_ex.feature_space.get('n', VW::hash_space(vw, "ns1")).values, FLOAT_TOL);
  check_collections_exact(src_ex.feature_space.get('n', VW::hash_space(vw, "ns1")).indicies,
      dest_ex.feature_space.get('n', VW::hash_space(vw, "ns1")).indicies);

  VW::finish(vw);
}
