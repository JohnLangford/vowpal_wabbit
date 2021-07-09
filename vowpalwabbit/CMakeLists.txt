configure_file(config.h.in config.h)

# Use position independent code for all targets in this directory
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(vw_io STATIC io/io_adapter.h io/io_adapter.cc io/logger.h io/logger.cc)
target_include_directories(vw_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(vw_io PUBLIC ${spdlog_target} fmt::fmt PRIVATE ZLIB::ZLIB)

if(SPDLOG_SYS_DEP)
  # this doesn't get defined when using a system-installed spdlog
  target_compile_definitions(vw_io PUBLIC SPDLOG_FMT_EXTERNAL)
endif()

add_library(VowpalWabbit::io ALIAS vw_io)

add_library(allreduce STATIC allreduce_sockets.cc allreduce_threads.cc vw_exception.cc)
target_include_directories(allreduce PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

add_library(VowpalWabbit::allreduce ALIAS allreduce)

target_link_libraries(allreduce PRIVATE vw_io)

# Winsock32 should be available on Windows
if(WIN32)
  target_link_libraries(allreduce PUBLIC wsock32 ws2_32)
else()
  target_compile_options(allreduce PUBLIC ${linux_flags})
endif()


if(BUILD_FLATBUFFERS)
  add_subdirectory(parser/flatbuffer)
  add_library(FlatbuffersTarget INTERFACE)
  target_include_directories(FlatbuffersTarget INTERFACE ${FLATBUFFERS_INCLUDE_DIR})
  add_dependencies(FlatbuffersTarget fbschemas)
endif()

set(vw_all_headers
  accumulate.h
  action_score.h
  active_cover.h
  active.h
  allreduce.h
  api_status.h
  array_parameters_dense.h
  array_parameters.h
  audit_regressor.h
  autolink.h
  baseline.h
  beam.h
  best_constant.h
  bfgs.h
  binary.h
  boosting.h
  bs.h
  cache.h
  cats_pdf.h
  cats_tree.h
  cats.h
  cb_adf.h
  cb_algs.h
  cb_continuous_label.h
  cb_dro.h
  cb_explore_adf_bag.h
  cb_explore_adf_common.h
  cb_explore_adf_cover.h
  cb_explore_adf_first.h
  cb_explore_adf_greedy.h
  cb_explore_adf_regcb.h
  cb_explore_adf_squarecb.h
  cb_explore_adf_synthcover.h
  cb_explore_adf_rnd.h
  cb_explore_adf_softmax.h
  cb_explore_pdf.h
  cb_explore.h
  cb_label_parser.h
  cb.h
  cbify.h
  cb_to_cb_adf.h
  ccb_label.h
  ccb_reduction_features.h
  chained_proxy_iterator.h
  continuous_actions_reduction_features.h
  classweight.h
  compat.h
  conditional_contextual_bandit.h
  confidence.h
  config.h.in
  constant.h
  cbzo.h
  correctedMath.h
  cost_sensitive.h
  crossplat_compat.h
  cs_active.h
  csoaa.h
  debug_print.h
  decision_scores.h
  distributionally_robust.h
  ect.h
  error_constants.h
  error_data.h
  error_reporting.h
  example_predict.h
  example.h
  explore_eval.h
  expreplay.h
  ezexample.h
  fast_pow10.h
  feature_group.h
  ftrl.h
  gd_mf.h
  gd_predict.h
  gd.h
  gen_cs_example.h
  generate_interactions.h
  get_pmf.h
  global_data.h
  guard.h
  hashstring.h
  interact.h
  interactions_predict.h
  interactions.h
  io_buf.h
  json_utils.h
  kernel_svm.h
  label_dictionary.h
  label_parser.h
  lda_core.h
  learner.h
  log_multi.h
  loss_functions.h
  lrq.h
  lrqfa.h
  marginal.h
  memory_tree.h
  memory.h
  metrics.h
  metric_sink.h
  mf.h
  multiclass.h
  multilabel_oaa.h
  multilabel.h
  mwt.h
  namespaced_features.h
  numeric_casts.h
  named_labels.h
  network.h
  kskip_ngram_transformer.h
  nn.h
  no_label.h
  noop.h
  oaa.h
  object_pool.h
  offset_tree.h
  OjaNewton.h
  options_boost_po.h
  options_serializer_boost_po.h
  options_types.h
  options.h
  parse_args.h
  parse_dispatch_loop.h
  parse_example_json.h
  parse_example.h
  parse_primitives.h
  parse_regressor.h
  parse_slates_example_json.h
  parser.h
  pmf_to_pdf.h
  plt.h
  reduction_features.h
  print.h
  prob_dist_cont.h
  queue.h
  rand48.h
  recall_tree.h
  reductions.h
  reductions_fwd.h
  sample_pdf.h
  scope_exit.h
  scorer.h
  search_dep_parser.h
  search_entityrelationtask.h
  search_graph.h
  search_hooktask.h
  search_meta.h
  search_multiclasstask.h
  search_sequencetask.h
  search.h
  sender.h
  shared_data.h
  shared_feature_merger.h
  simple_label_parser.h
  simple_label.h
  slates_label.h
  slates.h
  spanning_tree.h
  stable_unique.h
  stagewise_poly.h
  svrg.h
  tag_utils.h
  topk.h
  unique_sort.h
  v_array.h
  version.h
  vw_allreduce.h
  vw_exception.h
  vw_math.h
  vw_string_view.h
  vw_validate.h
  vw_versions.h
  vw.h
  vwdll.h
  vwvis.h
  warm_cb.h
  red_python.h
  generic_range.h
  active_multiclass_prediction.h
  debug_log.h
  rand_state.h
)

if(BUILD_FLATBUFFERS)
  set(vw_all_headers ${vw_all_headers} parser/flatbuffer/parse_example_flatbuffer.h)
endif()

if(BUILD_EXTERNAL_PARSER)
  set(vw_all_headers ${vw_all_headers} ${external_parser_headers})
endif()

set(vw_all_sources
  accumulate.cc
  action_score.cc
  active_cover.cc
  active.cc
  api_status.cc
  audit_regressor.cc
  autolink.cc
  baseline.cc
  best_constant.cc
  bfgs.cc
  binary.cc
  boosting.cc
  bs.cc
  cache.cc
  cats_pdf.cc
  cats_tree.cc
  cats.cc
  cb_adf.cc
  cb_algs.cc
  cb_continuous_label.cc
  cb_dro.cc
  cb_explore_adf_bag.cc
  cb_explore_adf_cover.cc
  cb_explore_adf_first.cc
  cb_explore_adf_greedy.cc
  cb_explore_adf_regcb.cc
  cb_explore_adf_squarecb.cc
  cb_explore_adf_synthcover.cc
  cb_explore_adf_rnd.cc
  cb_explore_adf_softmax.cc
  cb_explore_pdf.cc
  cb_explore.cc
  cb_sample.cc
  cb.cc
  cbify.cc
  cb_to_cb_adf.cc
  ccb_label.cc
  classweight.cc
  conditional_contextual_bandit.cc
  confidence.cc
  cbzo.cc
  cost_sensitive.cc
  cs_active.cc
  csoaa.cc
  decision_scores.cc
  distributionally_robust.cc
  ect.cc
  example_predict.cc
  example.cc
  explore_eval.cc
  feature_group.cc
  ftrl.cc
  gd_mf.cc
  gd.cc
  gen_cs_example.cc
  generate_interactions.cc
  get_pmf.cc
  global_data.cc
  hashstring.cc
  interact.cc
  interactions.cc
  io_buf.cc
  kernel_svm.cc
  label_dictionary.cc
  lda_core.cc
  learner.cc
  log_multi.cc
  loss_functions.cc
  lrq.cc
  lrqfa.cc
  marginal.cc
  memory_tree.cc
  metrics.cc
  mf.cc
  multiclass.cc
  multilabel_oaa.cc
  multilabel.cc
  mwt.cc
  namespaced_features.cc
  named_labels.cc
  network.cc
  kskip_ngram_transformer.cc
  nn.cc
  no_label.cc
  noop.cc
  oaa.cc
  offset_tree.cc
  OjaNewton.cc
  options_boost_po.cc
  options_serializer_boost_po.cc
  parse_args.cc
  parse_example.cc
  parse_primitives.cc
  parse_regressor.cc
  parser.cc
  pmf_to_pdf.cc
  plt.cc
  print.cc
  prob_dist_cont.cc
  rand48.cc
  recall_tree.cc
  sample_pdf.cc
  scorer.cc
  search_dep_parser.cc
  search_entityrelationtask.cc
  search_graph.cc
  search_hooktask.cc
  search_meta.cc
  search_multiclasstask.cc
  search_sequencetask.cc
  search.cc
  sender.cc
  shared_data.cc
  shared_feature_merger.cc
  simple_label_parser.cc
  simple_label.cc
  slates_label.cc
  slates.cc
  stagewise_poly.cc
  svrg.cc
  tag_utils.cc
  topk.cc
  unique_sort.cc
  version.cc
  vw_exception.cc
  vw_validate.cc
  warm_cb.cc
  red_python.cc
)

if(BUILD_FLATBUFFERS)
  set(vw_all_sources ${vw_all_sources}
    parser/flatbuffer/parse_example_flatbuffer.cc
    parser/flatbuffer/parse_label.cc)
endif()

if(BUILD_EXTERNAL_PARSER)
  set(vw_all_sources ${vw_all_sources} ${external_parser_sources})
endif()

if (vw_BUILD_NET_FRAMEWORK)
set(vw_all_sources ${vw_all_sources} spanning_tree.cc)
endif()

add_library(vw STATIC ${vw_all_sources} ${vw_all_headers})

target_link_libraries(vw
  PUBLIC
    VowpalWabbit::explore VowpalWabbit::allreduce Boost::boost ${spdlog_target} fmt::fmt
  PRIVATE
    Boost::program_options ${CMAKE_DL_LIBS} ${LINK_THREADS} vw_io
    # Workaround an issue where RapidJSON needed to be exported tom install the target. This is
    # actually a private dependency and so do not "link" when processing targets for installation.
    # https://gitlab.kitware.com/cmake/cmake/issues/15415
    $<BUILD_INTERFACE:RapidJSON>)

if (BUILD_EXTERNAL_PARSER)
  target_compile_definitions(vw PUBLIC BUILD_EXTERNAL_PARSER)
  target_include_directories(vw PRIVATE ${EXTERNAL_PARSER_DIR})
  target_link_libraries(vw PRIVATE ${EXTERNAL_PARSER_LIBS})
endif()

if(BUILD_FLATBUFFERS)
  target_link_libraries(vw
      PRIVATE
      $<BUILD_INTERFACE:FlatbuffersTarget>)
  target_compile_definitions(vw PUBLIC BUILD_FLATBUFFERS)
endif()


add_library(VowpalWabbit::vw ALIAS vw)

if(GCOV)
  target_link_libraries(vw PUBLIC gcov --coverage)
endif()

target_include_directories(vw PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(vw PUBLIC _FILE_OFFSET_BITS=64 $<$<CONFIG:RELEASE>:NDEBUG>)
# Only define if Clang is not used
if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_definitions(vw PUBLIC __extern_always_inline=inline)
endif()

# Add natvis file if the Generator is Visual Studio
if (MSVC_IDE)
  target_sources(vw PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/vw_types.natvis> )
endif()

# Clang-cl on Windows has issues with our usage of SIMD types. Turn it off explicitly for Windows + clang-cl to mitigate.
# See issue #
if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_definitions(vw PUBLIC VW_NO_INLINE_SIMD)
endif()

# TODO code analysis
if(WIN32)
  target_compile_definitions(vw PUBLIC __SSE2__)
  target_compile_options(vw PUBLIC /MP /Zc:__cplusplus)
else()
  target_compile_options(vw PUBLIC ${linux_flags})
endif()

# Turn on warnings
if(WARNINGS)
  if(MSVC)
    target_compile_options(vw PRIVATE /W4)
  else(MSVC)
    target_compile_options(vw PRIVATE -Wall -Wextra -Wpedantic)
  endif(MSVC)
endif(WARNINGS)

# Turn on warnings as errors
if(WARNING_AS_ERROR)
  if(MSVC)
    target_compile_options(vw PRIVATE /WX)
  else(MSVC)
    target_compile_options(vw PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif(MSVC)
endif(WARNING_AS_ERROR)

if(NOT WIN32)
  add_executable(active_interactor active_interactor.cc)

  if(GCOV)
    target_link_libraries(active_interactor PUBLIC gcov --coverage)
  endif()

  if(VW_INSTALL)
    install(
      TARGETS active_interactor
      EXPORT VowpalWabbitConfig
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

  target_compile_definitions(active_interactor PUBLIC _FILE_OFFSET_BITS=64)
  target_compile_definitions(active_interactor PUBLIC $<$<CONFIG:RELEASE>:NDEBUG>)
  target_compile_options(active_interactor PUBLIC ${linux_flags})

  find_file(HELP2MAN_EXECUTABLE help2man HINTS /bin /usr/bin /usr/local/bin)
  if (HELP2MAN_EXECUTABLE)
    add_custom_target(manpage ALL
      COMMAND ${HELP2MAN_EXECUTABLE}
        --no-info
        --name="Vowpal Wabbit -- fast online learning tool" $<TARGET_FILE:vw-bin>
        --output=$<TARGET_FILE:vw-bin>.1
      DEPENDS vw-bin)

      if(VW_INSTALL)
        install(FILES $<TARGET_FILE:vw-bin>.1 DESTINATION share/man/man1)
      endif()
  else()
    message(STATUS "help2man not found, please install it to generate manpages")
  endif()
endif()

# build main executable
add_executable(vw-bin main.cc)
target_link_libraries(vw-bin PRIVATE VowpalWabbit::vw Boost::program_options)
set_target_properties(vw-bin PROPERTIES OUTPUT_NAME vw)

if(STATIC_LINK_VW)
  target_link_libraries(vw-bin PRIVATE ${unix_static_flag})
endif()

# This is used by Travis to not build vw_c_wrapper, this required <codecvt> which became available in GCC 5.1
if(NOT DEFINED DO_NOT_BUILD_VW_C_WRAPPER)
  if(STATIC_LINK_VW)
    add_library(vw_c_wrapper vwdll.cpp)
  else()
    add_library(vw_c_wrapper SHARED vwdll.cpp)
  endif()

  add_library(VowpalWabbit::vw_c_wrapper ALIAS vw_c_wrapper)

  target_compile_definitions(vw_c_wrapper PUBLIC VWDLL_EXPORTS _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE ZLIB_WINAPI)
  target_link_libraries(vw_c_wrapper PUBLIC VowpalWabbit::vw)
  if(VW_INSTALL)
    install(
      TARGETS vw_c_wrapper
      EXPORT VowpalWabbitConfig
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if(VW_INSTALL)
  install(
    TARGETS vw vw-bin allreduce vw_io
    EXPORT VowpalWabbitConfig
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  # VW target headers
  install(
    FILES ${vw_all_headers} ${CMAKE_CURRENT_BINARY_DIR}/config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vowpalwabbit)
  install(
    FILES io/io_adapter.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vowpalwabbit/io
  )
endif()
