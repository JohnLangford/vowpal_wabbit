set(VW_SLIM_SOURCES
  example_predict_builder.cc
  model_parser.cc
  opts.cc
  vw_slim_predict.cc
  ../../feature_group.cc
  ../../example_predict.cc
  ../../learner_no_throw.cc
  ../../api_status.cc
  ../../interactions.cc)

set(VW_SLIM_HEADERS
  ../include/example_predict_builder.h
  ../include/model_parser.h
  ../include/opts.h
  ../include/vw_slim_predict.h
  ../include/vw_slim_return_codes.h)

add_library(vwslim ${VW_SLIM_SOURCES} ${VW_SLIM_HEADERS})

target_include_directories(vwslim PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../include
  ${CMAKE_CURRENT_LIST_DIR}/../../
  ${CMAKE_CURRENT_BINARY_DIR}/../../)
target_compile_definitions(vwslim PUBLIC EXPLORE_NOEXCEPT VW_NOEXCEPT)
target_link_libraries(vwslim PUBLIC VowpalWabbit::explore)

include(GNUInstallDirs)

install(
  TARGETS vwslim
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
  FILES ${VW_SLIM_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vowpalwabbit/slim)
