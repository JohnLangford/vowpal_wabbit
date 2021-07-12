# make sure it works with OpenJDK
set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_JVM_LIBRARY NotNeeded)

find_package(JNI REQUIRED)

set(src_base src/main/c++)
set(vw_jni_headers
  ${src_base}/jni_base_learner.h
  ${src_base}/vowpalWabbit_learner_VWCCBLearner.h
  ${src_base}/vowpalWabbit_learner_VWActionProbsLearner.h
  ${src_base}/vowpalWabbit_learner_VWActionScoresLearner.h
  ${src_base}/vowpalWabbit_learner_VWLearners.h
  ${src_base}/vowpalWabbit_learner_VWMulticlassLearner.h
  ${src_base}/vowpalWabbit_learner_VWMultilabelsLearner.h
  ${src_base}/vowpalWabbit_learner_VWProbLearner.h
  ${src_base}/vowpalWabbit_learner_VWScalarLearner.h
  ${src_base}/vowpalWabbit_learner_VWScalarsLearner.h
  ${src_base}/vowpalWabbit_VW.h
  ${src_base}/jni_spark_vw.h
  ${src_base}/jni_spark_vw_generated.h
  ${src_base}/util.h
)

set(vw_jni_sources
  ${src_base}/jni_base_learner.cc
  ${src_base}/vowpalWabbit_learner_VWCCBLearner.cc
  ${src_base}/vowpalWabbit_learner_VWActionProbsLearner.cc
  ${src_base}/vowpalWabbit_learner_VWActionScoresLearner.cc
  ${src_base}/vowpalWabbit_learner_VWLearners.cc
  ${src_base}/vowpalWabbit_learner_VWMulticlassLearner.cc
  ${src_base}/vowpalWabbit_learner_VWMultilabelsLearner.cc
  ${src_base}/vowpalWabbit_learner_VWProbLearner.cc
  ${src_base}/vowpalWabbit_learner_VWScalarLearner.cc
  ${src_base}/vowpalWabbit_learner_VWScalarsLearner.cc
  ${src_base}/vowpalWabbit_VW.cc
  ${src_base}/jni_spark_vw.cc
  ${src_base}/jni_spark_cluster.cc
  ${src_base}/util.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../vowpalwabbit/spanning_tree.cc
)

add_library(vw_jni SHARED ${vw_jni_headers} ${vw_jni_sources})
target_link_libraries(vw_jni PUBLIC vw)
set_property(TARGET vw_jni PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(vw_jni PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${JNI_INCLUDE_DIRS})

# Ensure target directory exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/target/bin/natives/linux_64/)
add_custom_command(TARGET vw_jni POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vw_jni> ${CMAKE_CURRENT_SOURCE_DIR}/target/bin/natives/linux_64/
)

if(STATIC_LINK_VW_JAVA)
  target_link_libraries(vw_jni PUBLIC -Wl,--exclude-libs,ALL -static-libgcc -static-libstdc++ -fPIC)
else()
  if(${CMAKE_EXECUTABLE_FORMAT} MATCHES "ELF")
    # enable-new-dtags and rpath enables shared object library lookup in the location of libvw_spark_jni.so
    target_link_libraries(vw_jni PUBLIC -fPIC -Wl,--enable-new-dtags -Wl,-rpath,\"\$ORIGIN\")
  else()
    # dtags are an ELF feature
    target_link_libraries(vw_jni PUBLIC -fPIC -Wl,-rpath,\"\$ORIGIN\")
  endif()
endif()

# Replace version number in POM
configure_file(pom.xml.in ${CMAKE_CURRENT_SOURCE_DIR}/pom.xml @ONLY)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(JAVA_INSTALL_PATH /usr/lib)
else()
    set(JAVA_INSTALL_PATH /Library/Java/Extensions)
endif()

if(VW_INSTALL)
  install(TARGETS vw_jni
    RUNTIME DESTINATION ${JAVA_INSTALL_PATH}
    LIBRARY DESTINATION ${JAVA_INSTALL_PATH}
  )
endif()

if(NOT WIN32)
  # Ensure target directory exists
  file(MAKE_DIRECTORY target/classes)
  file(MAKE_DIRECTORY target/test-classes)
  file(MAKE_DIRECTORY target/bin/natives/linux_64)

  # Development
  # - uncomment the following section to generate the jni headers
  # - it's commented to speed up the build as it's not expected to change frequently

  # find_package(Java)
  # include(UseJava)

  # add_custom_target(javacompile
      # COMMAND mvn compile
      # WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      # COMMENT "Compile classes for javah")

  # create_javah(TARGET javaheaders
  #   CLASSES
  #     org.vowpalwabbit.spark.VowpalWabbitNative
  #     org.vowpalwabbit.spark.VowpalWabbitExample
  #     org.vowpalwabbit.spark.ClusterSpanningTree
  #   CLASSPATH ${CMAKE_CURRENT_SOURCE_DIR}/target/classes
  #   OUTPUT_NAME ${CMAKE_CURRENT_SOURCE_DIR}/src/main/c++/jni_spark_vw_generated.h)
  # add_dependencies(javaheaders javacompile)

  SET(mvn_extra_args "")
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      set(JAVA_INSTALL_PATH /usr/lib)
      # the integration tests don't currently pass on mac
      SET(mvn_extra_args "-DskipTests")
  else()
      set(JAVA_INSTALL_PATH /Library/Java/Extensions)
  endif()

  add_custom_command(TARGET vw_jni POST_BUILD
      COMMAND cp $<TARGET_FILE:vw_jni> target/bin/natives/linux_64
      COMMAND echo $<TARGET_FILE:vw-bin> > ${CMAKE_CURRENT_SOURCE_DIR}/target/test-classes/vw-bin.txt
      # -B is batch mode and avoids the massive download progress print out
      COMMAND mvn -B verify ${mvn_extra_args}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Copying shared libary dependencies to output directory")

  # Replace version number in POM
  configure_file(pom.xml.in ${CMAKE_CURRENT_SOURCE_DIR}/pom.xml @ONLY)

  if(VW_INSTALL)
      install(TARGETS vw_jni
      RUNTIME DESTINATION ${JAVA_INSTALL_PATH}
      LIBRARY DESTINATION ${JAVA_INSTALL_PATH}
      )
  endif()

endif()
