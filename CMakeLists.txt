# stop if cmake version is below 3.0
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# project name and enable C++ support
project(castor CXX C Fortran)

# Optimization on (for debug flags, change to Debug)
set(CMAKE_BUILD_TYPE Release)

# Git module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_describe(GIT_DESCRIBE)
git_local_changes(GIT_LOCAL_CHANGES)

include_directories("../amplitudelib_v2/")
# Directory where you have build AmplitudeLib libraries
link_directories("../amplitudelib_v2/build/lib/")

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
include_directories("./src/")


add_subdirectory(src)

set_target_properties(castor_spectra PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )

