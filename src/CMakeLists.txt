configure_file("${CMAKE_CURRENT_SOURCE_DIR}/gitsha1.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/gitsha1.cpp" @ONLY)

add_executable(
	castor_spectra
	castor_xs.cpp
	${CMAKE_CURRENT_BINARY_DIR}/gitsha1.cpp
    dummy.f
)

target_link_libraries(
	castor_spectra
	PRIVATE
	amplitude
	partondistribution
	fragmentation
	GSL::gsl
	GSL::gslcblas
)

