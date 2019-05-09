add_executable(
	castor_spectra
	castor_xs.cpp
	gitsha1.cpp
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

