#include <pybind11/pybind11.h>
#include <stream.h>

namespace py = pybind11;

PYBIND11_MODULE( astraPython, m )
{
	py::class_< AstraStream >( m, "AstraStream" )
		.def( py::init() )
		.def( "initialize", &AstraStream::init )
		.def( "proceedFrame", &AstraStream::proceedFrame )
		.def( "getRGB", &AstraStream::getRGB )
		.def( "close", &AstraStream::close )
		;
}