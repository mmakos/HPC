cmake_minimum_required( VERSION 3.17.2 )
project( astraPython )

include_directories( ../astraStream/src )
include_directories( ../externals/AstraSDK/include )
link_directories( ../externals/AstraSDK/lib )
add_subdirectory( pybind11 )

add_library( astraPython MODULE src/astraWrapper.cpp  )
target_link_libraries( astraPython PRIVATE pybind11::module astra astra_core astra_core_api )

install( TARGETS astraPython LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin 
			  LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../bin )