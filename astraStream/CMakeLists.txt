cmake_minimum_required( VERSION 3.17 )
project( astraStream )

add_subdirectory( src )

install( TARGETS astraStream LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin 
			  LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../bin  )