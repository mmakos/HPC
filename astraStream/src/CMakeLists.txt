SET( INT_LIBS )
IF( WIN32 )
    SET( EXT_LIBS )
ELSE()
    SET( EXT_LIBS )
ENDIF( WIN32 )

SET (TARGET astraStream)
INCLUDE_DIRECTORIES( ../../externals/AstraSDK/include )

FILE(GLOB cpp_files *.cpp)
FILE(GLOB h_files ./*.h* )

SET( cpp_files ${cpp_files} )

ADD_LIBRARY( ${TARGET} MODULE ${h_files} ${cpp_files} )

	
#LINK_INTERNAL( ${TARGET} "${INT_LIBS}" )
#LINK_EXTERNAL( ${TARGET} "${EXT_LIBS}" )

TARGET_LINK_LIBRARIES(${TARGET} )