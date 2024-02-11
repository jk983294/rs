file( GLOB APP_SOURCES "*.cpp")
foreach( sourcefile ${APP_SOURCES} )
    # executable name cut off .cpp
    string( REGEX REPLACE ".+/" "" executablename1 ${sourcefile} )
    string( REPLACE ".cpp" "" executablename ${executablename1} )

    message ("source files: " ${sourcefile} " executable: " ${executablename})
    add_executable( ${executablename} ${sourcefile} )
    
    target_link_libraries( ${executablename} feval_static arrow arrow_dataset fst liblz4 libzstd stdc++fs )
    if (FEVAL_BUILD_WITH_CUDA)
        target_link_libraries( ${executablename} gpuxx_static cudart_static)
    endif()
    target_link_libraries( ${executablename} pthread rt z dl )
endforeach( sourcefile ${APP_SOURCES} )