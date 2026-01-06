# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/OpenCV-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${opencv_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${OpenCV_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET opencv::opencv)
    add_library(opencv::opencv INTERFACE IMPORTED)
    message(${OpenCV_MESSAGE_MODE} "Conan: Target declared 'opencv::opencv'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/OpenCV-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()