# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/libdeflate-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${libdeflate_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${libdeflate_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET libdeflate::libdeflate_static)
    add_library(libdeflate::libdeflate_static INTERFACE IMPORTED)
    message(${libdeflate_MESSAGE_MODE} "Conan: Target declared 'libdeflate::libdeflate_static'")
endif()
if(NOT TARGET libdeflate::libdeflate)
    add_library(libdeflate::libdeflate INTERFACE IMPORTED)
    set_property(TARGET libdeflate::libdeflate PROPERTY INTERFACE_LINK_LIBRARIES libdeflate::libdeflate_static)
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/libdeflate-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()