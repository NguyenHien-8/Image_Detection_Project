# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/gtk-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${gtk_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${gtk_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET gtk::gtk)
    add_library(gtk::gtk INTERFACE IMPORTED)
    message(${gtk_MESSAGE_MODE} "Conan: Target declared 'gtk::gtk'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/gtk-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()