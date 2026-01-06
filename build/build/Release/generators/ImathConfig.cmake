########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(Imath_FIND_QUIETLY)
    set(Imath_MESSAGE_MODE VERBOSE)
else()
    set(Imath_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ImathTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${imath_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(Imath_VERSION_STRING "3.1.9")
set(Imath_INCLUDE_DIRS ${imath_INCLUDE_DIRS_RELEASE} )
set(Imath_INCLUDE_DIR ${imath_INCLUDE_DIRS_RELEASE} )
set(Imath_LIBRARIES ${imath_LIBRARIES_RELEASE} )
set(Imath_DEFINITIONS ${imath_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${imath_BUILD_MODULES_PATHS_RELEASE} )
    message(${Imath_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


