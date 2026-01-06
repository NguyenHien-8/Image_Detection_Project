# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(quirc_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(quirc_FRAMEWORKS_FOUND_RELEASE "${quirc_FRAMEWORKS_RELEASE}" "${quirc_FRAMEWORK_DIRS_RELEASE}")

set(quirc_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET quirc_DEPS_TARGET)
    add_library(quirc_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET quirc_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${quirc_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${quirc_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### quirc_DEPS_TARGET to all of them
conan_package_library_targets("${quirc_LIBS_RELEASE}"    # libraries
                              "${quirc_LIB_DIRS_RELEASE}" # package_libdir
                              "${quirc_BIN_DIRS_RELEASE}" # package_bindir
                              "${quirc_LIBRARY_TYPE_RELEASE}"
                              "${quirc_IS_HOST_WINDOWS_RELEASE}"
                              quirc_DEPS_TARGET
                              quirc_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "quirc"    # package_name
                              "${quirc_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${quirc_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${quirc_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${quirc_LIBRARIES_TARGETS}>
                 )

    if("${quirc_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET quirc::quirc
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     quirc_DEPS_TARGET)
    endif()

    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${quirc_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${quirc_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${quirc_LIB_DIRS_RELEASE}>)
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${quirc_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET quirc::quirc
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${quirc_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(quirc_LIBRARIES_RELEASE quirc::quirc)
