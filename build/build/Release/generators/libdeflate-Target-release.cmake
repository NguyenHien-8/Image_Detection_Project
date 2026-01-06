# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(libdeflate_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(libdeflate_FRAMEWORKS_FOUND_RELEASE "${libdeflate_FRAMEWORKS_RELEASE}" "${libdeflate_FRAMEWORK_DIRS_RELEASE}")

set(libdeflate_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET libdeflate_DEPS_TARGET)
    add_library(libdeflate_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET libdeflate_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${libdeflate_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${libdeflate_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### libdeflate_DEPS_TARGET to all of them
conan_package_library_targets("${libdeflate_LIBS_RELEASE}"    # libraries
                              "${libdeflate_LIB_DIRS_RELEASE}" # package_libdir
                              "${libdeflate_BIN_DIRS_RELEASE}" # package_bindir
                              "${libdeflate_LIBRARY_TYPE_RELEASE}"
                              "${libdeflate_IS_HOST_WINDOWS_RELEASE}"
                              libdeflate_DEPS_TARGET
                              libdeflate_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "libdeflate"    # package_name
                              "${libdeflate_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${libdeflate_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT libdeflate::libdeflate_static #############

        set(libdeflate_libdeflate_libdeflate_static_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(libdeflate_libdeflate_libdeflate_static_FRAMEWORKS_FOUND_RELEASE "${libdeflate_libdeflate_libdeflate_static_FRAMEWORKS_RELEASE}" "${libdeflate_libdeflate_libdeflate_static_FRAMEWORK_DIRS_RELEASE}")

        set(libdeflate_libdeflate_libdeflate_static_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET libdeflate_libdeflate_libdeflate_static_DEPS_TARGET)
            add_library(libdeflate_libdeflate_libdeflate_static_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET libdeflate_libdeflate_libdeflate_static_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'libdeflate_libdeflate_libdeflate_static_DEPS_TARGET' to all of them
        conan_package_library_targets("${libdeflate_libdeflate_libdeflate_static_LIBS_RELEASE}"
                              "${libdeflate_libdeflate_libdeflate_static_LIB_DIRS_RELEASE}"
                              "${libdeflate_libdeflate_libdeflate_static_BIN_DIRS_RELEASE}" # package_bindir
                              "${libdeflate_libdeflate_libdeflate_static_LIBRARY_TYPE_RELEASE}"
                              "${libdeflate_libdeflate_libdeflate_static_IS_HOST_WINDOWS_RELEASE}"
                              libdeflate_libdeflate_libdeflate_static_DEPS_TARGET
                              libdeflate_libdeflate_libdeflate_static_LIBRARIES_TARGETS
                              "_RELEASE"
                              "libdeflate_libdeflate_libdeflate_static"
                              "${libdeflate_libdeflate_libdeflate_static_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET libdeflate::libdeflate_static
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_LIBRARIES_TARGETS}>
                     )

        if("${libdeflate_libdeflate_libdeflate_static_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET libdeflate::libdeflate_static
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         libdeflate_libdeflate_libdeflate_static_DEPS_TARGET)
        endif()

        set_property(TARGET libdeflate::libdeflate_static APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET libdeflate::libdeflate_static APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET libdeflate::libdeflate_static APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_LIB_DIRS_RELEASE}>)
        set_property(TARGET libdeflate::libdeflate_static APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET libdeflate::libdeflate_static APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${libdeflate_libdeflate_libdeflate_static_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET libdeflate::libdeflate_static APPEND PROPERTY INTERFACE_LINK_LIBRARIES libdeflate::libdeflate_static)

########## For the modules (FindXXX)
set(libdeflate_LIBRARIES_RELEASE libdeflate::libdeflate_static)
