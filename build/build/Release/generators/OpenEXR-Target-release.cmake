# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(openexr_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(openexr_FRAMEWORKS_FOUND_RELEASE "${openexr_FRAMEWORKS_RELEASE}" "${openexr_FRAMEWORK_DIRS_RELEASE}")

set(openexr_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET openexr_DEPS_TARGET)
    add_library(openexr_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET openexr_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${openexr_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${openexr_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:OpenEXR::IexConfig;OpenEXR::IlmThreadConfig;OpenEXR::Iex;OpenEXR::OpenEXRConfig;ZLIB::ZLIB;libdeflate::libdeflate_static;OpenEXR::OpenEXRCore;OpenEXR::IlmThread;Imath::Imath;OpenEXR::OpenEXR>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### openexr_DEPS_TARGET to all of them
conan_package_library_targets("${openexr_LIBS_RELEASE}"    # libraries
                              "${openexr_LIB_DIRS_RELEASE}" # package_libdir
                              "${openexr_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_LIBRARY_TYPE_RELEASE}"
                              "${openexr_IS_HOST_WINDOWS_RELEASE}"
                              openexr_DEPS_TARGET
                              openexr_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "openexr"    # package_name
                              "${openexr_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${openexr_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT OpenEXR::OpenEXRUtil #############

        set(openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_FOUND_RELEASE "${openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_RELEASE}" "${openexr_OpenEXR_OpenEXRUtil_FRAMEWORK_DIRS_RELEASE}")

        set(openexr_OpenEXR_OpenEXRUtil_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET)
            add_library(openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_OpenEXRUtil_LIBS_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRUtil_LIB_DIRS_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRUtil_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_OpenEXR_OpenEXRUtil_LIBRARY_TYPE_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRUtil_IS_HOST_WINDOWS_RELEASE}"
                              openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET
                              openexr_OpenEXR_OpenEXRUtil_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openexr_OpenEXR_OpenEXRUtil"
                              "${openexr_OpenEXR_OpenEXRUtil_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::OpenEXRUtil
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_OpenEXRUtil_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::OpenEXRUtil
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_OpenEXRUtil_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRUtil APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT OpenEXR::OpenEXR #############

        set(openexr_OpenEXR_OpenEXR_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openexr_OpenEXR_OpenEXR_FRAMEWORKS_FOUND_RELEASE "${openexr_OpenEXR_OpenEXR_FRAMEWORKS_RELEASE}" "${openexr_OpenEXR_OpenEXR_FRAMEWORK_DIRS_RELEASE}")

        set(openexr_OpenEXR_OpenEXR_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_OpenEXR_DEPS_TARGET)
            add_library(openexr_OpenEXR_OpenEXR_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_OpenEXR_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_OpenEXR_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_OpenEXR_LIBS_RELEASE}"
                              "${openexr_OpenEXR_OpenEXR_LIB_DIRS_RELEASE}"
                              "${openexr_OpenEXR_OpenEXR_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_OpenEXR_OpenEXR_LIBRARY_TYPE_RELEASE}"
                              "${openexr_OpenEXR_OpenEXR_IS_HOST_WINDOWS_RELEASE}"
                              openexr_OpenEXR_OpenEXR_DEPS_TARGET
                              openexr_OpenEXR_OpenEXR_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openexr_OpenEXR_OpenEXR"
                              "${openexr_OpenEXR_OpenEXR_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::OpenEXR
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_OpenEXR_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::OpenEXR
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_OpenEXR_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXR APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT OpenEXR::IlmThread #############

        set(openexr_OpenEXR_IlmThread_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openexr_OpenEXR_IlmThread_FRAMEWORKS_FOUND_RELEASE "${openexr_OpenEXR_IlmThread_FRAMEWORKS_RELEASE}" "${openexr_OpenEXR_IlmThread_FRAMEWORK_DIRS_RELEASE}")

        set(openexr_OpenEXR_IlmThread_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_IlmThread_DEPS_TARGET)
            add_library(openexr_OpenEXR_IlmThread_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_IlmThread_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_IlmThread_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_IlmThread_LIBS_RELEASE}"
                              "${openexr_OpenEXR_IlmThread_LIB_DIRS_RELEASE}"
                              "${openexr_OpenEXR_IlmThread_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_OpenEXR_IlmThread_LIBRARY_TYPE_RELEASE}"
                              "${openexr_OpenEXR_IlmThread_IS_HOST_WINDOWS_RELEASE}"
                              openexr_OpenEXR_IlmThread_DEPS_TARGET
                              openexr_OpenEXR_IlmThread_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openexr_OpenEXR_IlmThread"
                              "${openexr_OpenEXR_IlmThread_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::IlmThread
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_IlmThread_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::IlmThread
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_IlmThread_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenEXR::IlmThread APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT OpenEXR::OpenEXRCore #############

        set(openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_FOUND_RELEASE "${openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_RELEASE}" "${openexr_OpenEXR_OpenEXRCore_FRAMEWORK_DIRS_RELEASE}")

        set(openexr_OpenEXR_OpenEXRCore_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_OpenEXRCore_DEPS_TARGET)
            add_library(openexr_OpenEXR_OpenEXRCore_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_OpenEXRCore_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_OpenEXRCore_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_OpenEXRCore_LIBS_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRCore_LIB_DIRS_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRCore_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_OpenEXR_OpenEXRCore_LIBRARY_TYPE_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRCore_IS_HOST_WINDOWS_RELEASE}"
                              openexr_OpenEXR_OpenEXRCore_DEPS_TARGET
                              openexr_OpenEXR_OpenEXRCore_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openexr_OpenEXR_OpenEXRCore"
                              "${openexr_OpenEXR_OpenEXRCore_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::OpenEXRCore
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_OpenEXRCore_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::OpenEXRCore
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_OpenEXRCore_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRCore APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT OpenEXR::Iex #############

        set(openexr_OpenEXR_Iex_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openexr_OpenEXR_Iex_FRAMEWORKS_FOUND_RELEASE "${openexr_OpenEXR_Iex_FRAMEWORKS_RELEASE}" "${openexr_OpenEXR_Iex_FRAMEWORK_DIRS_RELEASE}")

        set(openexr_OpenEXR_Iex_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_Iex_DEPS_TARGET)
            add_library(openexr_OpenEXR_Iex_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_Iex_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_Iex_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_Iex_LIBS_RELEASE}"
                              "${openexr_OpenEXR_Iex_LIB_DIRS_RELEASE}"
                              "${openexr_OpenEXR_Iex_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_OpenEXR_Iex_LIBRARY_TYPE_RELEASE}"
                              "${openexr_OpenEXR_Iex_IS_HOST_WINDOWS_RELEASE}"
                              openexr_OpenEXR_Iex_DEPS_TARGET
                              openexr_OpenEXR_Iex_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openexr_OpenEXR_Iex"
                              "${openexr_OpenEXR_Iex_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::Iex
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_Iex_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::Iex
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_Iex_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenEXR::Iex APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_Iex_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT OpenEXR::IlmThreadConfig #############

        set(openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_FOUND_RELEASE "${openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_RELEASE}" "${openexr_OpenEXR_IlmThreadConfig_FRAMEWORK_DIRS_RELEASE}")

        set(openexr_OpenEXR_IlmThreadConfig_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET)
            add_library(openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_IlmThreadConfig_LIBS_RELEASE}"
                              "${openexr_OpenEXR_IlmThreadConfig_LIB_DIRS_RELEASE}"
                              "${openexr_OpenEXR_IlmThreadConfig_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_OpenEXR_IlmThreadConfig_LIBRARY_TYPE_RELEASE}"
                              "${openexr_OpenEXR_IlmThreadConfig_IS_HOST_WINDOWS_RELEASE}"
                              openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET
                              openexr_OpenEXR_IlmThreadConfig_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openexr_OpenEXR_IlmThreadConfig"
                              "${openexr_OpenEXR_IlmThreadConfig_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::IlmThreadConfig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_IlmThreadConfig_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::IlmThreadConfig
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_IlmThreadConfig_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenEXR::IlmThreadConfig APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT OpenEXR::IexConfig #############

        set(openexr_OpenEXR_IexConfig_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openexr_OpenEXR_IexConfig_FRAMEWORKS_FOUND_RELEASE "${openexr_OpenEXR_IexConfig_FRAMEWORKS_RELEASE}" "${openexr_OpenEXR_IexConfig_FRAMEWORK_DIRS_RELEASE}")

        set(openexr_OpenEXR_IexConfig_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_IexConfig_DEPS_TARGET)
            add_library(openexr_OpenEXR_IexConfig_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_IexConfig_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_IexConfig_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_IexConfig_LIBS_RELEASE}"
                              "${openexr_OpenEXR_IexConfig_LIB_DIRS_RELEASE}"
                              "${openexr_OpenEXR_IexConfig_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_OpenEXR_IexConfig_LIBRARY_TYPE_RELEASE}"
                              "${openexr_OpenEXR_IexConfig_IS_HOST_WINDOWS_RELEASE}"
                              openexr_OpenEXR_IexConfig_DEPS_TARGET
                              openexr_OpenEXR_IexConfig_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openexr_OpenEXR_IexConfig"
                              "${openexr_OpenEXR_IexConfig_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::IexConfig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_IexConfig_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::IexConfig
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_IexConfig_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenEXR::IexConfig APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT OpenEXR::OpenEXRConfig #############

        set(openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_FOUND_RELEASE "${openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_RELEASE}" "${openexr_OpenEXR_OpenEXRConfig_FRAMEWORK_DIRS_RELEASE}")

        set(openexr_OpenEXR_OpenEXRConfig_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET)
            add_library(openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET' to all of them
        conan_package_library_targets("${openexr_OpenEXR_OpenEXRConfig_LIBS_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRConfig_LIB_DIRS_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRConfig_BIN_DIRS_RELEASE}" # package_bindir
                              "${openexr_OpenEXR_OpenEXRConfig_LIBRARY_TYPE_RELEASE}"
                              "${openexr_OpenEXR_OpenEXRConfig_IS_HOST_WINDOWS_RELEASE}"
                              openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET
                              openexr_OpenEXR_OpenEXRConfig_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openexr_OpenEXR_OpenEXRConfig"
                              "${openexr_OpenEXR_OpenEXRConfig_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenEXR::OpenEXRConfig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_LIBRARIES_TARGETS}>
                     )

        if("${openexr_OpenEXR_OpenEXRConfig_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenEXR::OpenEXRConfig
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openexr_OpenEXR_OpenEXRConfig_DEPS_TARGET)
        endif()

        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenEXR::OpenEXRConfig APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::OpenEXRUtil)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::OpenEXR)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::IlmThread)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::OpenEXRCore)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::Iex)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::IlmThreadConfig)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::IexConfig)
    set_property(TARGET openexr::openexr APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenEXR::OpenEXRConfig)

########## For the modules (FindXXX)
set(openexr_LIBRARIES_RELEASE openexr::openexr)
