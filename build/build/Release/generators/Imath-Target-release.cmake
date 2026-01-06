# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(imath_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(imath_FRAMEWORKS_FOUND_RELEASE "${imath_FRAMEWORKS_RELEASE}" "${imath_FRAMEWORK_DIRS_RELEASE}")

set(imath_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET imath_DEPS_TARGET)
    add_library(imath_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET imath_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${imath_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${imath_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:Imath::ImathConfig>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### imath_DEPS_TARGET to all of them
conan_package_library_targets("${imath_LIBS_RELEASE}"    # libraries
                              "${imath_LIB_DIRS_RELEASE}" # package_libdir
                              "${imath_BIN_DIRS_RELEASE}" # package_bindir
                              "${imath_LIBRARY_TYPE_RELEASE}"
                              "${imath_IS_HOST_WINDOWS_RELEASE}"
                              imath_DEPS_TARGET
                              imath_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "imath"    # package_name
                              "${imath_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${imath_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT Imath::Imath #############

        set(imath_Imath_Imath_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(imath_Imath_Imath_FRAMEWORKS_FOUND_RELEASE "${imath_Imath_Imath_FRAMEWORKS_RELEASE}" "${imath_Imath_Imath_FRAMEWORK_DIRS_RELEASE}")

        set(imath_Imath_Imath_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET imath_Imath_Imath_DEPS_TARGET)
            add_library(imath_Imath_Imath_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET imath_Imath_Imath_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${imath_Imath_Imath_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${imath_Imath_Imath_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${imath_Imath_Imath_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'imath_Imath_Imath_DEPS_TARGET' to all of them
        conan_package_library_targets("${imath_Imath_Imath_LIBS_RELEASE}"
                              "${imath_Imath_Imath_LIB_DIRS_RELEASE}"
                              "${imath_Imath_Imath_BIN_DIRS_RELEASE}" # package_bindir
                              "${imath_Imath_Imath_LIBRARY_TYPE_RELEASE}"
                              "${imath_Imath_Imath_IS_HOST_WINDOWS_RELEASE}"
                              imath_Imath_Imath_DEPS_TARGET
                              imath_Imath_Imath_LIBRARIES_TARGETS
                              "_RELEASE"
                              "imath_Imath_Imath"
                              "${imath_Imath_Imath_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET Imath::Imath
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${imath_Imath_Imath_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${imath_Imath_Imath_LIBRARIES_TARGETS}>
                     )

        if("${imath_Imath_Imath_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET Imath::Imath
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         imath_Imath_Imath_DEPS_TARGET)
        endif()

        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${imath_Imath_Imath_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${imath_Imath_Imath_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${imath_Imath_Imath_LIB_DIRS_RELEASE}>)
        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${imath_Imath_Imath_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${imath_Imath_Imath_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT Imath::ImathConfig #############

        set(imath_Imath_ImathConfig_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(imath_Imath_ImathConfig_FRAMEWORKS_FOUND_RELEASE "${imath_Imath_ImathConfig_FRAMEWORKS_RELEASE}" "${imath_Imath_ImathConfig_FRAMEWORK_DIRS_RELEASE}")

        set(imath_Imath_ImathConfig_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET imath_Imath_ImathConfig_DEPS_TARGET)
            add_library(imath_Imath_ImathConfig_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET imath_Imath_ImathConfig_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'imath_Imath_ImathConfig_DEPS_TARGET' to all of them
        conan_package_library_targets("${imath_Imath_ImathConfig_LIBS_RELEASE}"
                              "${imath_Imath_ImathConfig_LIB_DIRS_RELEASE}"
                              "${imath_Imath_ImathConfig_BIN_DIRS_RELEASE}" # package_bindir
                              "${imath_Imath_ImathConfig_LIBRARY_TYPE_RELEASE}"
                              "${imath_Imath_ImathConfig_IS_HOST_WINDOWS_RELEASE}"
                              imath_Imath_ImathConfig_DEPS_TARGET
                              imath_Imath_ImathConfig_LIBRARIES_TARGETS
                              "_RELEASE"
                              "imath_Imath_ImathConfig"
                              "${imath_Imath_ImathConfig_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET Imath::ImathConfig
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_LIBRARIES_TARGETS}>
                     )

        if("${imath_Imath_ImathConfig_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET Imath::ImathConfig
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         imath_Imath_ImathConfig_DEPS_TARGET)
        endif()

        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_LIB_DIRS_RELEASE}>)
        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET Imath::ImathConfig APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${imath_Imath_ImathConfig_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_LINK_LIBRARIES Imath::Imath)
    set_property(TARGET Imath::Imath APPEND PROPERTY INTERFACE_LINK_LIBRARIES Imath::ImathConfig)

########## For the modules (FindXXX)
set(imath_LIBRARIES_RELEASE Imath::Imath)
