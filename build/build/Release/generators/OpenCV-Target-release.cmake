# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(opencv_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(opencv_FRAMEWORKS_FOUND_RELEASE "${opencv_FRAMEWORKS_RELEASE}" "${opencv_FRAMEWORK_DIRS_RELEASE}")

set(opencv_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET opencv_DEPS_TARGET)
    add_library(opencv_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET opencv_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${opencv_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${opencv_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:opencv_core;opencv_features2d;opencv_flann;opencv_imgproc;opencv_calib3d;ZLIB::ZLIB;protobuf::protobuf;opencv_dnn;ade;opencv_video;opencv_gapi;opencv_imgcodecs;opencv_videoio;gtk::gtk;opencv_highgui;openjp2;PNG::PNG;JPEG::JPEG;TIFF::TIFF;openexr::openexr;libwebp::libwebp;opencv_ml;quirc::quirc;opencv_objdetect;opencv_photo;opencv_stitching>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### opencv_DEPS_TARGET to all of them
conan_package_library_targets("${opencv_LIBS_RELEASE}"    # libraries
                              "${opencv_LIB_DIRS_RELEASE}" # package_libdir
                              "${opencv_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_LIBRARY_TYPE_RELEASE}"
                              "${opencv_IS_HOST_WINDOWS_RELEASE}"
                              opencv_DEPS_TARGET
                              opencv_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "opencv"    # package_name
                              "${opencv_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${opencv_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT opencv::opencv_gapi_alias #############

        set(opencv_opencv_opencv_gapi_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_gapi_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_gapi_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_gapi_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_gapi_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_gapi_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_gapi_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_gapi_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_gapi_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_gapi_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_gapi_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_gapi_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_gapi_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_gapi_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_gapi_alias_DEPS_TARGET
                              opencv_opencv_opencv_gapi_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_gapi_alias"
                              "${opencv_opencv_opencv_gapi_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_gapi_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_gapi_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_gapi_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_gapi_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_gapi_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_gapi_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_gapi_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_gapi_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_gapi_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_gapi_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_video_alias #############

        set(opencv_opencv_opencv_video_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_video_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_video_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_video_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_video_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_video_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_video_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_video_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_video_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_video_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_video_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_video_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_video_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_video_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_video_alias_DEPS_TARGET
                              opencv_opencv_opencv_video_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_video_alias"
                              "${opencv_opencv_opencv_video_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_video_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_video_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_video_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_video_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_video_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_video_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_video_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_video_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_video_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_stitching_alias #############

        set(opencv_opencv_opencv_stitching_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_stitching_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_stitching_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_stitching_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_stitching_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_stitching_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_stitching_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_stitching_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_stitching_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_stitching_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_stitching_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_stitching_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_stitching_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_stitching_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_stitching_alias_DEPS_TARGET
                              opencv_opencv_opencv_stitching_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_stitching_alias"
                              "${opencv_opencv_opencv_stitching_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_stitching_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_stitching_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_stitching_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_stitching_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_stitching_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_stitching_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_stitching_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_stitching_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_stitching_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_objdetect_alias #############

        set(opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_objdetect_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_objdetect_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_objdetect_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_objdetect_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_objdetect_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_objdetect_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_objdetect_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_objdetect_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_objdetect_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_objdetect_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_objdetect_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_objdetect_alias_DEPS_TARGET
                              opencv_opencv_opencv_objdetect_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_objdetect_alias"
                              "${opencv_opencv_opencv_objdetect_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_objdetect_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_objdetect_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_objdetect_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_objdetect_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_objdetect_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_objdetect_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_objdetect_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_objdetect_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_objdetect_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_highgui_alias #############

        set(opencv_opencv_opencv_highgui_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_highgui_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_highgui_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_highgui_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_highgui_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_highgui_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_highgui_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_highgui_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_highgui_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_highgui_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_highgui_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_highgui_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_highgui_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_highgui_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_highgui_alias_DEPS_TARGET
                              opencv_opencv_opencv_highgui_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_highgui_alias"
                              "${opencv_opencv_opencv_highgui_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_highgui_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_highgui_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_highgui_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_highgui_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_highgui_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_highgui_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_highgui_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_highgui_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_highgui_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_gapi #############

        set(opencv_opencv_gapi_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_gapi_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_gapi_FRAMEWORKS_RELEASE}" "${opencv_opencv_gapi_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_gapi_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_gapi_DEPS_TARGET)
            add_library(opencv_opencv_gapi_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_gapi_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_gapi_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_gapi_LIBS_RELEASE}"
                              "${opencv_opencv_gapi_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_gapi_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_gapi_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_gapi_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_gapi_DEPS_TARGET
                              opencv_opencv_gapi_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_gapi"
                              "${opencv_opencv_gapi_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_gapi
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_gapi_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_gapi
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_gapi_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_gapi APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_gapi APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_gapi APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_gapi APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_gapi APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_gapi_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_videoio_alias #############

        set(opencv_opencv_opencv_videoio_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_videoio_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_videoio_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_videoio_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_videoio_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_videoio_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_videoio_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_videoio_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_videoio_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_videoio_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_videoio_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_videoio_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_videoio_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_videoio_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_videoio_alias_DEPS_TARGET
                              opencv_opencv_opencv_videoio_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_videoio_alias"
                              "${opencv_opencv_opencv_videoio_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_videoio_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_videoio_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_videoio_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_videoio_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_videoio_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_videoio_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_videoio_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_videoio_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_videoio_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_video #############

        set(opencv_opencv_video_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_video_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_video_FRAMEWORKS_RELEASE}" "${opencv_opencv_video_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_video_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_video_DEPS_TARGET)
            add_library(opencv_opencv_video_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_video_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_video_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_video_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_video_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_video_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_video_LIBS_RELEASE}"
                              "${opencv_opencv_video_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_video_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_video_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_video_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_video_DEPS_TARGET
                              opencv_opencv_video_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_video"
                              "${opencv_opencv_video_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_video
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_video_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_video_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_video_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_video
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_video_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_video APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_video_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_video APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_video_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_video APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_video_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_video APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_video_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_video APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_video_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_stitching #############

        set(opencv_opencv_stitching_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_stitching_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_stitching_FRAMEWORKS_RELEASE}" "${opencv_opencv_stitching_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_stitching_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_stitching_DEPS_TARGET)
            add_library(opencv_opencv_stitching_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_stitching_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_stitching_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_stitching_LIBS_RELEASE}"
                              "${opencv_opencv_stitching_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_stitching_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_stitching_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_stitching_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_stitching_DEPS_TARGET
                              opencv_opencv_stitching_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_stitching"
                              "${opencv_opencv_stitching_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_stitching
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_stitching_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_stitching
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_stitching_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_stitching APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_stitching APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_stitching APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_stitching APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_stitching APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_stitching_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_objdetect #############

        set(opencv_opencv_objdetect_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_objdetect_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_objdetect_FRAMEWORKS_RELEASE}" "${opencv_opencv_objdetect_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_objdetect_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_objdetect_DEPS_TARGET)
            add_library(opencv_opencv_objdetect_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_objdetect_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_objdetect_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_objdetect_LIBS_RELEASE}"
                              "${opencv_opencv_objdetect_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_objdetect_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_objdetect_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_objdetect_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_objdetect_DEPS_TARGET
                              opencv_opencv_objdetect_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_objdetect"
                              "${opencv_opencv_objdetect_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_objdetect
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_objdetect_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_objdetect
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_objdetect_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_objdetect APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_objdetect APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_objdetect APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_objdetect APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_objdetect APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_objdetect_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_highgui #############

        set(opencv_opencv_highgui_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_highgui_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_highgui_FRAMEWORKS_RELEASE}" "${opencv_opencv_highgui_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_highgui_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_highgui_DEPS_TARGET)
            add_library(opencv_opencv_highgui_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_highgui_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_highgui_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_highgui_LIBS_RELEASE}"
                              "${opencv_opencv_highgui_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_highgui_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_highgui_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_highgui_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_highgui_DEPS_TARGET
                              opencv_opencv_highgui_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_highgui"
                              "${opencv_opencv_highgui_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_highgui
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_highgui_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_highgui
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_highgui_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_highgui APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_highgui APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_highgui APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_highgui APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_highgui APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_highgui_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_calib3d_alias #############

        set(opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_calib3d_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_calib3d_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_calib3d_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_calib3d_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_calib3d_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_calib3d_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_calib3d_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_calib3d_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_calib3d_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_calib3d_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_calib3d_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_calib3d_alias_DEPS_TARGET
                              opencv_opencv_opencv_calib3d_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_calib3d_alias"
                              "${opencv_opencv_opencv_calib3d_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_calib3d_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_calib3d_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_calib3d_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_calib3d_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_calib3d_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_calib3d_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_calib3d_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_calib3d_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_calib3d_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_videoio #############

        set(opencv_opencv_videoio_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_videoio_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_videoio_FRAMEWORKS_RELEASE}" "${opencv_opencv_videoio_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_videoio_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_videoio_DEPS_TARGET)
            add_library(opencv_opencv_videoio_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_videoio_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_videoio_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_videoio_LIBS_RELEASE}"
                              "${opencv_opencv_videoio_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_videoio_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_videoio_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_videoio_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_videoio_DEPS_TARGET
                              opencv_opencv_videoio_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_videoio"
                              "${opencv_opencv_videoio_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_videoio
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_videoio_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_videoio
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_videoio_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_videoio APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_videoio APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_videoio APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_videoio APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_videoio APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_videoio_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_photo_alias #############

        set(opencv_opencv_opencv_photo_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_photo_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_photo_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_photo_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_photo_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_photo_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_photo_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_photo_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_photo_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_photo_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_photo_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_photo_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_photo_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_photo_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_photo_alias_DEPS_TARGET
                              opencv_opencv_opencv_photo_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_photo_alias"
                              "${opencv_opencv_opencv_photo_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_photo_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_photo_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_photo_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_photo_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_photo_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_photo_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_photo_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_photo_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_photo_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_imgcodecs_alias #############

        set(opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_imgcodecs_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_imgcodecs_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_imgcodecs_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_imgcodecs_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_imgcodecs_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_imgcodecs_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_imgcodecs_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_imgcodecs_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_imgcodecs_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_imgcodecs_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_imgcodecs_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_imgcodecs_alias_DEPS_TARGET
                              opencv_opencv_opencv_imgcodecs_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_imgcodecs_alias"
                              "${opencv_opencv_opencv_imgcodecs_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_imgcodecs_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_imgcodecs_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_imgcodecs_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_imgcodecs_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_imgcodecs_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_imgcodecs_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_imgcodecs_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_imgcodecs_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_imgcodecs_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_features2d_alias #############

        set(opencv_opencv_opencv_features2d_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_features2d_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_features2d_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_features2d_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_features2d_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_features2d_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_features2d_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_features2d_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_features2d_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_features2d_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_features2d_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_features2d_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_features2d_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_features2d_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_features2d_alias_DEPS_TARGET
                              opencv_opencv_opencv_features2d_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_features2d_alias"
                              "${opencv_opencv_opencv_features2d_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_features2d_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_features2d_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_features2d_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_features2d_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_features2d_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_features2d_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_features2d_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_features2d_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_features2d_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_dnn_alias #############

        set(opencv_opencv_opencv_dnn_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_dnn_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_dnn_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_dnn_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_dnn_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_dnn_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_dnn_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_dnn_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_dnn_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_dnn_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_dnn_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_dnn_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_dnn_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_dnn_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_dnn_alias_DEPS_TARGET
                              opencv_opencv_opencv_dnn_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_dnn_alias"
                              "${opencv_opencv_opencv_dnn_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_dnn_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_dnn_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_dnn_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_dnn_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_dnn_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_dnn_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_dnn_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_dnn_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_dnn_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_dnn_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_calib3d #############

        set(opencv_opencv_calib3d_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_calib3d_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_calib3d_FRAMEWORKS_RELEASE}" "${opencv_opencv_calib3d_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_calib3d_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_calib3d_DEPS_TARGET)
            add_library(opencv_opencv_calib3d_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_calib3d_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_calib3d_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_calib3d_LIBS_RELEASE}"
                              "${opencv_opencv_calib3d_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_calib3d_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_calib3d_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_calib3d_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_calib3d_DEPS_TARGET
                              opencv_opencv_calib3d_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_calib3d"
                              "${opencv_opencv_calib3d_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_calib3d
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_calib3d_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_calib3d
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_calib3d_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_calib3d APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_calib3d APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_calib3d APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_calib3d APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_calib3d APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_calib3d_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_photo #############

        set(opencv_opencv_photo_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_photo_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_photo_FRAMEWORKS_RELEASE}" "${opencv_opencv_photo_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_photo_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_photo_DEPS_TARGET)
            add_library(opencv_opencv_photo_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_photo_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_photo_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_photo_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_photo_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_photo_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_photo_LIBS_RELEASE}"
                              "${opencv_opencv_photo_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_photo_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_photo_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_photo_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_photo_DEPS_TARGET
                              opencv_opencv_photo_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_photo"
                              "${opencv_opencv_photo_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_photo
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_photo_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_photo_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_photo_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_photo
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_photo_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_photo APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_photo_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_photo APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_photo_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_photo APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_photo_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_photo APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_photo_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_photo APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_photo_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_ml_alias #############

        set(opencv_opencv_opencv_ml_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_ml_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_ml_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_ml_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_ml_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_ml_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_ml_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_ml_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_ml_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_ml_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_ml_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_ml_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_ml_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_ml_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_ml_alias_DEPS_TARGET
                              opencv_opencv_opencv_ml_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_ml_alias"
                              "${opencv_opencv_opencv_ml_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_ml_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_ml_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_ml_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_ml_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_ml_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_ml_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_ml_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_ml_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_ml_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_imgproc_alias #############

        set(opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_imgproc_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_imgproc_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_imgproc_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_imgproc_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_imgproc_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_imgproc_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_imgproc_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_imgproc_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_imgproc_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_imgproc_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_imgproc_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_imgproc_alias_DEPS_TARGET
                              opencv_opencv_opencv_imgproc_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_imgproc_alias"
                              "${opencv_opencv_opencv_imgproc_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_imgproc_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_imgproc_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_imgproc_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_imgproc_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_imgproc_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_imgproc_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_imgproc_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_imgproc_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_imgproc_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_imgcodecs #############

        set(opencv_opencv_imgcodecs_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_imgcodecs_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_imgcodecs_FRAMEWORKS_RELEASE}" "${opencv_opencv_imgcodecs_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_imgcodecs_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_imgcodecs_DEPS_TARGET)
            add_library(opencv_opencv_imgcodecs_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_imgcodecs_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_imgcodecs_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_imgcodecs_LIBS_RELEASE}"
                              "${opencv_opencv_imgcodecs_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_imgcodecs_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_imgcodecs_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_imgcodecs_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_imgcodecs_DEPS_TARGET
                              opencv_opencv_imgcodecs_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_imgcodecs"
                              "${opencv_opencv_imgcodecs_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_imgcodecs
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_imgcodecs_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_imgcodecs
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_imgcodecs_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_imgcodecs APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_imgcodecs APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_imgcodecs APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_imgcodecs APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_imgcodecs APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_imgcodecs_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_flann_alias #############

        set(opencv_opencv_opencv_flann_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_flann_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_flann_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_flann_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_flann_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_flann_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_flann_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_flann_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_flann_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_flann_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_flann_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_flann_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_flann_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_flann_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_flann_alias_DEPS_TARGET
                              opencv_opencv_opencv_flann_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_flann_alias"
                              "${opencv_opencv_opencv_flann_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_flann_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_flann_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_flann_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_flann_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_flann_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_flann_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_flann_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_flann_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_flann_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_features2d #############

        set(opencv_opencv_features2d_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_features2d_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_features2d_FRAMEWORKS_RELEASE}" "${opencv_opencv_features2d_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_features2d_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_features2d_DEPS_TARGET)
            add_library(opencv_opencv_features2d_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_features2d_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_features2d_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_features2d_LIBS_RELEASE}"
                              "${opencv_opencv_features2d_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_features2d_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_features2d_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_features2d_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_features2d_DEPS_TARGET
                              opencv_opencv_features2d_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_features2d"
                              "${opencv_opencv_features2d_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_features2d
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_features2d_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_features2d
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_features2d_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_features2d APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_features2d APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_features2d APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_features2d APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_features2d APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_features2d_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_dnn #############

        set(opencv_opencv_dnn_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_dnn_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_dnn_FRAMEWORKS_RELEASE}" "${opencv_opencv_dnn_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_dnn_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_dnn_DEPS_TARGET)
            add_library(opencv_opencv_dnn_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_dnn_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_dnn_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_dnn_LIBS_RELEASE}"
                              "${opencv_opencv_dnn_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_dnn_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_dnn_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_dnn_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_dnn_DEPS_TARGET
                              opencv_opencv_dnn_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_dnn"
                              "${opencv_opencv_dnn_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_dnn
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_dnn_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_dnn
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_dnn_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_dnn APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_dnn APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_dnn APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_dnn APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_dnn APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_dnn_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_ml #############

        set(opencv_opencv_ml_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_ml_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_ml_FRAMEWORKS_RELEASE}" "${opencv_opencv_ml_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_ml_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_ml_DEPS_TARGET)
            add_library(opencv_opencv_ml_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_ml_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_ml_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_ml_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_ml_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_ml_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_ml_LIBS_RELEASE}"
                              "${opencv_opencv_ml_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_ml_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_ml_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_ml_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_ml_DEPS_TARGET
                              opencv_opencv_ml_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_ml"
                              "${opencv_opencv_ml_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_ml
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_ml_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_ml_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_ml_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_ml
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_ml_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_ml APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_ml_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_ml APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_ml_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_ml APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_ml_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_ml APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_ml_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_ml APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_ml_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_imgproc #############

        set(opencv_opencv_imgproc_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_imgproc_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_imgproc_FRAMEWORKS_RELEASE}" "${opencv_opencv_imgproc_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_imgproc_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_imgproc_DEPS_TARGET)
            add_library(opencv_opencv_imgproc_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_imgproc_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_imgproc_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_imgproc_LIBS_RELEASE}"
                              "${opencv_opencv_imgproc_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_imgproc_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_imgproc_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_imgproc_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_imgproc_DEPS_TARGET
                              opencv_opencv_imgproc_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_imgproc"
                              "${opencv_opencv_imgproc_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_imgproc
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_imgproc_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_imgproc
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_imgproc_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_imgproc APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_imgproc APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_imgproc APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_imgproc APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_imgproc APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_imgproc_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_flann #############

        set(opencv_opencv_flann_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_flann_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_flann_FRAMEWORKS_RELEASE}" "${opencv_opencv_flann_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_flann_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_flann_DEPS_TARGET)
            add_library(opencv_opencv_flann_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_flann_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_flann_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_flann_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_flann_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_flann_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_flann_LIBS_RELEASE}"
                              "${opencv_opencv_flann_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_flann_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_flann_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_flann_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_flann_DEPS_TARGET
                              opencv_opencv_flann_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_flann"
                              "${opencv_opencv_flann_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_flann
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_flann_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_flann_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_flann_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_flann
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_flann_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_flann APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_flann_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_flann APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_flann_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_flann APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_flann_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_flann APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_flann_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_flann APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_flann_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv::opencv_core_alias #############

        set(opencv_opencv_opencv_core_alias_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_opencv_core_alias_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_opencv_core_alias_FRAMEWORKS_RELEASE}" "${opencv_opencv_opencv_core_alias_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_opencv_core_alias_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_opencv_core_alias_DEPS_TARGET)
            add_library(opencv_opencv_opencv_core_alias_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_opencv_core_alias_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_opencv_core_alias_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_opencv_core_alias_LIBS_RELEASE}"
                              "${opencv_opencv_opencv_core_alias_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_opencv_core_alias_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_opencv_core_alias_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_opencv_core_alias_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_opencv_core_alias_DEPS_TARGET
                              opencv_opencv_opencv_core_alias_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_opencv_core_alias"
                              "${opencv_opencv_opencv_core_alias_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv::opencv_core_alias
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_opencv_core_alias_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv::opencv_core_alias
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_opencv_core_alias_DEPS_TARGET)
        endif()

        set_property(TARGET opencv::opencv_core_alias APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv::opencv_core_alias APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_core_alias APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv::opencv_core_alias APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv::opencv_core_alias APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_RELEASE}>)


    ########## COMPONENT opencv_core #############

        set(opencv_opencv_core_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(opencv_opencv_core_FRAMEWORKS_FOUND_RELEASE "${opencv_opencv_core_FRAMEWORKS_RELEASE}" "${opencv_opencv_core_FRAMEWORK_DIRS_RELEASE}")

        set(opencv_opencv_core_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET opencv_opencv_core_DEPS_TARGET)
            add_library(opencv_opencv_core_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET opencv_opencv_core_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_core_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_core_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_core_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'opencv_opencv_core_DEPS_TARGET' to all of them
        conan_package_library_targets("${opencv_opencv_core_LIBS_RELEASE}"
                              "${opencv_opencv_core_LIB_DIRS_RELEASE}"
                              "${opencv_opencv_core_BIN_DIRS_RELEASE}" # package_bindir
                              "${opencv_opencv_core_LIBRARY_TYPE_RELEASE}"
                              "${opencv_opencv_core_IS_HOST_WINDOWS_RELEASE}"
                              opencv_opencv_core_DEPS_TARGET
                              opencv_opencv_core_LIBRARIES_TARGETS
                              "_RELEASE"
                              "opencv_opencv_core"
                              "${opencv_opencv_core_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET opencv_core
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${opencv_opencv_core_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${opencv_opencv_core_LIBRARIES_TARGETS}>
                     )

        if("${opencv_opencv_core_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET opencv_core
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         opencv_opencv_core_DEPS_TARGET)
        endif()

        set_property(TARGET opencv_core APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_core_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET opencv_core APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_core_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET opencv_core APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${opencv_opencv_core_LIB_DIRS_RELEASE}>)
        set_property(TARGET opencv_core APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${opencv_opencv_core_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET opencv_core APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${opencv_opencv_core_COMPILE_OPTIONS_RELEASE}>)


    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_gapi_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_video_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_stitching_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_objdetect_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_highgui_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_gapi)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_videoio_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_video)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_stitching)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_objdetect)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_highgui)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_calib3d_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_videoio)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_photo_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_imgcodecs_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_features2d_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_dnn_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_calib3d)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_photo)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_ml_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_imgproc_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_imgcodecs)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_flann_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_features2d)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_dnn)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_ml)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_imgproc)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_flann)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv::opencv_core_alias)
    set_property(TARGET opencv::opencv APPEND PROPERTY INTERFACE_LINK_LIBRARIES opencv_core)

########## For the modules (FindXXX)
set(opencv_LIBRARIES_RELEASE opencv::opencv)
