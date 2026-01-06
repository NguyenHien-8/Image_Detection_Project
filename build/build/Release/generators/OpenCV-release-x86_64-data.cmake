########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND opencv_COMPONENT_NAMES opencv_core opencv::opencv_core_alias opencv_flann opencv_imgproc opencv_ml opencv_dnn opencv_features2d opencv::opencv_flann_alias opencv_imgcodecs opencv::opencv_imgproc_alias opencv::opencv_ml_alias opencv_photo opencv_calib3d opencv::opencv_dnn_alias opencv::opencv_features2d_alias opencv::opencv_imgcodecs_alias opencv::opencv_photo_alias opencv_videoio opencv::opencv_calib3d_alias opencv_highgui opencv_objdetect opencv_stitching opencv_video opencv::opencv_videoio_alias opencv_gapi opencv::opencv_highgui_alias opencv::opencv_objdetect_alias opencv::opencv_stitching_alias opencv::opencv_video_alias opencv::opencv_gapi_alias)
list(REMOVE_DUPLICATES opencv_COMPONENT_NAMES)
if(DEFINED opencv_FIND_DEPENDENCY_NAMES)
  list(APPEND opencv_FIND_DEPENDENCY_NAMES protobuf ade gtk OpenJPEG PNG OpenEXR TIFF ZLIB JPEG WebP quirc)
  list(REMOVE_DUPLICATES opencv_FIND_DEPENDENCY_NAMES)
else()
  set(opencv_FIND_DEPENDENCY_NAMES protobuf ade gtk OpenJPEG PNG OpenEXR TIFF ZLIB JPEG WebP quirc)
endif()
set(protobuf_FIND_MODE "NO_MODULE")
set(ade_FIND_MODE "NO_MODULE")
set(gtk_FIND_MODE "NO_MODULE")
set(OpenJPEG_FIND_MODE "NO_MODULE")
set(PNG_FIND_MODE "NO_MODULE")
set(OpenEXR_FIND_MODE "NO_MODULE")
set(TIFF_FIND_MODE "NO_MODULE")
set(ZLIB_FIND_MODE "NO_MODULE")
set(JPEG_FIND_MODE "NO_MODULE")
set(WebP_FIND_MODE "NO_MODULE")
set(quirc_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(opencv_PACKAGE_FOLDER_RELEASE "/home/nguyen-hien/.conan2/p/b/openc5e26d1fdc62a3/p")
set(opencv_BUILD_MODULES_PATHS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib/cmake/conan-official-opencv-variables.cmake")


set(opencv_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_DEFINITIONS_RELEASE )
set(opencv_SHARED_LINK_FLAGS_RELEASE )
set(opencv_EXE_LINK_FLAGS_RELEASE )
set(opencv_OBJECTS_RELEASE )
set(opencv_COMPILE_DEFINITIONS_RELEASE )
set(opencv_COMPILE_OPTIONS_C_RELEASE )
set(opencv_COMPILE_OPTIONS_CXX_RELEASE )
set(opencv_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_BIN_DIRS_RELEASE )
set(opencv_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_LIBS_RELEASE opencv_gapi opencv_video opencv_stitching opencv_objdetect opencv_highgui opencv_videoio opencv_calib3d opencv_photo opencv_imgcodecs opencv_features2d opencv_dnn opencv_ml opencv_imgproc opencv_flann opencv_core)
set(opencv_SYSTEM_LIBS_RELEASE dl m pthread rt)
set(opencv_FRAMEWORK_DIRS_RELEASE )
set(opencv_FRAMEWORKS_RELEASE )
set(opencv_BUILD_DIRS_RELEASE )
set(opencv_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(opencv_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_COMPILE_OPTIONS_C_RELEASE}>")
set(opencv_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_EXE_LINK_FLAGS_RELEASE}>")


set(opencv_COMPONENTS_RELEASE opencv_core opencv::opencv_core_alias opencv_flann opencv_imgproc opencv_ml opencv_dnn opencv_features2d opencv::opencv_flann_alias opencv_imgcodecs opencv::opencv_imgproc_alias opencv::opencv_ml_alias opencv_photo opencv_calib3d opencv::opencv_dnn_alias opencv::opencv_features2d_alias opencv::opencv_imgcodecs_alias opencv::opencv_photo_alias opencv_videoio opencv::opencv_calib3d_alias opencv_highgui opencv_objdetect opencv_stitching opencv_video opencv::opencv_videoio_alias opencv_gapi opencv::opencv_highgui_alias opencv::opencv_objdetect_alias opencv::opencv_stitching_alias opencv::opencv_video_alias opencv::opencv_gapi_alias)
########### COMPONENT opencv::opencv_gapi_alias VARIABLES ############################################

set(opencv_opencv_opencv_gapi_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_gapi_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_gapi_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_gapi_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_gapi_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_DEPENDENCIES_RELEASE opencv_gapi)
set(opencv_opencv_opencv_gapi_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_gapi_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_gapi_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_gapi_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_gapi_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_gapi_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_gapi_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_gapi_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_gapi_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_video_alias VARIABLES ############################################

set(opencv_opencv_opencv_video_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_video_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_video_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_video_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_video_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_video_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_video_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_video_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_video_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_video_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_video_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_video_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_video_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_video_alias_DEPENDENCIES_RELEASE opencv_video)
set(opencv_opencv_opencv_video_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_video_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_video_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_video_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_video_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_video_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_video_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_stitching_alias VARIABLES ############################################

set(opencv_opencv_opencv_stitching_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_stitching_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_stitching_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_stitching_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_DEPENDENCIES_RELEASE opencv_stitching)
set(opencv_opencv_opencv_stitching_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_stitching_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_stitching_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_stitching_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_stitching_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_stitching_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_objdetect_alias VARIABLES ############################################

set(opencv_opencv_opencv_objdetect_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_objdetect_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_objdetect_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_objdetect_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_DEPENDENCIES_RELEASE opencv_objdetect)
set(opencv_opencv_opencv_objdetect_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_objdetect_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_objdetect_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_objdetect_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_objdetect_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_objdetect_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_highgui_alias VARIABLES ############################################

set(opencv_opencv_opencv_highgui_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_highgui_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_highgui_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_highgui_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_DEPENDENCIES_RELEASE opencv_highgui)
set(opencv_opencv_opencv_highgui_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_highgui_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_highgui_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_highgui_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_highgui_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_highgui_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_gapi VARIABLES ############################################

set(opencv_opencv_gapi_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_gapi_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_gapi_BIN_DIRS_RELEASE )
set(opencv_opencv_gapi_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_gapi_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_gapi_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_gapi_DEFINITIONS_RELEASE )
set(opencv_opencv_gapi_OBJECTS_RELEASE )
set(opencv_opencv_gapi_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_gapi_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_gapi_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_gapi_LIBS_RELEASE opencv_gapi)
set(opencv_opencv_gapi_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_gapi_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_gapi_FRAMEWORKS_RELEASE )
set(opencv_opencv_gapi_DEPENDENCIES_RELEASE opencv_imgproc ade opencv_video opencv_calib3d)
set(opencv_opencv_gapi_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_gapi_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_gapi_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_gapi_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_gapi_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_gapi_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_gapi_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_gapi_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_gapi_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_gapi_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_videoio_alias VARIABLES ############################################

set(opencv_opencv_opencv_videoio_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_videoio_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_videoio_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_videoio_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_DEPENDENCIES_RELEASE opencv_videoio)
set(opencv_opencv_opencv_videoio_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_videoio_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_videoio_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_videoio_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_videoio_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_videoio_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_video VARIABLES ############################################

set(opencv_opencv_video_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_video_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_video_BIN_DIRS_RELEASE )
set(opencv_opencv_video_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_video_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_video_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_video_DEFINITIONS_RELEASE )
set(opencv_opencv_video_OBJECTS_RELEASE )
set(opencv_opencv_video_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_video_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_video_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_video_LIBS_RELEASE opencv_video)
set(opencv_opencv_video_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_video_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_video_FRAMEWORKS_RELEASE )
set(opencv_opencv_video_DEPENDENCIES_RELEASE opencv_imgproc opencv_calib3d opencv_dnn)
set(opencv_opencv_video_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_video_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_video_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_video_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_video_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_video_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_video_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_video_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_video_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_video_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_stitching VARIABLES ############################################

set(opencv_opencv_stitching_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_stitching_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_stitching_BIN_DIRS_RELEASE )
set(opencv_opencv_stitching_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_stitching_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_stitching_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_stitching_DEFINITIONS_RELEASE )
set(opencv_opencv_stitching_OBJECTS_RELEASE )
set(opencv_opencv_stitching_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_stitching_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_stitching_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_stitching_LIBS_RELEASE opencv_stitching)
set(opencv_opencv_stitching_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_stitching_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_stitching_FRAMEWORKS_RELEASE )
set(opencv_opencv_stitching_DEPENDENCIES_RELEASE opencv_calib3d opencv_features2d opencv_flann opencv_imgproc)
set(opencv_opencv_stitching_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_stitching_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_stitching_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_stitching_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_stitching_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_stitching_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_stitching_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_stitching_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_stitching_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_stitching_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_objdetect VARIABLES ############################################

set(opencv_opencv_objdetect_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_objdetect_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_objdetect_BIN_DIRS_RELEASE )
set(opencv_opencv_objdetect_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_objdetect_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_objdetect_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_objdetect_DEFINITIONS_RELEASE )
set(opencv_opencv_objdetect_OBJECTS_RELEASE )
set(opencv_opencv_objdetect_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_objdetect_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_objdetect_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_objdetect_LIBS_RELEASE opencv_objdetect)
set(opencv_opencv_objdetect_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_objdetect_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_objdetect_FRAMEWORKS_RELEASE )
set(opencv_opencv_objdetect_DEPENDENCIES_RELEASE opencv_core opencv_calib3d opencv_imgproc quirc::quirc opencv_dnn)
set(opencv_opencv_objdetect_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_objdetect_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_objdetect_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_objdetect_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_objdetect_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_objdetect_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_objdetect_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_objdetect_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_objdetect_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_objdetect_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_highgui VARIABLES ############################################

set(opencv_opencv_highgui_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_highgui_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_highgui_BIN_DIRS_RELEASE )
set(opencv_opencv_highgui_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_highgui_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_highgui_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_highgui_DEFINITIONS_RELEASE )
set(opencv_opencv_highgui_OBJECTS_RELEASE )
set(opencv_opencv_highgui_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_highgui_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_highgui_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_highgui_LIBS_RELEASE opencv_highgui)
set(opencv_opencv_highgui_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_highgui_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_highgui_FRAMEWORKS_RELEASE )
set(opencv_opencv_highgui_DEPENDENCIES_RELEASE opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio gtk::gtk)
set(opencv_opencv_highgui_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_highgui_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_highgui_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_highgui_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_highgui_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_highgui_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_highgui_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_highgui_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_highgui_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_highgui_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_calib3d_alias VARIABLES ############################################

set(opencv_opencv_opencv_calib3d_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_calib3d_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_calib3d_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_calib3d_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_DEPENDENCIES_RELEASE opencv_calib3d)
set(opencv_opencv_opencv_calib3d_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_calib3d_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_calib3d_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_calib3d_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_calib3d_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_calib3d_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_videoio VARIABLES ############################################

set(opencv_opencv_videoio_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_videoio_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_videoio_BIN_DIRS_RELEASE )
set(opencv_opencv_videoio_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_videoio_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_videoio_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_videoio_DEFINITIONS_RELEASE )
set(opencv_opencv_videoio_OBJECTS_RELEASE )
set(opencv_opencv_videoio_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_videoio_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_videoio_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_videoio_LIBS_RELEASE opencv_videoio)
set(opencv_opencv_videoio_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_videoio_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_videoio_FRAMEWORKS_RELEASE )
set(opencv_opencv_videoio_DEPENDENCIES_RELEASE opencv_imgcodecs opencv_imgproc)
set(opencv_opencv_videoio_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_videoio_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_videoio_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_videoio_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_videoio_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_videoio_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_videoio_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_videoio_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_videoio_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_videoio_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_photo_alias VARIABLES ############################################

set(opencv_opencv_opencv_photo_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_photo_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_photo_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_photo_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_photo_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_photo_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_photo_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_photo_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_photo_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_photo_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_photo_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_photo_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_photo_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_photo_alias_DEPENDENCIES_RELEASE opencv_photo)
set(opencv_opencv_opencv_photo_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_photo_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_photo_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_photo_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_photo_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_photo_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_photo_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_imgcodecs_alias VARIABLES ############################################

set(opencv_opencv_opencv_imgcodecs_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_imgcodecs_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_imgcodecs_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_imgcodecs_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_DEPENDENCIES_RELEASE opencv_imgcodecs)
set(opencv_opencv_opencv_imgcodecs_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_imgcodecs_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_imgcodecs_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_imgcodecs_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_imgcodecs_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_imgcodecs_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_features2d_alias VARIABLES ############################################

set(opencv_opencv_opencv_features2d_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_features2d_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_features2d_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_features2d_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_DEPENDENCIES_RELEASE opencv_features2d)
set(opencv_opencv_opencv_features2d_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_features2d_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_features2d_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_features2d_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_features2d_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_features2d_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_dnn_alias VARIABLES ############################################

set(opencv_opencv_opencv_dnn_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_dnn_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_dnn_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_dnn_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_dnn_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_DEPENDENCIES_RELEASE opencv_dnn)
set(opencv_opencv_opencv_dnn_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_dnn_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_dnn_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_dnn_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_dnn_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_dnn_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_dnn_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_dnn_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_dnn_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_calib3d VARIABLES ############################################

set(opencv_opencv_calib3d_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_calib3d_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_calib3d_BIN_DIRS_RELEASE )
set(opencv_opencv_calib3d_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_calib3d_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_calib3d_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_calib3d_DEFINITIONS_RELEASE )
set(opencv_opencv_calib3d_OBJECTS_RELEASE )
set(opencv_opencv_calib3d_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_calib3d_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_calib3d_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_calib3d_LIBS_RELEASE opencv_calib3d)
set(opencv_opencv_calib3d_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_calib3d_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_calib3d_FRAMEWORKS_RELEASE )
set(opencv_opencv_calib3d_DEPENDENCIES_RELEASE opencv_core opencv_features2d opencv_flann opencv_imgproc)
set(opencv_opencv_calib3d_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_calib3d_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_calib3d_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_calib3d_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_calib3d_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_calib3d_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_calib3d_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_calib3d_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_calib3d_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_calib3d_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_photo VARIABLES ############################################

set(opencv_opencv_photo_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_photo_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_photo_BIN_DIRS_RELEASE )
set(opencv_opencv_photo_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_photo_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_photo_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_photo_DEFINITIONS_RELEASE )
set(opencv_opencv_photo_OBJECTS_RELEASE )
set(opencv_opencv_photo_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_photo_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_photo_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_photo_LIBS_RELEASE opencv_photo)
set(opencv_opencv_photo_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_photo_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_photo_FRAMEWORKS_RELEASE )
set(opencv_opencv_photo_DEPENDENCIES_RELEASE opencv_imgproc)
set(opencv_opencv_photo_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_photo_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_photo_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_photo_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_photo_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_photo_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_photo_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_photo_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_photo_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_photo_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_ml_alias VARIABLES ############################################

set(opencv_opencv_opencv_ml_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_ml_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_ml_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_ml_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_ml_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_ml_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_ml_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_ml_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_ml_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_ml_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_ml_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_ml_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_ml_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_ml_alias_DEPENDENCIES_RELEASE opencv_ml)
set(opencv_opencv_opencv_ml_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_ml_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_ml_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_ml_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_ml_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_ml_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_ml_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_imgproc_alias VARIABLES ############################################

set(opencv_opencv_opencv_imgproc_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_imgproc_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_imgproc_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_imgproc_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_DEPENDENCIES_RELEASE opencv_imgproc)
set(opencv_opencv_opencv_imgproc_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_imgproc_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_imgproc_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_imgproc_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_imgproc_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_imgproc_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_imgcodecs VARIABLES ############################################

set(opencv_opencv_imgcodecs_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_imgcodecs_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_imgcodecs_BIN_DIRS_RELEASE )
set(opencv_opencv_imgcodecs_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_imgcodecs_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_imgcodecs_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_imgcodecs_DEFINITIONS_RELEASE )
set(opencv_opencv_imgcodecs_OBJECTS_RELEASE )
set(opencv_opencv_imgcodecs_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_imgcodecs_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_imgcodecs_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_imgcodecs_LIBS_RELEASE opencv_imgcodecs)
set(opencv_opencv_imgcodecs_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_imgcodecs_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_imgcodecs_FRAMEWORKS_RELEASE )
set(opencv_opencv_imgcodecs_DEPENDENCIES_RELEASE opencv_imgproc ZLIB::ZLIB openjp2 PNG::PNG JPEG::JPEG TIFF::TIFF openexr::openexr libwebp::libwebp)
set(opencv_opencv_imgcodecs_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_imgcodecs_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_imgcodecs_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_imgcodecs_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_imgcodecs_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_imgcodecs_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_imgcodecs_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_imgcodecs_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_imgcodecs_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_imgcodecs_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_flann_alias VARIABLES ############################################

set(opencv_opencv_opencv_flann_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_flann_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_flann_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_flann_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_flann_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_flann_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_flann_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_flann_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_flann_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_flann_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_flann_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_flann_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_flann_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_flann_alias_DEPENDENCIES_RELEASE opencv_flann)
set(opencv_opencv_opencv_flann_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_flann_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_flann_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_flann_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_flann_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_flann_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_flann_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_features2d VARIABLES ############################################

set(opencv_opencv_features2d_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_features2d_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_features2d_BIN_DIRS_RELEASE )
set(opencv_opencv_features2d_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_features2d_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_features2d_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_features2d_DEFINITIONS_RELEASE )
set(opencv_opencv_features2d_OBJECTS_RELEASE )
set(opencv_opencv_features2d_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_features2d_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_features2d_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_features2d_LIBS_RELEASE opencv_features2d)
set(opencv_opencv_features2d_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_features2d_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_features2d_FRAMEWORKS_RELEASE )
set(opencv_opencv_features2d_DEPENDENCIES_RELEASE opencv_imgproc opencv_flann)
set(opencv_opencv_features2d_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_features2d_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_features2d_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_features2d_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_features2d_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_features2d_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_features2d_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_features2d_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_features2d_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_features2d_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_dnn VARIABLES ############################################

set(opencv_opencv_dnn_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_dnn_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_dnn_BIN_DIRS_RELEASE )
set(opencv_opencv_dnn_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_dnn_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_dnn_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_dnn_DEFINITIONS_RELEASE )
set(opencv_opencv_dnn_OBJECTS_RELEASE )
set(opencv_opencv_dnn_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_dnn_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_dnn_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_dnn_LIBS_RELEASE opencv_dnn)
set(opencv_opencv_dnn_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_dnn_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_dnn_FRAMEWORKS_RELEASE )
set(opencv_opencv_dnn_DEPENDENCIES_RELEASE opencv_core opencv_imgproc protobuf::protobuf)
set(opencv_opencv_dnn_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_dnn_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_dnn_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_dnn_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_dnn_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_dnn_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_dnn_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_dnn_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_dnn_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_dnn_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_ml VARIABLES ############################################

set(opencv_opencv_ml_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_ml_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_ml_BIN_DIRS_RELEASE )
set(opencv_opencv_ml_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_ml_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_ml_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_ml_DEFINITIONS_RELEASE )
set(opencv_opencv_ml_OBJECTS_RELEASE )
set(opencv_opencv_ml_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_ml_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_ml_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_ml_LIBS_RELEASE opencv_ml)
set(opencv_opencv_ml_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_ml_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_ml_FRAMEWORKS_RELEASE )
set(opencv_opencv_ml_DEPENDENCIES_RELEASE opencv_core)
set(opencv_opencv_ml_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_ml_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_ml_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_ml_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_ml_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_ml_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_ml_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_ml_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_ml_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_ml_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_imgproc VARIABLES ############################################

set(opencv_opencv_imgproc_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_imgproc_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_imgproc_BIN_DIRS_RELEASE )
set(opencv_opencv_imgproc_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_imgproc_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_imgproc_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_imgproc_DEFINITIONS_RELEASE )
set(opencv_opencv_imgproc_OBJECTS_RELEASE )
set(opencv_opencv_imgproc_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_imgproc_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_imgproc_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_imgproc_LIBS_RELEASE opencv_imgproc)
set(opencv_opencv_imgproc_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_imgproc_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_imgproc_FRAMEWORKS_RELEASE )
set(opencv_opencv_imgproc_DEPENDENCIES_RELEASE opencv_core)
set(opencv_opencv_imgproc_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_imgproc_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_imgproc_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_imgproc_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_imgproc_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_imgproc_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_imgproc_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_imgproc_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_imgproc_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_imgproc_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_flann VARIABLES ############################################

set(opencv_opencv_flann_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_flann_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_flann_BIN_DIRS_RELEASE )
set(opencv_opencv_flann_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_flann_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_flann_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_flann_DEFINITIONS_RELEASE )
set(opencv_opencv_flann_OBJECTS_RELEASE )
set(opencv_opencv_flann_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_flann_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_flann_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_flann_LIBS_RELEASE opencv_flann)
set(opencv_opencv_flann_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_flann_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_flann_FRAMEWORKS_RELEASE )
set(opencv_opencv_flann_DEPENDENCIES_RELEASE opencv_core)
set(opencv_opencv_flann_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_flann_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_flann_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_flann_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_flann_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_flann_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_flann_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_flann_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_flann_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_flann_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv::opencv_core_alias VARIABLES ############################################

set(opencv_opencv_opencv_core_alias_INCLUDE_DIRS_RELEASE )
set(opencv_opencv_opencv_core_alias_LIB_DIRS_RELEASE )
set(opencv_opencv_opencv_core_alias_BIN_DIRS_RELEASE )
set(opencv_opencv_opencv_core_alias_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_opencv_core_alias_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_opencv_core_alias_RES_DIRS_RELEASE )
set(opencv_opencv_opencv_core_alias_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_core_alias_OBJECTS_RELEASE )
set(opencv_opencv_opencv_core_alias_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_opencv_core_alias_LIBS_RELEASE )
set(opencv_opencv_opencv_core_alias_SYSTEM_LIBS_RELEASE )
set(opencv_opencv_opencv_core_alias_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_opencv_core_alias_FRAMEWORKS_RELEASE )
set(opencv_opencv_opencv_core_alias_DEPENDENCIES_RELEASE opencv_core)
set(opencv_opencv_opencv_core_alias_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_core_alias_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_opencv_core_alias_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_opencv_core_alias_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_opencv_core_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_opencv_core_alias_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_opencv_core_alias_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT opencv_core VARIABLES ############################################

set(opencv_opencv_core_INCLUDE_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/include"
			"${opencv_PACKAGE_FOLDER_RELEASE}/include/opencv4")
set(opencv_opencv_core_LIB_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/lib")
set(opencv_opencv_core_BIN_DIRS_RELEASE )
set(opencv_opencv_core_LIBRARY_TYPE_RELEASE STATIC)
set(opencv_opencv_core_IS_HOST_WINDOWS_RELEASE 0)
set(opencv_opencv_core_RES_DIRS_RELEASE "${opencv_PACKAGE_FOLDER_RELEASE}/res")
set(opencv_opencv_core_DEFINITIONS_RELEASE )
set(opencv_opencv_core_OBJECTS_RELEASE )
set(opencv_opencv_core_COMPILE_DEFINITIONS_RELEASE )
set(opencv_opencv_core_COMPILE_OPTIONS_C_RELEASE "")
set(opencv_opencv_core_COMPILE_OPTIONS_CXX_RELEASE "")
set(opencv_opencv_core_LIBS_RELEASE opencv_core)
set(opencv_opencv_core_SYSTEM_LIBS_RELEASE dl m pthread rt)
set(opencv_opencv_core_FRAMEWORK_DIRS_RELEASE )
set(opencv_opencv_core_FRAMEWORKS_RELEASE )
set(opencv_opencv_core_DEPENDENCIES_RELEASE ZLIB::ZLIB)
set(opencv_opencv_core_SHARED_LINK_FLAGS_RELEASE )
set(opencv_opencv_core_EXE_LINK_FLAGS_RELEASE )
set(opencv_opencv_core_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(opencv_opencv_core_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_opencv_core_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_opencv_core_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_opencv_core_EXE_LINK_FLAGS_RELEASE}>
)
set(opencv_opencv_core_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_opencv_core_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_opencv_core_COMPILE_OPTIONS_C_RELEASE}>")