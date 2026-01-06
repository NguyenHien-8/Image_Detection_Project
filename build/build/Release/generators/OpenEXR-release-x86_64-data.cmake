########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND openexr_COMPONENT_NAMES OpenEXR::OpenEXRConfig OpenEXR::IexConfig OpenEXR::IlmThreadConfig OpenEXR::Iex OpenEXR::OpenEXRCore OpenEXR::IlmThread OpenEXR::OpenEXR OpenEXR::OpenEXRUtil)
list(REMOVE_DUPLICATES openexr_COMPONENT_NAMES)
if(DEFINED openexr_FIND_DEPENDENCY_NAMES)
  list(APPEND openexr_FIND_DEPENDENCY_NAMES Imath libdeflate ZLIB)
  list(REMOVE_DUPLICATES openexr_FIND_DEPENDENCY_NAMES)
else()
  set(openexr_FIND_DEPENDENCY_NAMES Imath libdeflate ZLIB)
endif()
set(Imath_FIND_MODE "NO_MODULE")
set(libdeflate_FIND_MODE "NO_MODULE")
set(ZLIB_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(openexr_PACKAGE_FOLDER_RELEASE "/home/nguyen-hien/.conan2/p/b/openec415f9807fcc3/p")
set(openexr_BUILD_MODULES_PATHS_RELEASE )


set(openexr_INCLUDE_DIRS_RELEASE )
set(openexr_RES_DIRS_RELEASE )
set(openexr_DEFINITIONS_RELEASE )
set(openexr_SHARED_LINK_FLAGS_RELEASE )
set(openexr_EXE_LINK_FLAGS_RELEASE )
set(openexr_OBJECTS_RELEASE )
set(openexr_COMPILE_DEFINITIONS_RELEASE )
set(openexr_COMPILE_OPTIONS_C_RELEASE )
set(openexr_COMPILE_OPTIONS_CXX_RELEASE )
set(openexr_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_BIN_DIRS_RELEASE )
set(openexr_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_LIBS_RELEASE OpenEXRUtil-3_2 OpenEXR-3_2 IlmThread-3_2 OpenEXRCore-3_2 Iex-3_2)
set(openexr_SYSTEM_LIBS_RELEASE m pthread)
set(openexr_FRAMEWORK_DIRS_RELEASE )
set(openexr_FRAMEWORKS_RELEASE )
set(openexr_BUILD_DIRS_RELEASE )
set(openexr_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(openexr_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_COMPILE_OPTIONS_C_RELEASE}>")
set(openexr_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_EXE_LINK_FLAGS_RELEASE}>")


set(openexr_COMPONENTS_RELEASE OpenEXR::OpenEXRConfig OpenEXR::IexConfig OpenEXR::IlmThreadConfig OpenEXR::Iex OpenEXR::OpenEXRCore OpenEXR::IlmThread OpenEXR::OpenEXR OpenEXR::OpenEXRUtil)
########### COMPONENT OpenEXR::OpenEXRUtil VARIABLES ############################################

set(openexr_OpenEXR_OpenEXRUtil_INCLUDE_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_OpenEXR_OpenEXRUtil_BIN_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_OpenEXR_OpenEXRUtil_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_OpenEXR_OpenEXRUtil_RES_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_OBJECTS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_COMPILE_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_C_RELEASE "")
set(openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_CXX_RELEASE "")
set(openexr_OpenEXR_OpenEXRUtil_LIBS_RELEASE OpenEXRUtil-3_2)
set(openexr_OpenEXR_OpenEXRUtil_SYSTEM_LIBS_RELEASE m)
set(openexr_OpenEXR_OpenEXRUtil_FRAMEWORK_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_FRAMEWORKS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_DEPENDENCIES_RELEASE OpenEXR::OpenEXR)
set(openexr_OpenEXR_OpenEXRUtil_SHARED_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_EXE_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_OpenEXRUtil_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_OpenEXRUtil_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_OpenEXRUtil_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_OpenEXRUtil_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_OpenEXRUtil_EXE_LINK_FLAGS_RELEASE}>
)
set(openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_OpenEXRUtil_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT OpenEXR::OpenEXR VARIABLES ############################################

set(openexr_OpenEXR_OpenEXR_INCLUDE_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXR_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_OpenEXR_OpenEXR_BIN_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXR_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_OpenEXR_OpenEXR_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_OpenEXR_OpenEXR_RES_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXR_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_OpenEXR_OBJECTS_RELEASE )
set(openexr_OpenEXR_OpenEXR_COMPILE_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_C_RELEASE "")
set(openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_CXX_RELEASE "")
set(openexr_OpenEXR_OpenEXR_LIBS_RELEASE OpenEXR-3_2)
set(openexr_OpenEXR_OpenEXR_SYSTEM_LIBS_RELEASE m)
set(openexr_OpenEXR_OpenEXR_FRAMEWORK_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXR_FRAMEWORKS_RELEASE )
set(openexr_OpenEXR_OpenEXR_DEPENDENCIES_RELEASE OpenEXR::OpenEXRCore OpenEXR::IlmThread OpenEXR::Iex Imath::Imath)
set(openexr_OpenEXR_OpenEXR_SHARED_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_OpenEXR_EXE_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_OpenEXR_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_OpenEXR_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_OpenEXR_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_OpenEXR_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_OpenEXR_EXE_LINK_FLAGS_RELEASE}>
)
set(openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_OpenEXR_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT OpenEXR::IlmThread VARIABLES ############################################

set(openexr_OpenEXR_IlmThread_INCLUDE_DIRS_RELEASE )
set(openexr_OpenEXR_IlmThread_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_OpenEXR_IlmThread_BIN_DIRS_RELEASE )
set(openexr_OpenEXR_IlmThread_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_OpenEXR_IlmThread_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_OpenEXR_IlmThread_RES_DIRS_RELEASE )
set(openexr_OpenEXR_IlmThread_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_IlmThread_OBJECTS_RELEASE )
set(openexr_OpenEXR_IlmThread_COMPILE_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_C_RELEASE "")
set(openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_CXX_RELEASE "")
set(openexr_OpenEXR_IlmThread_LIBS_RELEASE IlmThread-3_2)
set(openexr_OpenEXR_IlmThread_SYSTEM_LIBS_RELEASE pthread m)
set(openexr_OpenEXR_IlmThread_FRAMEWORK_DIRS_RELEASE )
set(openexr_OpenEXR_IlmThread_FRAMEWORKS_RELEASE )
set(openexr_OpenEXR_IlmThread_DEPENDENCIES_RELEASE OpenEXR::IlmThreadConfig OpenEXR::Iex)
set(openexr_OpenEXR_IlmThread_SHARED_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_IlmThread_EXE_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_IlmThread_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_IlmThread_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_IlmThread_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_IlmThread_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_IlmThread_EXE_LINK_FLAGS_RELEASE}>
)
set(openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_IlmThread_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT OpenEXR::OpenEXRCore VARIABLES ############################################

set(openexr_OpenEXR_OpenEXRCore_INCLUDE_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_OpenEXR_OpenEXRCore_BIN_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_OpenEXR_OpenEXRCore_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_OpenEXR_OpenEXRCore_RES_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_OBJECTS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_COMPILE_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_C_RELEASE "")
set(openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_CXX_RELEASE "")
set(openexr_OpenEXR_OpenEXRCore_LIBS_RELEASE OpenEXRCore-3_2)
set(openexr_OpenEXR_OpenEXRCore_SYSTEM_LIBS_RELEASE m)
set(openexr_OpenEXR_OpenEXRCore_FRAMEWORK_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_FRAMEWORKS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_DEPENDENCIES_RELEASE OpenEXR::OpenEXRConfig ZLIB::ZLIB libdeflate::libdeflate_static)
set(openexr_OpenEXR_OpenEXRCore_SHARED_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_EXE_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_OpenEXRCore_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_OpenEXRCore_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_OpenEXRCore_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_OpenEXRCore_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_OpenEXRCore_EXE_LINK_FLAGS_RELEASE}>
)
set(openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_OpenEXRCore_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT OpenEXR::Iex VARIABLES ############################################

set(openexr_OpenEXR_Iex_INCLUDE_DIRS_RELEASE )
set(openexr_OpenEXR_Iex_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_OpenEXR_Iex_BIN_DIRS_RELEASE )
set(openexr_OpenEXR_Iex_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_OpenEXR_Iex_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_OpenEXR_Iex_RES_DIRS_RELEASE )
set(openexr_OpenEXR_Iex_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_Iex_OBJECTS_RELEASE )
set(openexr_OpenEXR_Iex_COMPILE_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_Iex_COMPILE_OPTIONS_C_RELEASE "")
set(openexr_OpenEXR_Iex_COMPILE_OPTIONS_CXX_RELEASE "")
set(openexr_OpenEXR_Iex_LIBS_RELEASE Iex-3_2)
set(openexr_OpenEXR_Iex_SYSTEM_LIBS_RELEASE m)
set(openexr_OpenEXR_Iex_FRAMEWORK_DIRS_RELEASE )
set(openexr_OpenEXR_Iex_FRAMEWORKS_RELEASE )
set(openexr_OpenEXR_Iex_DEPENDENCIES_RELEASE OpenEXR::IexConfig)
set(openexr_OpenEXR_Iex_SHARED_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_Iex_EXE_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_Iex_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_Iex_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_Iex_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_Iex_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_Iex_EXE_LINK_FLAGS_RELEASE}>
)
set(openexr_OpenEXR_Iex_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_Iex_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_Iex_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT OpenEXR::IlmThreadConfig VARIABLES ############################################

set(openexr_OpenEXR_IlmThreadConfig_INCLUDE_DIRS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_OpenEXR_IlmThreadConfig_BIN_DIRS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_OpenEXR_IlmThreadConfig_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_OpenEXR_IlmThreadConfig_RES_DIRS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_OBJECTS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_COMPILE_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_C_RELEASE "")
set(openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_CXX_RELEASE "")
set(openexr_OpenEXR_IlmThreadConfig_LIBS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_SYSTEM_LIBS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_FRAMEWORK_DIRS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_FRAMEWORKS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_DEPENDENCIES_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_SHARED_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_EXE_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_IlmThreadConfig_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_IlmThreadConfig_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_IlmThreadConfig_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_IlmThreadConfig_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_IlmThreadConfig_EXE_LINK_FLAGS_RELEASE}>
)
set(openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_IlmThreadConfig_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT OpenEXR::IexConfig VARIABLES ############################################

set(openexr_OpenEXR_IexConfig_INCLUDE_DIRS_RELEASE )
set(openexr_OpenEXR_IexConfig_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_OpenEXR_IexConfig_BIN_DIRS_RELEASE )
set(openexr_OpenEXR_IexConfig_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_OpenEXR_IexConfig_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_OpenEXR_IexConfig_RES_DIRS_RELEASE )
set(openexr_OpenEXR_IexConfig_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_IexConfig_OBJECTS_RELEASE )
set(openexr_OpenEXR_IexConfig_COMPILE_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_C_RELEASE "")
set(openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_CXX_RELEASE "")
set(openexr_OpenEXR_IexConfig_LIBS_RELEASE )
set(openexr_OpenEXR_IexConfig_SYSTEM_LIBS_RELEASE )
set(openexr_OpenEXR_IexConfig_FRAMEWORK_DIRS_RELEASE )
set(openexr_OpenEXR_IexConfig_FRAMEWORKS_RELEASE )
set(openexr_OpenEXR_IexConfig_DEPENDENCIES_RELEASE )
set(openexr_OpenEXR_IexConfig_SHARED_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_IexConfig_EXE_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_IexConfig_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_IexConfig_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_IexConfig_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_IexConfig_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_IexConfig_EXE_LINK_FLAGS_RELEASE}>
)
set(openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_IexConfig_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT OpenEXR::OpenEXRConfig VARIABLES ############################################

set(openexr_OpenEXR_OpenEXRConfig_INCLUDE_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_LIB_DIRS_RELEASE "${openexr_PACKAGE_FOLDER_RELEASE}/lib")
set(openexr_OpenEXR_OpenEXRConfig_BIN_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_LIBRARY_TYPE_RELEASE STATIC)
set(openexr_OpenEXR_OpenEXRConfig_IS_HOST_WINDOWS_RELEASE 0)
set(openexr_OpenEXR_OpenEXRConfig_RES_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_OBJECTS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_COMPILE_DEFINITIONS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_C_RELEASE "")
set(openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_CXX_RELEASE "")
set(openexr_OpenEXR_OpenEXRConfig_LIBS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_SYSTEM_LIBS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_FRAMEWORK_DIRS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_FRAMEWORKS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_DEPENDENCIES_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_SHARED_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_EXE_LINK_FLAGS_RELEASE )
set(openexr_OpenEXR_OpenEXRConfig_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openexr_OpenEXR_OpenEXRConfig_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openexr_OpenEXR_OpenEXRConfig_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openexr_OpenEXR_OpenEXRConfig_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openexr_OpenEXR_OpenEXRConfig_EXE_LINK_FLAGS_RELEASE}>
)
set(openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openexr_OpenEXR_OpenEXRConfig_COMPILE_OPTIONS_C_RELEASE}>")