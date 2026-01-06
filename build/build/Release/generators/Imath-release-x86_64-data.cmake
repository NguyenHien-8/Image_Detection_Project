########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND imath_COMPONENT_NAMES Imath::ImathConfig Imath::Imath)
list(REMOVE_DUPLICATES imath_COMPONENT_NAMES)
if(DEFINED imath_FIND_DEPENDENCY_NAMES)
  list(APPEND imath_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES imath_FIND_DEPENDENCY_NAMES)
else()
  set(imath_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(imath_PACKAGE_FOLDER_RELEASE "/home/nguyen-hien/.conan2/p/imath64bc612641a6b/p")
set(imath_BUILD_MODULES_PATHS_RELEASE )


set(imath_INCLUDE_DIRS_RELEASE )
set(imath_RES_DIRS_RELEASE )
set(imath_DEFINITIONS_RELEASE )
set(imath_SHARED_LINK_FLAGS_RELEASE )
set(imath_EXE_LINK_FLAGS_RELEASE )
set(imath_OBJECTS_RELEASE )
set(imath_COMPILE_DEFINITIONS_RELEASE )
set(imath_COMPILE_OPTIONS_C_RELEASE )
set(imath_COMPILE_OPTIONS_CXX_RELEASE )
set(imath_LIB_DIRS_RELEASE "${imath_PACKAGE_FOLDER_RELEASE}/lib")
set(imath_BIN_DIRS_RELEASE )
set(imath_LIBRARY_TYPE_RELEASE STATIC)
set(imath_IS_HOST_WINDOWS_RELEASE 0)
set(imath_LIBS_RELEASE Imath-3_1)
set(imath_SYSTEM_LIBS_RELEASE )
set(imath_FRAMEWORK_DIRS_RELEASE )
set(imath_FRAMEWORKS_RELEASE )
set(imath_BUILD_DIRS_RELEASE )
set(imath_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(imath_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${imath_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${imath_COMPILE_OPTIONS_C_RELEASE}>")
set(imath_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${imath_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${imath_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${imath_EXE_LINK_FLAGS_RELEASE}>")


set(imath_COMPONENTS_RELEASE Imath::ImathConfig Imath::Imath)
########### COMPONENT Imath::Imath VARIABLES ############################################

set(imath_Imath_Imath_INCLUDE_DIRS_RELEASE )
set(imath_Imath_Imath_LIB_DIRS_RELEASE "${imath_PACKAGE_FOLDER_RELEASE}/lib")
set(imath_Imath_Imath_BIN_DIRS_RELEASE )
set(imath_Imath_Imath_LIBRARY_TYPE_RELEASE STATIC)
set(imath_Imath_Imath_IS_HOST_WINDOWS_RELEASE 0)
set(imath_Imath_Imath_RES_DIRS_RELEASE )
set(imath_Imath_Imath_DEFINITIONS_RELEASE )
set(imath_Imath_Imath_OBJECTS_RELEASE )
set(imath_Imath_Imath_COMPILE_DEFINITIONS_RELEASE )
set(imath_Imath_Imath_COMPILE_OPTIONS_C_RELEASE "")
set(imath_Imath_Imath_COMPILE_OPTIONS_CXX_RELEASE "")
set(imath_Imath_Imath_LIBS_RELEASE Imath-3_1)
set(imath_Imath_Imath_SYSTEM_LIBS_RELEASE )
set(imath_Imath_Imath_FRAMEWORK_DIRS_RELEASE )
set(imath_Imath_Imath_FRAMEWORKS_RELEASE )
set(imath_Imath_Imath_DEPENDENCIES_RELEASE Imath::ImathConfig)
set(imath_Imath_Imath_SHARED_LINK_FLAGS_RELEASE )
set(imath_Imath_Imath_EXE_LINK_FLAGS_RELEASE )
set(imath_Imath_Imath_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(imath_Imath_Imath_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${imath_Imath_Imath_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${imath_Imath_Imath_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${imath_Imath_Imath_EXE_LINK_FLAGS_RELEASE}>
)
set(imath_Imath_Imath_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${imath_Imath_Imath_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${imath_Imath_Imath_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT Imath::ImathConfig VARIABLES ############################################

set(imath_Imath_ImathConfig_INCLUDE_DIRS_RELEASE )
set(imath_Imath_ImathConfig_LIB_DIRS_RELEASE "${imath_PACKAGE_FOLDER_RELEASE}/lib")
set(imath_Imath_ImathConfig_BIN_DIRS_RELEASE )
set(imath_Imath_ImathConfig_LIBRARY_TYPE_RELEASE STATIC)
set(imath_Imath_ImathConfig_IS_HOST_WINDOWS_RELEASE 0)
set(imath_Imath_ImathConfig_RES_DIRS_RELEASE )
set(imath_Imath_ImathConfig_DEFINITIONS_RELEASE )
set(imath_Imath_ImathConfig_OBJECTS_RELEASE )
set(imath_Imath_ImathConfig_COMPILE_DEFINITIONS_RELEASE )
set(imath_Imath_ImathConfig_COMPILE_OPTIONS_C_RELEASE "")
set(imath_Imath_ImathConfig_COMPILE_OPTIONS_CXX_RELEASE "")
set(imath_Imath_ImathConfig_LIBS_RELEASE )
set(imath_Imath_ImathConfig_SYSTEM_LIBS_RELEASE )
set(imath_Imath_ImathConfig_FRAMEWORK_DIRS_RELEASE )
set(imath_Imath_ImathConfig_FRAMEWORKS_RELEASE )
set(imath_Imath_ImathConfig_DEPENDENCIES_RELEASE )
set(imath_Imath_ImathConfig_SHARED_LINK_FLAGS_RELEASE )
set(imath_Imath_ImathConfig_EXE_LINK_FLAGS_RELEASE )
set(imath_Imath_ImathConfig_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(imath_Imath_ImathConfig_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${imath_Imath_ImathConfig_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${imath_Imath_ImathConfig_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${imath_Imath_ImathConfig_EXE_LINK_FLAGS_RELEASE}>
)
set(imath_Imath_ImathConfig_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${imath_Imath_ImathConfig_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${imath_Imath_ImathConfig_COMPILE_OPTIONS_C_RELEASE}>")