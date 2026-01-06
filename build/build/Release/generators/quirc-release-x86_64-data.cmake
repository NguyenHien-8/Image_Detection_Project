########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(quirc_COMPONENT_NAMES "")
if(DEFINED quirc_FIND_DEPENDENCY_NAMES)
  list(APPEND quirc_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES quirc_FIND_DEPENDENCY_NAMES)
else()
  set(quirc_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(quirc_PACKAGE_FOLDER_RELEASE "/home/nguyen-hien/.conan2/p/quirc09fba66b85dea/p")
set(quirc_BUILD_MODULES_PATHS_RELEASE )


set(quirc_INCLUDE_DIRS_RELEASE )
set(quirc_RES_DIRS_RELEASE )
set(quirc_DEFINITIONS_RELEASE )
set(quirc_SHARED_LINK_FLAGS_RELEASE )
set(quirc_EXE_LINK_FLAGS_RELEASE )
set(quirc_OBJECTS_RELEASE )
set(quirc_COMPILE_DEFINITIONS_RELEASE )
set(quirc_COMPILE_OPTIONS_C_RELEASE )
set(quirc_COMPILE_OPTIONS_CXX_RELEASE )
set(quirc_LIB_DIRS_RELEASE "${quirc_PACKAGE_FOLDER_RELEASE}/lib")
set(quirc_BIN_DIRS_RELEASE )
set(quirc_LIBRARY_TYPE_RELEASE STATIC)
set(quirc_IS_HOST_WINDOWS_RELEASE 0)
set(quirc_LIBS_RELEASE quirc)
set(quirc_SYSTEM_LIBS_RELEASE m)
set(quirc_FRAMEWORK_DIRS_RELEASE )
set(quirc_FRAMEWORKS_RELEASE )
set(quirc_BUILD_DIRS_RELEASE )
set(quirc_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(quirc_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${quirc_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${quirc_COMPILE_OPTIONS_C_RELEASE}>")
set(quirc_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${quirc_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${quirc_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${quirc_EXE_LINK_FLAGS_RELEASE}>")


set(quirc_COMPONENTS_RELEASE )