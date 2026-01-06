########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(gtk_COMPONENT_NAMES "")
if(DEFINED gtk_FIND_DEPENDENCY_NAMES)
  list(APPEND gtk_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES gtk_FIND_DEPENDENCY_NAMES)
else()
  set(gtk_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(gtk_PACKAGE_FOLDER_RELEASE "/home/nguyen-hien/.conan2/p/gtkbc9a62a2c0463/p")
set(gtk_BUILD_MODULES_PATHS_RELEASE )


set(gtk_INCLUDE_DIRS_RELEASE )
set(gtk_RES_DIRS_RELEASE )
set(gtk_DEFINITIONS_RELEASE )
set(gtk_SHARED_LINK_FLAGS_RELEASE )
set(gtk_EXE_LINK_FLAGS_RELEASE )
set(gtk_OBJECTS_RELEASE )
set(gtk_COMPILE_DEFINITIONS_RELEASE )
set(gtk_COMPILE_OPTIONS_C_RELEASE -pthread)
set(gtk_COMPILE_OPTIONS_CXX_RELEASE -pthread)
set(gtk_LIB_DIRS_RELEASE )
set(gtk_BIN_DIRS_RELEASE "${gtk_PACKAGE_FOLDER_RELEASE}/bin")
set(gtk_LIBRARY_TYPE_RELEASE SHARED)
set(gtk_IS_HOST_WINDOWS_RELEASE 0)
set(gtk_LIBS_RELEASE )
set(gtk_SYSTEM_LIBS_RELEASE gtk-x11-2.0 gdk-x11-2.0 pangocairo-1.0 atk-1.0 cairo gdk_pixbuf-2.0 gio-2.0 pangoft2-1.0 pango-1.0 gobject-2.0 glib-2.0 harfbuzz fontconfig freetype)
set(gtk_FRAMEWORK_DIRS_RELEASE )
set(gtk_FRAMEWORKS_RELEASE )
set(gtk_BUILD_DIRS_RELEASE )
set(gtk_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(gtk_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${gtk_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${gtk_COMPILE_OPTIONS_C_RELEASE}>")
set(gtk_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${gtk_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${gtk_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${gtk_EXE_LINK_FLAGS_RELEASE}>")


set(gtk_COMPONENTS_RELEASE )