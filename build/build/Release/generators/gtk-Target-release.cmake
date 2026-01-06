# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(gtk_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(gtk_FRAMEWORKS_FOUND_RELEASE "${gtk_FRAMEWORKS_RELEASE}" "${gtk_FRAMEWORK_DIRS_RELEASE}")

set(gtk_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET gtk_DEPS_TARGET)
    add_library(gtk_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET gtk_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${gtk_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${gtk_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### gtk_DEPS_TARGET to all of them
conan_package_library_targets("${gtk_LIBS_RELEASE}"    # libraries
                              "${gtk_LIB_DIRS_RELEASE}" # package_libdir
                              "${gtk_BIN_DIRS_RELEASE}" # package_bindir
                              "${gtk_LIBRARY_TYPE_RELEASE}"
                              "${gtk_IS_HOST_WINDOWS_RELEASE}"
                              gtk_DEPS_TARGET
                              gtk_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "gtk"    # package_name
                              "${gtk_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${gtk_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET gtk::gtk
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${gtk_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${gtk_LIBRARIES_TARGETS}>
                 )

    if("${gtk_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET gtk::gtk
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     gtk_DEPS_TARGET)
    endif()

    set_property(TARGET gtk::gtk
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${gtk_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET gtk::gtk
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${gtk_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET gtk::gtk
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${gtk_LIB_DIRS_RELEASE}>)
    set_property(TARGET gtk::gtk
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${gtk_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET gtk::gtk
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${gtk_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(gtk_LIBRARIES_RELEASE gtk::gtk)
