cmake_minimum_required(VERSION 2.6)
project(KIRO C)

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(TARNAME "kiro")
set(LIBKIRO_VERSION_MAJOR "1")
set(LIBKIRO_VERSION_MINOR "3")
set(LIBKIRO_VERSION_PATCH "0")
set(LIBKIRO_VERSION_RELEASE "9")
set(LIBKIRO_VERSION_STRING "${LIBKIRO_VERSION_MAJOR}.${LIBKIRO_VERSION_MINOR}.${LIBKIRO_VERSION_PATCH}")
set(VERSION "${LIBKIRO_VERSION_STRING}")
set(LIBKIRO_DESCRIPTION "Small InfiniBand communication library")

# Increase the ABI version when binary compatibility cannot be guaranteed, e.g.
# symbols have been removed, function signatures, structures, constants etc.
# changed.
set(LIBKIRO_ABI_VERSION "1")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/common/cmake")

# create_enums
# @prefix: prefix of the output enum files
# @template_prefix: prefix of the template (should be src/kiro-enums)
# @header_list: list with header files that should be searched for enums
macro(create_enums prefix template_prefix header_list)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${prefix}.h
        COMMAND ${GLIB2_MKENUMS}
        ARGS
            --template ${template_prefix}.h.template
            ${header_list} > ${CMAKE_CURRENT_BINARY_DIR}/${prefix}.h
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${header_list})

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${prefix}.c
        COMMAND ${GLIB2_MKENUMS}
        ARGS
            --template ${template_prefix}.c.template
            ${header_list} > ${CMAKE_CURRENT_BINARY_DIR}/${prefix}.c
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${header_list}
                ${CMAKE_CURRENT_BINARY_DIR}/${prefix}.h
        )
endmacro()

include(ConfigurePaths)
include(PkgConfigVars)
configure_paths(KIRO)

find_package(PkgConfig)
find_program(GLIB2_MKENUMS glib-mkenums REQUIRED)
pkg_check_modules(GLIB2 glib-2.0>=2.32 REQUIRED)
pkg_check_modules(GOBJECT2 gobject-2.0>=2.32 REQUIRED)
pkg_check_modules(GMODULE2 gmodule-2.0>=2.32 REQUIRED)
pkg_check_modules(GIO2 gio-2.0>=2.32 REQUIRED)

message(STATUS "checking for rdmacm-devel library")
find_path(RDMACM_DIR rdma/rdma_verbs.h)
if(NOT RDMACM_DIR)
    message( FATAL_ERROR "rdma/rdma_verbs.h Not Found! Probably your version of rdmacm-devel is too old. (Needs to be 1.0.15 or newer)" )
else()
    message(STATUS "  found rdma/rdma_verbs.h")
endif()


include_directories(
    SYSTEM
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GLIB2_INCLUDE_DIRS}
    ${GOBJECT2_INCLUDE_DIRS}
    ${GMODULE2_INCLUDE_DIRS}
    ${GIO2_INCLUDE_DIRS}
    ${RDMACM_DIR})

set(KIRO_DEPS
    rdmacm
    ibverbs
    ${GLIB2_LIBRARIES}
    ${GOBJECT2_LIBRARIES}
    ${GMODULE2_LIBRARIES}
    ${GIO2_LIBRARIES})

set(LIBKIRO_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-Wall -Wextra -std=c99)

add_subdirectory(src)
add_subdirectory(test)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/kiro.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/kiro.pc" @ONLY IMMEDIATE)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/kiro.pc DESTINATION ${KIRO_PKGCONFIGDIR})

#"{{{CPAK
if(WITH_PACKAGE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/package.sh.in
                   ${CMAKE_CURRENT_BINARY_DIR}/package.sh)

    set(CPACK_PACKAGE_DESCRIPTION ${LIBKIRO_DESCRIPTION})
    set(CPACK_PACKAGE_NAME ${TARNAME})
    set(CPACK_PACKAGE_CONTACT "Timo Dritschler <timo.dritschler@kit.edu>")
    set(CPACK_PACKAGE_VENDOR "Karlsruhe Institute of Technology")
    set(CPACK_PACKAGE_VERSION_MAJOR "${LIBKIRO_VERSION_MAJOR}")
    set(CPACK_PACKAGE_VERSION_MINOR "${LIBKIRO_VERSION_MINOR}")
    set(CPACK_PACKAGE_VERSION_PATCH "${LIBKIRO_VERSION_PATCH}")
    set(CPACK_PACKAGE_RELEASE "${LIBKIRO_VERSION_RELEASE}")

    set(CPACK_GENERATOR "TGZ;RPM;")
    set(CPACK_SOURCE_GENERATOR "TGZ")
    set(CPACK_SOURCE_IGNORE_FILES "tags" ".git")
    set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
    set(CPACK_SOURCE_PACKAGE_FILE_NAME "LIBKIRO-${LIBKIRO_VERSION_STRING}" CACHE INTERNAL "tarball basename")
    set(CPACK_PACKAGE_FILE_NAME
        "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")


    include(CPack)
endif()
#}}}
