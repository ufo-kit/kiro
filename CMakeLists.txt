project(KIRO)

cmake_minimum_required(VERSION 2.6)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(TARNAME "kiro")
set(LIBKIRO_VERSION_MAJOR "0")
set(LIBKIRO_VERSION_MINOR "3")
set(LIBKIRO_VERSION_PATCH "1")
set(LIBKIRO_VERSION_RELEASE "5")
set(LIBKIRO_VERSION_STRING "${LIBKIRO_VERSION_MAJOR}.${LIBKIRO_VERSION_MINOR}.${LIBKIRO_VERSION_PATCH}")
set(VERSION "${LIBKIRO_VERSION_STRING}")
set(LIBKIRO_DESCRIPTION "Small InfiniBand communication Server and Client")

set(LIBKIRO_ABI_VERSION "${LIBKIRO_VERSION_MAJOR}.${LIBKIRO_VERSION_MINOR}")
set(LIBKIRO_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/common/cmake")

message(STATUS "checking for rdmacm-devel library")
find_path(RDMACM_DIR rdma/rdma_verbs.h)
if(NOT RDMACM_DIR)
  message( FATAL_ERROR "rdma/rdma_verbs.h Not Found! Probably your version of rdmacm-devel is too old. (Needs to be 1.0.15 or newer)" )
else()
  message(STATUS "  found rdma/rdma_verbs.h")
endif()

find_package(PkgConfig)
pkg_check_modules(GLIB2 glib-2.0>=2.32 REQUIRED)
pkg_check_modules(GOBJECT2 gobject-2.0>=2.32 REQUIRED)

include_directories(
  SYSTEM
  ${GLIB2_INCLUDE_DIRS}
  ${GOBJECT2_INCLUDE_DIRS}
)

add_definitions(-Wall -Wextra -std=c99)

include(ConfigurePaths)
configure_paths(KIRO)


add_subdirectory(src)
add_subdirectory(test)

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

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/kiro.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/kiro.pc" @ONLY IMMEDIATE)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/kiro.pc DESTINATION ${KIRO_PKGCONFIGDIR})

include(CPack)
