include_directories(${KIRO_SOURCE_DIR}/src)
link_directories(${KIRO_BINARY_DIR}/src)

find_package(PkgConfig)
pkg_check_modules(SDL sdl>=1.2.15)

if (SDL_FOUND)
   add_executable(kiro-client-sdl test-client.c)
   target_link_libraries(kiro-client-sdl kiro SDL m gobject-2.0 glib-2.0)
   install(TARGETS kiro-client-sdl RUNTIME DESTINATION ${KIRO_BINDIR})
else ()
   message("SDL not found: Won't build kiro-client-sdl binary.")
endif ()

add_executable(kiro-test-latency test-client-latency.c)
target_link_libraries(kiro-test-latency kiro gobject-2.0 glib-2.0)
add_executable(kiro-test-bandwidth test-client-bandwidth.c)
target_link_libraries(kiro-test-bandwidth kiro gobject-2.0 glib-2.0)
add_executable(kiro-server test-server.c)
target_link_libraries(kiro-server kiro gobject-2.0 glib-2.0)
add_executable(kiro-test-partial test-partial.c)
target_link_libraries(kiro-test-partial kiro gobject-2.0 glib-2.0)

option(GPUDIRECT "GPUDIRECT: Disabled" OFF)
if (GPUDIRECT)
   find_package(CUDA)
   add_definitions(-DGPUDIRECT)
   cuda_add_executable(kiro-server-gpudirect test-server-gpudirect.c)
   target_link_libraries(kiro-server-gpudirect kiro gobject-2.0 glib-2.0)
   cuda_add_executable(kiro-test-gpudirect test-client-gpudirect.cu)
   target_link_libraries(kiro-test-gpudirect kiro gobject-2.0 glib-2.0)
   install(TARGETS kiro-test-bandwidth kiro-test-latency kiro-server
      kiro-server-gpudirect kiro-test-gpudirect
  	  RUNTIME DESTINATION ${KIRO_BINDIR})
else()
   install(TARGETS kiro-test-bandwidth kiro-test-latency kiro-server
	  RUNTIME DESTINATION ${KIRO_BINDIR})
endif()
