include_directories(${KIRO_SOURCE_DIR}/src)
link_directories(${KIRO_BINARY_DIR}/src)

find_package(PkgConfig)
pkg_check_modules(SDL sdl>=1.2.15)

if (SDL_FOUND)
   add_executable(kiro-client-sdl test-client.c)
   target_link_libraries(kiro-client-sdl kiro SDL m gobject-2.0 glib-2.0)
   install(TARGETS kiro-client-sdl RUNTIME DESTINATION bin)
else ()
   message("SDL not found: Won't build KIRO test-client-sdl binary.")
endif ()

add_executable(kiro-test-latency test-client-latency.c)
target_link_libraries(kiro-test-latency kiro gobject-2.0 glib-2.0)
add_executable(kiro-test-bandwidth test-client-bandwidth.c)
target_link_libraries(kiro-test-bandwidth kiro gobject-2.0 glib-2.0)
add_executable(kiro-server test-server.c)
target_link_libraries(kiro-server kiro gobject-2.0 glib-2.0)

install(TARGETS kiro-test-bandwidth kiro-test-latency kiro-server
    RUNTIME DESTINATION bin)
