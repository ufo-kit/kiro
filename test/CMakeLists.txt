include_directories(${KIRO_SOURCE_DIR}/src)
link_directories(${KIRO_BINARY_DIR}/src)

find_package(PkgConfig)
pkg_check_modules(SDL sdl>=1.2.15)

if (SDL_FOUND)
   add_executable(client-sdl test-client.c)
   target_link_libraries(client-sdl kiro SDL m gobject-2.0 glib-2.0)
else ()
   message("SDL not found: Won't build KIRO test-client-sdl binary.")
endif ()

add_executable(kiro-test-latency test-client-latency.c)
target_link_libraries(kiro-test-latency kiro gobject-2.0 glib-2.0)
add_executable(kiro-test-bandwith test-client-bandwith.c)
target_link_libraries(kiro-test-bandwith kiro gobject-2.0 glib-2.0)
add_executable(server test-server.c)
target_link_libraries(server kiro gobject-2.0 glib-2.0)
