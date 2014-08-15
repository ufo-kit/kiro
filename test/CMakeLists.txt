include_directories(${KIRO_SOURCE_DIR}/src)
link_directories(${KIRO_BINARY_DIR}/src)

find_package(PkgConfig)
pkg_check_modules(SDL sdl>=1.2.15)

if (SDL_FOUND)
   add_executable(client test-client.c)
   target_link_libraries(client kiro SDL m gobject-2.0 glib-2.0)
else ()
   message("SDL not found: Won't build KIRO test-client binary.")
endif ()

add_executable(server test-server.c)
target_link_libraries(server kiro gobject-2.0 glib-2.0)
