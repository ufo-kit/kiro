include_directories(${KIRO_SOURCE_DIR}/src)
link_directories(${KIRO_BINARY_DIR}/src)

add_executable(client test-client.c)
target_link_libraries(client kiro SDL m gobject-2.0 glib-2.0)

add_executable(server test-server.c)
target_link_libraries(server kiro gobject-2.0 glib-2.0)
