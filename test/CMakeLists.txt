include_directories(${KIRO_SOURCE_DIR}/src)
link_directories(${KIRO_BINARY_DIR}/src)

find_package(PkgConfig)
pkg_check_modules(SDL sdl>=1.2.15)

if (SDL_FOUND)
   add_executable(kiro-client-sdl test-client.c)
   target_link_libraries(kiro-client-sdl kiro SDL m ${KIRO_DEPS})
   install(TARGETS kiro-client-sdl RUNTIME DESTINATION ${KIRO_BINDIR})
else ()
   message("SDL not found: Won't build kiro-client-sdl binary.")
endif ()

add_executable(kiro-server test-server.c)
target_link_libraries(kiro-server kiro m ${KIRO_DEPS})

add_executable(kiro-test-latency test-client-latency.c)
target_link_libraries(kiro-test-latency kiro ${KIRO_DEPS})

add_executable(kiro-test-bandwidth test-client-bandwidth.c)
target_link_libraries(kiro-test-bandwidth kiro ${KIRO_DEPS})

add_executable(kiro-test-partial test-partial.c)
target_link_libraries(kiro-test-partial kiro ${KIRO_DEPS})

add_executable(kiro-test-messenger test-messenger.c)
target_link_libraries(kiro-test-messenger kiro ${KIRO_DEPS})

install(TARGETS kiro-test-bandwidth kiro-test-latency kiro-test-messenger
    kiro-server RUNTIME DESTINATION ${KIRO_BINDIR})
