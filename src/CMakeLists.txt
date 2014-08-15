add_library(kiro SHARED kiro-trb.c kiro-client.c kiro-server.c)
set_target_properties(kiro PROPERTIES
    VERSION "${LIBKIRO_VERSION_MAJOR}.${LIBKIRO_VERSION_MINOR}"
    SOVERSION ${LIBKIRO_VERSION_PATCH}
)
target_link_libraries(kiro m rdmacm ibverbs pthread)

install(FILES kiro-rdma.h kiro-trb.h kiro-client.h kiro-server.h DESTINATION ${HEADER_INSTALL_DIR})

install(TARGETS kiro
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
)
