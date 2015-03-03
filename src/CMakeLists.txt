# If cuda is installed and gpudirect flag is set for cmake then define 
# GPUDIRECT and add cuda as library for compilation.
if(GPUDIRECT)
    find_package(CUDA)
    message(STATUS "GPUDIRECT: Enabled")
    add_definitions(-DGPUDIRECT)
    cuda_add_library(kiro SHARED kiro-trb.c kiro-client.c kiro-server.c kiro-sb.c)
else()
    add_library(kiro SHARED kiro-trb.c kiro-client.c kiro-server.c kiro-sb.c)
endif()

set_target_properties(kiro PROPERTIES
    VERSION "${LIBKIRO_VERSION_MAJOR}.${LIBKIRO_VERSION_MINOR}"
    SOVERSION ${LIBKIRO_VERSION_PATCH}
)
target_link_libraries(kiro m rdmacm ibverbs pthread)

install(FILES kiro-rdma.h kiro-trb.h kiro-client.h kiro-server.h kiro-sb.h
    DESTINATION ${KIRO_INCLUDEDIR}/kiro)

install(TARGETS kiro
    LIBRARY DESTINATION ${KIRO_LIBDIR}
)
