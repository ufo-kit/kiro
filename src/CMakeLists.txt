add_library(kiro kiro-trb.c kiro-client.c kiro-server.c)
target_link_libraries(kiro SDL m rdmacm ibverbs pthread)
