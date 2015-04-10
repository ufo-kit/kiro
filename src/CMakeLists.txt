cmake_minimum_required(VERSION 2.6)
project(KIRO C)

set(kiro_SRCS
    kiro-server.c
    kiro-client.c
    kiro-trb.c
    kiro-sb.c
    )

set(kiro_HDRS
    kiro-server.h
    kiro-client.h
    kiro-trb.h
    kiro-sb.h
    )

create_enums(kiro-enums
             ${CMAKE_CURRENT_SOURCE_DIR}/kiro-enums
             ${kiro_HDRS}
             kiro-rdma.h
             )

add_library(kiro SHARED
    ${kiro_SRCS}
    ${CMAKE_CURRENT_BINARY_DIR}/kiro-enums.c)

set_target_properties(kiro PROPERTIES
    VERSION "${LIBKIRO_VERSION_MAJOR}.${LIBKIRO_VERSION_MINOR}"
    SOVERSION ${LIBKIRO_VERSION_PATCH}
)
target_link_libraries(kiro m ${KIRO_DEPS})


pkg_check_modules(GOBJECT_INTROSPECTION gobject-introspection-1.0)

if (GOBJECT_INTROSPECTION_FOUND)

    option(WITH_GIR "Build introspection files" ON)

    if (WITH_GIR)
        find_program(INTROSPECTION_SCANNER "g-ir-scanner")
        find_program(INTROSPECTION_COMPILER "g-ir-compiler")
        pkg_check_variable(gobject-introspection-1.0 g_ir_scanner)
        pkg_check_variable(gobject-introspection-1.0 g_ir_compiler)

        set(GIR_PREFIX "Kiro-${LIBKIRO_ABI_VERSION}.0")
        set(GIR_XML "${GIR_PREFIX}.gir")
        set(GIR_TYPELIB "${GIR_PREFIX}.typelib")
        set(_gir_input)

        foreach(_src ${kiro_SRCS} ${kiro_HDRS})
            list(APPEND _gir_input "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
        endforeach()

        #list(APPEND _gir_input "${CMAKE_CURRENT_BINARY_DIR}/kiro-enums.h")
        #list(APPEND _gir_input "${CMAKE_CURRENT_BINARY_DIR}/kiro-enums.c")

        add_custom_command(OUTPUT ${GIR_XML}
            COMMAND ${INTROSPECTION_SCANNER}
                    --namespace=Kiro
                    --nsversion=${LIBKIRO_ABI_VERSION}
                    --library=kiro
                    --no-libtool
                    --include=GObject-2.0
                    --include=GModule-2.0
                    --output ${GIR_XML}
                    --warn-all
                    --quiet
                    ${_gir_input}
            DEPENDS ${kiro_SRCS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        add_custom_command(OUTPUT ${GIR_TYPELIB}
            COMMAND ${INTROSPECTION_COMPILER}
                    -o ${GIR_TYPELIB}
                    ${GIR_XML}
            DEPENDS ${GIR_XML}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        add_custom_target(gir ALL DEPENDS ${GIR_XML} ${GIR_TYPELIB})
        add_dependencies(gir kiro)

    endif()
endif()


install(FILES ${kiro_HDRS} kiro-rdma.h DESTINATION ${KIRO_INCLUDEDIR}/kiro)
install(TARGETS kiro LIBRARY DESTINATION ${KIRO_LIBDIR})

if(WITH_GIR)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${GIR_XML}
            DESTINATION ${KIRO_GIRDIR}
            COMPONENT libraries)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${GIR_TYPELIB}
            DESTINATION ${KIRO_TYPELIBDIR}
            COMPONENT libraries)
endif()
