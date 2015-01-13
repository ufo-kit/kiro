/* Copyright (C) 2014 Max Riechelmann <max.riechelmann@googlemail.com>
   (Karlsruhe Institute of Technology)

   This library is free software; you can redistribute it and/or modify it
   under the terms of the GNU Lesser General Public License as published by the
   Free Software Foundation; either version 2.1 of the License, or (at your
   option) any later version.

   This library is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
   details.

   You should have received a copy of the GNU Lesser General Public License along
   with this library; if not, write to the Free Software Foundation, Inc., 51
   Franklin St, Fifth Floor, Boston, MA 02110, USA
   */

/**
 * SECTION: test-server-gpudirect
 * @short_description: KIRO GPUDIRECT test server
 * @title: GPUDIRECTserver
 * @filename: test-server-gpudirect.c
 *
 * GPUDIRECTserver implements a server that holds data which can be read by the 
 * GPUDIRECTclient.
 *
 **/

#include <stdio.h>
#include <stdlib.h>
#include "kiro-server.h"
#include "kiro-trb.h"
#include <gmodule.h>
#include <gio/gio.h>
#include <cuda.h>
#include <cuda_runtime.h>


/**
 * main
 *
 * This is the main function which sets up the server for GPUDIRECT, populates 
 * the main memory with data and waits for a client to connect and read that 
 * data.
 *
 * Run this from shell with kiro-server-gpudirect.
 *
 **/
int 
main (void)
{
    KiroServer *server = kiro_server_new ();

    // Allocate 1MB of random data.
    int mem_size = 1024 * 256 * sizeof (int);
    int *mem = malloc (mem_size);
    int *ptr = mem;
    srand (time (NULL));
    for (int i = 0; i < 1024 * 256; i++) {
        *ptr = rand ();
        // Show the first 20 integers for inspection.
        if (i < 20) {
            g_info ("%d  ", *ptr);
        }
        ptr += 1;
    }

    if (0 > kiro_server_start (server, NULL, "60010", mem, mem_size )) {
        g_critical ("Failed to start server properly");
        goto done;
    }

    while (1) {
        // Endless loop to keep server alive.
    }

done:
    free (mem);
    kiro_server_free (server);
    return 0;
}
