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
 * TODO: fixme
 * SECTION: test-proxy-gpudirect
 * @short_description: KIRO GPUDIRECT test proxy
 * @title: GPUDIRECTproxy
 * @filename: test-proxy-gpudirect.c
 *
 * GPUDIRECTclient receives data from infiniband, runs one or multiple cuda 
 * kernels on the data and provides the data via server. Receiving and
 * serving data both work via GPUDirect.
 * 
 **/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <glib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "kiro-client.h"

/**
 * main
 *
 *
 **/
int 
main ( int argc, char *argv[])
{
    unsigned long int frame;
    void *data;
    if (argc < 3) {
        g_message ("Not enough arguments. Usage: kiro-proxy-gpudirect <address> <port>\n");
        return -1;
    }

    // Setup connection.
    KiroClient *kiroClient = kiro_client_new ();
    if (-1 == kiro_client_connect (kiroClient, argv[1], argv[2])) {
        kiro_client_free (kiroClient);
        return -1;
    }
    
    unsigned long int current_frame = 0;
    while (1) {
        // Receive data.
        kiro_client_sync (kiroClient);
        // Get frame
        frame = *(unsigned long int *)kiro_client_get_memory (kiroClient);
        // Get data
        data = kiro_client_get_memory (kiroClient) + sizeof (frame);

        if (current_frame < frame) {
            g_warning ("Frame: %ld, Data %d", frame, *(int *)data);
            current_frame = frame;
        }
    }
}
