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
#include "kiro-server.h"

#include "kernels/identity.cu"


/**
 * main
 *
 * This is the main function which connects to the GPUDIRECT server and reads
 * its data via GPUdirect to the GPUs memory. Then it runs a cuda kernel on
 * the gpu to access that memory. Eventually it provides a server to proxy the 
 * data.
 *
 * Run this from shell with kiro-proxy-gpudirect.
 *
 **/
    int 
main ( int argc, char *argv[])
{
    cudaError_t error;
    unsigned long int current_frame = 0;
    unsigned long int remote_frame = 0;

    if (argc < 3) {
        g_message ("Not enough arguments. Usage: kiro-proxy-gpudirect <address> <port>\n");
        return -1;
    }

    // Switch on GPU memory allocation and gpudirect data path.
    gpudirect = 1;
    // Select first graphics card.
    cudaSetDevice (0);

    // Setup connection.
    KiroClient *kiroClient = kiro_client_new ();
    if (-1 == kiro_client_connect (kiroClient, argv[1], argv[2])) {
        kiro_client_free (kiroClient);
        return -1;
    }

    // Malloc some cuda memory for the kernel result.
    void *result;
    size_t result_size = kiro_client_get_memory_size (kiroClient);
    error = cudaMalloc (&result, result_size);
    if (error != 0) {
        g_message ("cudaMalloc: %s \n", cudaGetErrorString(error));
        return -1;
    }

    // Start the server with that memory.
    KiroServer *kiroServer = kiro_server_new (); 
    if (0 > kiro_server_start (kiroServer, NULL, "60011", result, result_size)) {
        g_critical ("Failed to start server properly");
        goto done;
    }   

    // Now endlessly receive data, run the kernel on it and serve it.
    while (1) {
        // Receive current_frame.
        kiro_client_sync_partial (kiroClient, 0, sizeof (remote_frame), 0);
        error = cudaMemcpy (&remote_frame, kiro_client_get_memory (kiroClient), sizeof (remote_frame), cudaMemcpyDeviceToHost);
        if (error != 0) {
            g_message ("cudaMemcpy: %s \n", cudaGetErrorString(error));
            return -1;
        }
        cudaDeviceSynchronize ();
        // Check if new data (e.g. new Image) is ready.
        if (remote_frame > current_frame) {
            // Tell user if frames have been skipped.
            if (remote_frame - current_frame - 1) {
                g_warning ("Frames have been skipped! Now at frame: %ld, skipped %ld previous frame(s).", \
                 remote_frame, remote_frame - current_frame - 1);
            }
            // Update current_frame counter.
            current_frame = remote_frame;
            g_warning ("Current Frame: %ld", current_frame);
            // Receive data.
            // TODO: Only sync data when in triple buffering
            kiro_client_sync (kiroClient);
            // Run kernel on data.
            identity <<<1, 1>>> (kiro_client_get_memory (kiroClient), kiro_client_get_memory_size (kiroClient), result, result_size);
            // Wait for kernel to finish.
            cudaDeviceSynchronize ();

            // Sleep random amount of time.
            sleep (rand() % 10 / 2);
        }
    }
done:
    kiro_server_free (kiroServer);
    return 0;
}
