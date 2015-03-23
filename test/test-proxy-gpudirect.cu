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
 * @title: GPUDIRECTproxygpudirect
 * @filename: test-proxy-gpudirect.cu
 *
 * GPUDIRECTproxygpudirect receives data from infiniband, runs a cuda kernel
 * on the data and provides the data via server. Receiving and
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
#include <glib.h>
#include "kernels/identity.cu"
#include "kernels/twice.cu"


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
    // Benchmark variables
    GTimer *timer = g_timer_new ();
    float t_host_infiniband = 0;
    float t_host_algorithm = 0;
    const int iterations = 1000;
    int iterate = iterations;

    // Proxy variables
    cudaError_t error;
    unsigned long int current_frame = 0;
    unsigned long int remote_frame = 0;

    if (argc < 3) {
        g_message ("Not enough arguments. Usage: kiro-proxy-gpudirect <address> <port>\n");
        return -1;
    }

    // Switch off GPU memory allocation and gpudirect data path.
    gpudirect = 1;
    // Select first graphics card.
    cudaSetDevice (0);

    // Setup connection.
    KiroClient *kiroClient = kiro_client_new ();
    if (-1 == kiro_client_connect (kiroClient, argv[1], argv[2])) {
        kiro_client_free (kiroClient);
        return -1;
    }

    // Malloc some memory for the kernel result.
    void *result_gpu;
    kiro_client_sync (kiroClient);
    size_t result_size = kiro_client_get_memory_size (kiroClient);
    error = cudaMalloc (&result_gpu, result_size);
    if (error != 0) {
        g_message ("cudaMalloc: %s \n", cudaGetErrorString(error));
        return -1;
    }

    // Start the server with that memory.
    KiroServer *kiroServer = kiro_server_new (); 
    if (0 > kiro_server_start (kiroServer, NULL, "60011", result_gpu, result_size)) {
        g_critical ("Failed to start server properly");
        return -1;
    }

    while (1) {
        // Receive current_frame.
        kiro_client_sync_partial (kiroClient, 0, sizeof (remote_frame), 0);
        cudaMemcpy (&remote_frame, kiro_client_get_memory (kiroClient), sizeof (remote_frame), cudaMemcpyDeviceToHost);
        if (error != 0) {
            g_message ("cudaMalloc: %s \n", cudaGetErrorString(error));
            return -1;
        }
        // Check if new data (e.g. new Image) is ready.
        if (remote_frame > current_frame) {
            // Update current_frame counter.
            current_frame = remote_frame;

            // Receive data.
            g_timer_reset (timer);
            kiro_client_sync (kiroClient);
            t_host_infiniband += g_timer_elapsed (timer, NULL);
            // Run kernel on data.
            g_timer_reset (timer);
            twice <<<(kiro_client_get_memory_size (kiroClient) - sizeof (unsigned long int)) / 1024, 1024>>> (kiro_client_get_memory (kiroClient), kiro_client_get_memory_size (kiroClient), \
                result_gpu, result_size);
            cudaDeviceSynchronize ();
            t_host_algorithm += g_timer_elapsed (timer, NULL);
            // Status update over the last iterations.
            iterate -= 1;
            if (iterate == 0) {
                // Print times.
                float size_gb = ((float) kiro_client_get_memory_size (kiroClient) / (1024.0 * 1024.0 * 1024.0)) * iterations;
                g_message ("t_host_infiniband: %.2f ms", (t_host_infiniband / iterations) * 1000);
                g_message ("t_host_algorithm: %.2f ms", (t_host_algorithm / iterations) * 1000);

                // Print throughput.
                g_message ("Throughput Infiniband: %.2f Gbyte/s", size_gb / t_host_infiniband);
                g_message ("Throughput Algorithm: %.2f Gbyte/s", size_gb / t_host_algorithm);

                // Reset all counters.
                iterate = iterations;
                t_host_infiniband = 0;
                t_host_algorithm = 0;
            }
        }
    }

}
