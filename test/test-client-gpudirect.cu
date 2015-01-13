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
 * SECTION: test-client-gpudirect
 * @short_description: KIRO GPUDIRECT test client
 * @title: GPUDIRECTclient
 * @filename: test-client-gpudirect.c
 *
 * GPUDIRECTclient implements a client that reads data from the GPUDIRECTserver.
 * 
 **/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kiro-client.h"
#include <assert.h>
#include <glib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

/**
 * cuda_example
 *
 * Does simple math directly on gpu memory.
 *
 **/
__global__
void cuda_example (void *memory_pointer)
{
    int *mem_pointer = (int*) memory_pointer;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    mem_pointer[index/4] *= 2;

    /*
    for (int i = 0; i < 10; i++) {
        *mem_pointer[thread] *= 2;
        mem_pointer += 1;
    }
    */
}


/**
 * main
 *
 * This is the main function which connects to the GPUDIRECT server and reads
 * its data via GPUdirect to the GPUs memory. Then it runs a cuda kernel on
 * the gpu to access that memory (cuda_example). Evenutally it copies the
 * data from device to host, to print it.
 *
 * Run this from shell with kiro-test-gpudirect.
 *
 **/
int 
main ( int argc, char *argv[])
{
    if (argc < 3) {
        g_message ("Not enough arguments. Usage: kiro-test-gpudirect <address> <port>\n");
        return -1;
    }

    GTimer *timer = g_timer_new ();

    const int iterations = 10000;

    float t_host_infiniband = 0;
    float t_host_hosttodevice = 0;
    float t_host_algorithm = 0;
    float t_host_devicetohost = 0;

    float t_gpu_infiniband = 0;
    float t_gpu_algorithm = 0;
    float t_gpu_devicetohost = 0;

    /*
    *   GPU MEMORY
    */ 

    // Switch on GPU memory allocation and gpudirect data path.
    gpudirect = 1;

    // Create new kiro client.
    KiroClient *client_gpu = kiro_client_new ();

    // Connect to server and setup memory.
    if (-1 == kiro_client_connect (client_gpu, argv[1], argv[2])) {
        kiro_client_free (client_gpu);
        return -1;
    }

    // Select first graphics card.
    cudaSetDevice (0);
    

    // Transfer data "iterations" times to get average throughput.
    void *host_mem = malloc (kiro_client_get_memory_size (client_gpu));
    for (int i = 0; i < iterations; i++) {
        // Receive data from server into gpu memory via gpudirect.
        g_timer_reset (timer);
        kiro_client_sync (client_gpu);
        cudaDeviceSynchronize();
        t_gpu_infiniband += g_timer_elapsed (timer, NULL);
    
        // Do some simple math on transferred data.
        cudaDeviceSynchronize();
        g_timer_reset (timer);
        cuda_example <<<kiro_client_get_memory_size (client_gpu) / 1024, 1024>>> (kiro_client_get_memory (client_gpu));
        cudaDeviceSynchronize();
        t_gpu_algorithm += g_timer_elapsed (timer, NULL);

        // Copy received data into main memory, to inspect it.
        g_timer_reset (timer);
        cudaError_t error = cudaMemcpy (host_mem, kiro_client_get_memory (client_gpu), \
            kiro_client_get_memory_size (client_gpu), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        t_gpu_devicetohost += g_timer_elapsed (timer, NULL);

        // Check if copy was successfull.
        if (error != 0) {
            g_message ("Cuda error: %s \n", cudaGetErrorString(error));
            return -1;
        }
    }


    // Print the transported integers for inspection.
    for (int i = 0; i < 10; i++) {
        g_message ("%d", *(((int*) host_mem) + i));
    }

    // Release used memory.
    kiro_client_free (client_gpu);
    free (host_mem);


    /*
    *   HOST MEMORY
    */

    // Switch on GPU memory allocation and gpudirect data path.
    gpudirect = 0;

    // Create new kiro client.
    KiroClient *client_host = kiro_client_new ();

    // Connect to server and setup memory.
    if (-1 == kiro_client_connect (client_host, argv[1], argv[2])) {
        kiro_client_free (client_host);
        return -1;
    }
    
    // Transfer data 1000 times to get average througput.
    void *gpu_mem;
    void *host_mem_2 = malloc (kiro_client_get_memory_size (client_host));
    for (int i = 0; i < iterations; i++) {
        // Receive data from server into host memory.
        g_timer_reset (timer);
        kiro_client_sync (client_host);
        cudaDeviceSynchronize();
        t_host_infiniband += g_timer_elapsed (timer, NULL);
        
        // Copy memory from host to gpu memory.
        cudaError_t error = cudaMalloc (&gpu_mem, kiro_client_get_memory_size (client_host));
        g_timer_reset (timer);
        error = cudaMemcpy (gpu_mem, kiro_client_get_memory (client_host), \
            kiro_client_get_memory_size (client_host), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        t_host_hosttodevice += g_timer_elapsed (timer, NULL);

        // Check if copy was successfull.
        if (error != 0) {
            g_message ("Cuda error: %s \n", cudaGetErrorString(error));
            return -1;
        }
        // Do some simple math on transferred data.
        g_timer_reset (timer);
        cuda_example <<<kiro_client_get_memory_size (client_host) / 1024, 1024>>> (gpu_mem);
        cudaDeviceSynchronize();
        t_host_algorithm += g_timer_elapsed (timer, NULL);

        // Copy data back from gpu memory to host.
        g_timer_reset (timer);
        error = cudaMemcpy (host_mem_2, gpu_mem, \
            kiro_client_get_memory_size (client_host), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        t_host_devicetohost += g_timer_elapsed (timer, NULL);

        // Check if copy was successfull.
        if (error != 0) {
            g_message ("Cuda error: %s\n", cudaGetErrorString(error));
            return -1;
        }
    }

    // Print the transported integers for inspection.
    for (int i = 0; i < 10; i++) {
        g_message ("%d", *(((int*) host_mem_2) + i));
    }

    // Inspect Data.
    g_message ("t_host_infiniband \t %.2f ms\n", (t_host_infiniband / iterations) * 1000);
    g_message ("t_gpu_infiniband \t %.2f ms\n", (t_gpu_infiniband / iterations) * 1000);
    g_message ("+t_host_hosttodevice \t %.2f ms\n", (t_host_hosttodevice / iterations) * 1000);
    g_message ("t_host_algorithm \t %.2f ms\n", (t_host_algorithm / iterations) * 1000);
    g_message ("t_gpu_algorithm \t %.2f ms\n", (t_gpu_algorithm / iterations) * 1000);
    g_message ("t_host_devicetohost \t %.2f ms\n", (t_host_devicetohost / iterations) * 1000);
    g_message ("t_gpu_devicetohost \t %.2f ms\n", (t_gpu_devicetohost / iterations) * 1000);

    float size_gb = ((float) kiro_client_get_memory_size (client_host) / (1024.0 * 1024.0 * 1024.0)) * iterations;    

    g_message ("[HOST]\t Throughput Infiniband \t\t %.2f Gbyte/s\n", size_gb / t_host_infiniband);
    g_message ("[HOST]\t Throughput Host to Device \t %.2f Gbyte/s\n", size_gb / t_host_hosttodevice);
    g_message ("[HOST]\t Throughput Device to Host \t %.2f Gbyte/s\n", size_gb / t_host_devicetohost);

    g_message ("[GPU]\t Throughput Infiniband \t\t %.2f Gbyte/s\n", size_gb / t_gpu_infiniband);
    g_message ("[GPU]\t Throughput Device to Host \t %.2f Gbyte/s\n", size_gb / t_gpu_devicetohost);

    // Release used memory.
    kiro_client_free (client_host);
    cudaFree (gpu_mem);
    free (host_mem_2);
}
