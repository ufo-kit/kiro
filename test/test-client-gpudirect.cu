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
#include <cuda.h>
#include <cuda_runtime.h>


/**
 * cuda_example
 *
 * Does simple math directly on gpu memory.
 *
 **/
__global__
void cuda_example (void *memory_pointer)
{
    *(int*) memory_pointer *= 2;
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
        printf ("Not enough arguments. Usage: kiro-test-gpudirect <address> <port>\n");
        return -1;
    }

    KiroClient *client = kiro_client_new ();

    if (-1 == kiro_client_connect (client, argv[1], argv[2])) {
        kiro_client_free (client);
        return -1;
    }

    cudaSetDevice (0);
    kiro_client_sync (client);

    cuda_example <<<1, 1>>> (kiro_client_get_memory (client));

    void *mem = malloc (kiro_client_get_memory_size (client));
    int error = cudaMemcpy (mem, kiro_client_get_memory (client), kiro_client_get_memory_size (client), cudaMemcpyDeviceToHost);

    if (error != 0) {
        printf ("Cuda error: %d\n", error);
    }

    printf ("The transported integer times two is %d.\n", *(int*) mem);

    kiro_client_free (client);

}
