#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "kiro-client.h"
#include "kiro-trb.h"
#include <assert.h>


int
main ( int argc, char *argv[] )
{
    if (argc < 3) {
        printf ("Not enough aruments. Usage: kiro-test-latency <address> <port>\n");
        return -1;
    }

    KiroClient *client = kiro_client_new ();

    if (-1 == kiro_client_connect (client, argv[1], argv[2])) {
        kiro_client_free (client);
        return -1;
    }

    int iterations = 10000;

while (1) {
    int i = 0;
    float ping_us = 0;
    int fail_count = 0;
    while(i < iterations) {
        float tmp = kiro_client_ping_server (client);
        if (tmp < 0)
            fail_count++;
        else
            ping_us += tmp;
        i++;
    }

    printf ("Average Latency: %fus\n", ping_us/(float)(iterations - fail_count));
}
    kiro_client_free (client);
    return 0;
}

