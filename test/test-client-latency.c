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
        printf ("Not enough aruments. Usage: ./client <address> <port>\n");
        return -1;
    }

    KiroClient *client = kiro_client_new ();
    KiroTrb *trb = kiro_trb_new ();

    if (-1 == kiro_client_connect (client, argv[1], argv[2])) {
        kiro_client_free (client);
        return -1;
    }

    kiro_client_sync (client);
    kiro_trb_adopt (trb, kiro_client_get_memory (client));

    GTimer *timer = g_timer_new ();
while (1) { 
    g_timer_reset (timer);
    int i = 0;
    while(i < 50000) {
        kiro_client_sync (client);
        i++;
    }

    double elapsed = g_timer_elapsed (timer, NULL);
    size_t size = kiro_client_get_memory_size (client);
    printf ("Average Latency: %fus\n", (elapsed/50000.)*1000*1000);
}
    g_timer_stop (timer);
    kiro_client_free (client);
    kiro_trb_free (trb);
    return 0;
}








