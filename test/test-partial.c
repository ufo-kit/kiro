#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "kiro-trb.h"
#include "kiro-sb.h"


int count = 0;


KiroContinueFlag
callback (KiroSb *sb) {
    (void)sb;
    g_message ("Got new element");
    count++;
    return KIRO_CALLBACK_CONTINUE;
}

int 
main ( int argc, char *argv[] )
{
    if (argc < 3) {
        printf ("Not enough aruments. Usage: kiro-test-bandwidth <address> <port>\n");
        return -1;
    }


    KiroSb *ksb = kiro_sb_new ();
    kiro_sb_clone (ksb, argv[1], argv[2]);
    unsigned long int callback_id = kiro_sb_add_sync_callback (ksb, (KiroSbSyncCallbackFunc)callback, ksb);

    while (count < 3) {}
    kiro_sb_remove_sync_callback (ksb, callback_id);

    while (count < 6) {
        kiro_sb_get_data_blocking (ksb);
        g_message ("Got new element");
        count++;
    }

    kiro_sb_free (ksb);

    return 0;
}



