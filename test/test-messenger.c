#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <kiro-messenger.h>
#include <assert.h>


int 
main ( int argc, char *argv[] )
{
    if (argc < 3) {
        printf ("Not enough aruments. Usage: kiro-test-bandwidth <address> <port>\n");
        return -1;
    }

    KiroMessenger *messenger = kiro_messenger_new ();

    enum KiroMessengerType type = (argc > 3) ? KIRO_MESSENGER_SERVER : KIRO_MESSENGER_CLIENT; 
 
    if (-1 == kiro_messenger_start (messenger, argv[1], argv[2], type)) {
        kiro_messenger_free (messenger);
        return -1;
    }

    if (type == KIRO_MESSENGER_CLIENT) {
        sleep (5);
    }
    else {
        while (1) {sleep (1);}
    }
    
    kiro_messenger_free (messenger);
    return 0;
}








