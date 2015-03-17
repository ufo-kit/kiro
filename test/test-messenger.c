#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <kiro-messenger.h>
#include <assert.h>

gboolean
grab_message (struct KiroMessage *msg, gpointer user_data)
{
    (void)user_data;
    g_message ("Message received! Content: %s", (gchar *)(msg->payload));
    msg->message_handled = TRUE;
    return TRUE;
}


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
        struct KiroMessage msg;
        msg.payload = g_malloc0 (20);
        strcpy (msg.payload, "Hello World!");
        msg.size = 13;
        if (0 > kiro_messenger_send_message (messenger, &msg))
            printf ("Sending failed...");
        else
            printf ("Message sent successfully\n");
        sleep (3);
    }
    else {
        kiro_messenger_add_receive_callback (messenger, (KiroMessengerCallbackFunc*)(grab_message), NULL);
        while (1) {sleep (1);}
    }

    kiro_messenger_free (messenger);
    return 0;
}








