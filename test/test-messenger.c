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
    GOptionContext *context;
    GError *error = NULL;

    static gboolean server = FALSE;

    static GOptionEntry entries[] = {
        { "server", 's', 0, G_OPTION_ARG_NONE, &server, "Start as server (listener)", NULL },
        { NULL }
    };

#if !(GLIB_CHECK_VERSION (2, 36, 0))
    g_type_init ();
#endif

    context = g_option_context_new ("[-s | <ADDRESS>] <MESSAGE>");
    g_option_context_set_summary (context, "Send the supplied message to a peer KiroMessenger");
    g_option_context_add_main_entries (context, entries, NULL);

    if (!g_option_context_parse (context, &argc, &argv, &error)) {
        g_print ("Error parsing options: %s\n", error->message);
        return -1;
    }

    if (argc < 2 && !server) {
        g_print ("%s", g_option_context_get_help (context, TRUE, NULL));
        return 0;
    }

    KiroMessenger *messenger = kiro_messenger_new ();

    enum KiroMessengerType type = (server) ? KIRO_MESSENGER_SERVER : KIRO_MESSENGER_CLIENT;

    if (-1 == kiro_messenger_start (messenger, argv[1], "60010", type)) {
        kiro_messenger_free (messenger);
        return -1;
    }

    if (type == KIRO_MESSENGER_CLIENT) {
        struct KiroMessage msg;
        GString *str = g_string_new (argv[2]);
        msg.payload = str->str;
        msg.size = str->len + 1; // respect the NULL byte
        if (0 > kiro_messenger_send_message (messenger, &msg))
            printf ("Sending failed...");
        else
            printf ("Message submitted successfully\n");
        sleep (3);
    }
    else {
        kiro_messenger_add_receive_callback (messenger, (KiroMessengerCallbackFunc*)(grab_message), NULL);
        while (1) {sleep (1);}
    }

    kiro_messenger_free (messenger);
    return 0;
}








