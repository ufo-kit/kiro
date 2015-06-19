#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <kiro-messenger.h>
#include <assert.h>
#include <unistd.h>

gboolean
grab_message (KiroMessageStatus *status, gpointer user_data)
{
    gulong *rank = (gulong *)user_data;
    g_message ("Message received! Type: %u, Content: %s", status->message->msg, (gchar *)(status->message->payload));
    *rank = status->message->peer_rank;
    return KIRO_CALLBACK_CONTINUE;
}


gboolean
message_was_sent (KiroMessageStatus *status, gpointer user_data)
{
    gboolean *flag = (gboolean *)user_data;
    if (status->status == KIRO_MESSAGE_SEND_SUCCESS)
        g_message ("Message was sent successfully");
    else
        g_message ("Message sending failed");
    *flag = TRUE;
    return KIRO_CALLBACK_REMOVE;
}

KiroContinueFlag
connect_callback (gulong new_rank, gpointer user_data)
{
    (void) user_data;
    printf ("New peer with rank '%lu' connected\n", new_rank);
    return KIRO_CALLBACK_CONTINUE;
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

    gulong rank = 0;
    if (server)
        kiro_messenger_start_listen (messenger, argv[1], "60010", connect_callback, NULL, &error);
    else
        kiro_messenger_connect (messenger, argv[1], "60010", &rank, &error);

    if (error) {
        printf ("Failed to start Kiro Messenger!\nError: '%s'\n", error->message);
        kiro_messenger_free (messenger);
        return -1;
    }

    if (!server) {
        KiroMessage msg;
        GString *str = g_string_new (argv[2]);
        msg.msg = 42;
        msg.size = str->len + 1; // respect the NULL byte */
        msg.payload = str->str;
        msg.peer_rank = 1;

        gboolean can_leave = FALSE;

        if (!kiro_messenger_send_with_callback (messenger, &msg, message_was_sent, &can_leave, &error)) {
            printf ("Sending failed: '%s'\n", error->message);
            goto done;
        }
        else
            printf ("Message submitted successfully\n");

        while (can_leave == FALSE) {}
        can_leave = FALSE;
        kiro_messenger_add_receive_callback (messenger, grab_message, &can_leave);
        while (!can_leave) {}
    }
    else {
        gulong sender_rank = 0;
        kiro_messenger_add_receive_callback (messenger, grab_message, &sender_rank);
        g_message ("Messenger started. Waiting for incoming messages.");
        while (1) {
            while (sender_rank == 0) {};
            printf ("Sending Echo...\n");
            KiroMessage msg;
            msg.msg = 1337;
            msg.payload = g_strdup ("Echo");
            msg.size = 5; // respect the NULL byte
            msg.peer_rank = sender_rank;
            kiro_messenger_send_blocking (messenger, &msg, &error);
            sender_rank = 0;
        }
    }

done:
    kiro_messenger_free (messenger);
    return 0;
}







