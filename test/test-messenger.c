#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <kiro-messenger.h>
#include <assert.h>
#include <unistd.h>

gboolean
grab_message (struct KiroMessage *msg, gpointer user_data)
{
    gboolean *flag = (gboolean *)user_data;
    g_message ("Message received! Type: %u, Content: %s", msg->msg, (gchar *)(msg->payload));
    msg->message_handled = TRUE;
    *flag = TRUE;
    return TRUE;
}


gboolean
message_was_sent (struct KiroMessage *msg, gpointer user_data)
{
    gboolean *flag = (gboolean *)user_data;
    if (msg->status == KIRO_MESSAGE_SEND_SUCCESS)
        g_message ("Message was sent successfully");
    else
        g_message ("Message sending failed");
    *flag = TRUE;
    return FALSE;
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
        msg.msg = 42;
        msg.payload = str->str;
        msg.size = str->len + 1; // respect the NULL byte

        gboolean can_leave = FALSE;
        kiro_messenger_add_send_callback (messenger, message_was_sent, &can_leave);

        if (0 > kiro_messenger_submit_message (messenger, &msg, TRUE))
            printf ("Sending failed...");
        else
            printf ("Message submitted successfully\n");
        while (!can_leave) {}
        can_leave = FALSE;
        kiro_messenger_add_receive_callback (messenger, grab_message, &can_leave);
        while (!can_leave) {}
    }
    else {
        gboolean answer = FALSE;
        kiro_messenger_add_receive_callback (messenger, grab_message, &answer);
        g_message ("Messenger started. Waiting for incoming messages.");
        while (1) {
            while (!answer) {};
            struct KiroMessage *msg = g_malloc0 (sizeof (struct KiroMessage));
            msg->msg = 1337;
            msg->payload = g_strdup ("Echo");
            msg->size = 5; // respect the NULL byte
            kiro_messenger_submit_message (messenger, msg, TRUE);
            answer = FALSE;
        }
    }

    kiro_messenger_free (messenger);
    return 0;
}








