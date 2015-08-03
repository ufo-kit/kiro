#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <kiro-messenger.h>
#include <assert.h>
#include <unistd.h>

void
callback (KiroRequest *request, gpointer user_data)
{
    gboolean *flag = (gboolean *)user_data;
    g_free (request->message->payload);
    g_free (request->message);
    *flag = TRUE;
}

KiroContinueFlag
conn_callback (gulong new_rank, gpointer user_data) {
    (void) new_rank;
    (void) user_data;
    return KIRO_CALLBACK_CONTINUE;
}

int
main ( int argc, char *argv[] )
{
    GOptionContext *context;
    GError *error = NULL;

    static gboolean server = FALSE;
    static gint iterations = 1000;
    static gint size_mb = 1;

    static GOptionEntry entries[] = {
        { "server", 's', 0, G_OPTION_ARG_NONE, &server, "Start as server (listener)", NULL },
        { "iterations", 'i', 0, G_OPTION_ARG_INT, &iterations, "Number of iterations (1000 by default)", NULL },
        { "sizemb", 'b', 0, G_OPTION_ARG_INT, &size_mb, "Size in MB for each package (1 MB by default)", NULL },
        { NULL }
    };

#if !(GLIB_CHECK_VERSION (2, 36, 0))
    g_type_init ();
#endif

    context = g_option_context_new ("[-s] | <ADDRESS> [-i <ITERATIONS>] [-b <SIZE MB>]");
    g_option_context_set_summary (context, "");
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

    if (server) {
        kiro_messenger_start_listen (messenger, argv[1], "60010", conn_callback, NULL, &error);
    }
    else {
        gulong rank = 0;
        kiro_messenger_connect (messenger, argv[1], "60010", &rank, &error);
    }

    if (error) {
        printf ("Error: %s\n", error->message);
        kiro_messenger_free (messenger);
        return -1;
    }


    if (!server) {
        gulong size_bytes = size_mb * (1024 * 1024);
        KiroMessage msg;
        msg.msg = 42;
        msg.payload = malloc (size_bytes);
        msg.size = size_bytes;

        KiroRequest request;
        request.id = 0;
        request.callback = NULL;
        request.user_data = NULL;
        request.message = &msg;
        request.peer_rank = 1;

        GTimer *timer = g_timer_new ();
        g_timer_reset (timer);

        for (gint i = 0; i < iterations; i++) {
            request.id = i;
            if (!kiro_messenger_send (messenger, &request, &error)) {
                printf ("Sending failed...\n");
                exit(-1);
            }
            if (error) {
                printf ("Error: %s\n", error->message);
                exit (-1);
            }
            while (request.status == KIRO_MESSAGE_PENDING) {};

            if (request.status != KIRO_MESSAGE_SEND_SUCCESS) {
                printf ("Sending failed with status %i\n", request.status);
                exit (-1);
            }
        }

        gdouble elapsed = g_timer_elapsed (timer, NULL);
        gdouble size_gb = (iterations * size_mb) / 1024.0;
        gdouble throughput =  size_gb / elapsed;
        g_message ("Transmitted %.2f GB in %.2f Seconds. That's %.2f GB/s (%.2f Gb/s)", size_gb, elapsed, throughput, throughput*8);
        g_timer_destroy (timer);
    }
    else {
        gboolean received = FALSE;
        KiroRequest request;
        request.id = 0;
        request.callback = callback;
        request.user_data = (gpointer) &received;

        g_message ("Messenger started. Waiting for incoming messages.");
        while (1) {
            kiro_messenger_receive (messenger, &request);
            while (!received) {};
            received = FALSE;
        }
    }

    kiro_messenger_free (messenger);
    return 0;
}








