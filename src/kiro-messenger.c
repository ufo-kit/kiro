/* Copyright (C) 2015 Timo Dritschler <timo.dritschler@kit.edu>
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
 * SECTION: kiro-messenger
 * @Short_description: KIRO RDMA Messenger
 * @Title: KiroMessenger
 *
 * KiroMessenger implements a generic messenging interface for KIRO RDMA
 * communication. A messenger can be started either as listening or connecting
 * side. However, after connecting, both sides are identical in functionality.
 * Messenger connections are allways only single-point-to-point.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <rdma/rdma_verbs.h>
#include <glib.h>
#include "kiro-messenger.h"
#include "kiro-rdma.h"


/*
 * Definition of 'private' structures and members and macro to access them
 */

#define KIRO_MESSENGER_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), KIRO_TYPE_MESSENGER, KiroMessengerPrivate))

struct _KiroMessengerPrivate {

    /* Properties */
    // PLACEHOLDER //

    /* 'Real' private structures */
    /* (Not accessible by properties) */
    gboolean                    close_signal;    // Flag used to signal event listening to stop for server shutdown
    GThread                     *main_thread;    // Main KIRO server thread
    GMainLoop                   *main_loop;      // Main loop of the server for event polling and handling

    struct rdma_event_channel   *ec;             // Main Event Channel
    struct rdma_cm_id           *conn;           // Base-Connection
    GIOChannel                  *conn_ec;        // GLib IO Channel encapsulation for the connection manager event channel
    guint                       conn_ec_id;      // ID of the source created by g_io_add_watch, needed to remove it again

    struct rdma_cm_id           *client;         // Connected client
    GIOChannel                  *rdma_ec;        // GLib IO Channel encapsulation for the rdma event channel
    guint                       rdma_ec_id;      // ID of the source created by g_io_add_watch, needed to remove it again
};


G_DEFINE_TYPE (KiroMessenger, kiro_messenger, G_TYPE_OBJECT);

KiroMessenger *
kiro_messenger_new (void)
{
    return g_object_new (KIRO_TYPE_MESSENGER, NULL);
}


void
kiro_messenger_free (KiroMessenger *server)
{
    g_return_if_fail (server != NULL);
    if (KIRO_IS_MESSENGER (server))
        g_object_unref (server);
    else
        g_warning ("Trying to use kiro_messenger_free on an object which is not a KIRO server. Ignoring...");
}


static void
kiro_messenger_init (KiroMessenger *self)
{
    g_return_if_fail (self != NULL);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);
    memset (priv, 0, sizeof (&priv));
}


static void
kiro_messenger_finalize (GObject *object)
{
    g_return_if_fail (object != NULL);
    KiroMessenger *self = KIRO_MESSENGER (object);
    //Clean up the server
    kiro_messenger_stop (self);

    G_OBJECT_CLASS (kiro_messenger_parent_class)->finalize (object);
}


static void
kiro_messenger_class_init (KiroMessengerClass *klass)
{
    g_return_if_fail (klass != NULL);
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    gobject_class->finalize = kiro_messenger_finalize;
    g_type_class_add_private (klass, sizeof (KiroMessengerPrivate));
}



/// MESSENGER SPECIFIC IMPLEMENTATIONS ///
G_LOCK_DEFINE (send_lock);
G_LOCK_DEFINE (rdma_handling);
G_LOCK_DEFINE (connection_handling);


struct rdma_cm_id*
create_endpoint (const char *address, const char *port, enum KiroMessengerType role)
{
    struct rdma_cm_id *ep = NULL;

    struct rdma_addrinfo hints, *res_addrinfo;
    memset (&hints, 0, sizeof (hints));
    hints.ai_port_space = RDMA_PS_IB;

    if (role == KIRO_MESSENGER_SERVER)
        hints.ai_flags = RAI_PASSIVE;

    char *addr_c = g_strdup (address);
    char *port_c = g_strdup (port);

    int rtn = rdma_getaddrinfo (addr_c, port_c, &hints, &res_addrinfo);
    g_free (addr_c);
    g_free (port_c);

    if (rtn) {
        g_critical ("Failed to create address information: %s", strerror (errno));
        return NULL;
    }

    struct ibv_qp_init_attr qp_attr;
    memset (&qp_attr, 0, sizeof (qp_attr));
    qp_attr.cap.max_send_wr = 10;
    qp_attr.cap.max_recv_wr = 10;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.qp_context = ep;
    qp_attr.sq_sig_all = 1;

    if (rdma_create_ep (&ep, res_addrinfo, NULL, &qp_attr)) {
        g_critical ("Endpoint creation failed: %s", strerror (errno));
        g_free (res_addrinfo);
        return NULL;
    }
    g_free (res_addrinfo);
    return  ep;
}


static int
setup_connection (struct rdma_cm_id *conn)
{
    if (!conn)
        return -1;

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)g_try_malloc0 (sizeof (struct kiro_connection_context));

    if (!ctx) {
        g_critical ("Failed to create connection context");
        return -1;
    }

    ctx->cf_mr_recv = kiro_create_rdma_memory (conn->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);
    ctx->cf_mr_send = kiro_create_rdma_memory (conn->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);

    if (!ctx->cf_mr_recv || !ctx->cf_mr_send) {
        g_critical ("Failed to register control message memory");
        goto error;
    }

    ctx->cf_mr_recv->size = ctx->cf_mr_send->size = sizeof (struct kiro_ctrl_msg);
    conn->context = ctx;

    g_debug ("Connection setup successfull");
    return 0;

error:
    kiro_destroy_connection_context (&ctx);
    return -1;
}


static inline gboolean
send_msg (struct rdma_cm_id *id, struct kiro_rdma_mem *r)
{
    gboolean retval = TRUE;
    G_LOCK (send_lock);
    g_debug ("Sending message");
    if (rdma_post_send (id, id, r->mem, r->size, r->mr, IBV_SEND_SIGNALED)) {
        retval = FALSE;
    }
    else {
        struct ibv_wc wc;
        if (rdma_get_send_comp (id, &wc) < 0) {
            retval = FALSE;
        }
        g_debug ("WC Status: %i", wc.status);
    }

    G_UNLOCK (send_lock);
    return retval;
}


static gboolean
process_rdma_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source'
    // Tell the compiler to ignore it by (void)-ing it
    (void) source;

    if (!G_TRYLOCK (rdma_handling)) {
        g_debug ("RDMA handling will wait for the next dispatch.");
        return TRUE;
    }

    g_debug ("Got message on condition: %i", condition);
    struct rdma_cm_id *conn = (struct rdma_cm_id *)data;
    struct ibv_wc wc;

    gint num_comp = ibv_poll_cq (conn->recv_cq, 1, &wc);
    if (!num_comp) {
        g_critical ("RDMA event handling was triggered, but there is no completion on the queue");
        goto end_rmda_eh;
    }
    if (num_comp < 0) {
        g_critical ("Failure getting receive completion event from the queue: %s", strerror (errno));
        goto end_rmda_eh;
    }
    g_debug ("Got %i receive events from the queue", num_comp);
    void *cq_ctx;
    struct ibv_cq *cq;
    int err = ibv_get_cq_event (conn->recv_cq_channel, &cq, &cq_ctx);
    if (!err)
        ibv_ack_cq_events (cq, 1);

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)conn->context;
    guint type = ((struct kiro_ctrl_msg *)ctx->cf_mr_recv->mem)->msg_type;
    g_debug ("Received a message from Client of type %u", type);

    switch (type) {
        case KIRO_PING:
        {
            struct kiro_ctrl_msg *msg = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
            msg->msg_type = KIRO_PONG;

            if (!send_msg (conn, ctx->cf_mr_send)) {
                g_warning ("Failure while trying to post PONG send: %s", strerror (errno));
                goto done;
            }
            break;
        }
        default:
            g_debug ("Message Type is unknow. Ignoring...");
    }

done:
    //Post a generic receive in order to stay responsive to any messages from
    //the client
    if (rdma_post_recv (conn, conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
        //TODO: Connection teardown in an event handler routine? Not a good
        //idea...
        g_critical ("Posting generic receive for event handling failed: %s", strerror (errno));
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (conn);
        goto end_rmda_eh;
    }

    ibv_req_notify_cq (conn->recv_cq, 0); // Make the respective Queue push events onto the channel

    g_debug ("Finished RDMA event handling");

end_rmda_eh:
    G_UNLOCK (rdma_handling);
    return TRUE;
}


static gboolean
process_cm_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source' and 'condition'
    // Tell the compiler to ignore them by (void)-ing them
    (void) source;
    (void) condition;

    g_debug ("CM event handler triggered");
    if (!G_TRYLOCK (connection_handling)) {
        // Unsafe to handle connection management right now.
        // Wait for next dispatch.
        g_debug ("Connection handling is busy. Waiting for next dispatch");
        return TRUE;
    }

    KiroMessengerPrivate *priv = (KiroMessengerPrivate *)data;
    struct rdma_cm_event *active_event;

    if (0 <= rdma_get_cm_event (priv->ec, &active_event)) {
        struct rdma_cm_event *ev = g_try_malloc (sizeof (*active_event));

        if (!ev) {
            g_critical ("Unable to allocate memory for Event handling!");
            rdma_ack_cm_event (active_event);
            goto exit;
        }

        memcpy (ev, active_event, sizeof (*active_event));
        rdma_ack_cm_event (active_event);

        if (ev->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
            if (TRUE == priv->close_signal) {
                //Main thread has signalled shutdown!
                //Don't connect this client any more.
                //Sorry mate!
                rdma_reject (ev->id, NULL, 0);
                goto exit;
            }

            do {
                g_debug ("Got connection request from client");

                if ( -1 == kiro_attach_qp (ev->id)) {
                    g_critical ("Could not create a QP for the new connection: %s", strerror (errno));
                    goto fail;
                }

                if (0 > setup_connection (ev->id)) {
                    g_critical ("Connection setup for client failed.");
                    rdma_reject (ev->id, NULL, 0);
                }
                else {
                    if (rdma_accept (ev->id, NULL)) {
                        kiro_destroy_connection_context (ev->id->context);
                        goto fail;
                    }
                }

                // Connection set-up successfull
                // ctx was created by 'setup_connection'
                priv->client = ev->id;

                // Create a g_io_channel wrapper for the new clients receive
                // queue event channel and add a main_loop watch to it.
                GIOChannel *rdma_ec = g_io_channel_unix_new (priv->client->recv_cq_channel->fd);
                priv->rdma_ec_id = g_io_add_watch (rdma_ec, G_IO_IN | G_IO_PRI, process_rdma_event, (gpointer)priv);
                //
                // main_loop now holds a reference. We don't need ours any more
                g_io_channel_unref (rdma_ec);

                g_debug ("Client connection established");
                break;

                fail:
                    g_warning ("Failed to accept client connection: %s", strerror (errno));
                    if (errno == EINVAL)
                        g_message ("This might happen if the client pulls back the connection request before the server can handle it.");

            } while(0);
        }
        else if (ev->event == RDMA_CM_EVENT_DISCONNECTED) {
            struct kiro_connection_context *ctx = (struct kiro_connection_context *) (ev->id->context);
            if (!(ctx == priv->client->context)) {
                g_debug ("Got disconnect request from unknown client");
                goto exit;
            }
            else {
                g_debug ("Got disconnect request from client");
                g_source_remove (priv->conn_ec_id); // this also unrefs the GIOChannel of the source. Nice.
                priv->client = NULL;
                priv->conn_ec = NULL;
                priv->conn_ec_id = 0;

            }

            // Note:
            // The ProtectionDomain needs to be buffered and freed manually.
            // Each connecting client is attached with its own pd, which we
            // create manually. So we also need to clean it up manually.
            // This needs to be done AFTER the connection is brought down, so we
            // buffer the pointer to the pd and clean it up afterwards.
            struct ibv_pd *pd = ev->id->pd;
            kiro_destroy_connection (& (ev->id));
            g_free (pd);

            g_debug ("Connection closed successfully.");
        }

exit:
        g_free (ev);
    }

    G_UNLOCK (connection_handling);
    g_debug ("CM event handling done");
    return TRUE;
}


gpointer
start_messenger_main_loop (gpointer data)
{
    g_main_loop_run ((GMainLoop *)data);
    return NULL;
}


gboolean
stop_messenger_main_loop (KiroMessengerPrivate *priv)
{
    if (priv->close_signal) {
        // Get the IO Channels and destroy them.
        // This will also unref their respective GIOChannels
        GSource *tmp = g_main_context_find_source_by_id (NULL, priv->conn_ec_id);
        g_source_destroy (tmp);
        priv->conn_ec_id = 0;

        tmp = g_main_context_find_source_by_id (NULL, priv->rdma_ec_id);
        g_source_destroy (tmp);
        priv->rdma_ec_id = 0;

        g_main_loop_quit (priv->main_loop);
        g_debug ("Event handling stopped");
        return FALSE;
    }
    return TRUE;
}


int
kiro_messenger_start (KiroMessenger *self, const char *address, const char *port, enum KiroMessengerType role)
{
    g_return_val_if_fail (self != NULL, -1);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (priv->conn) {
        g_debug ("Messenger already started.");
        return -1;
    }

    priv->conn = create_endpoint (address, port, role);
    if (!priv->conn) {
        return -1;
    }

    g_debug ("Endpoint created");

    priv->ec = priv->conn->channel;
    priv->main_loop = g_main_loop_new (NULL, FALSE);
    g_idle_add ((GSourceFunc)stop_messenger_main_loop, priv);
    GIOChannel *conn_ec = g_io_channel_unix_new (priv->ec->fd);
    priv->conn_ec_id = g_io_add_watch (conn_ec, G_IO_IN | G_IO_PRI, process_cm_event, (gpointer)priv);
    priv->main_thread = g_thread_new ("KIRO Messenger main loop", start_messenger_main_loop, priv->main_loop);

    // We gave control to the main_loop (with add_watch) and don't need our ref
    // any longer
    g_io_channel_unref (conn_ec);

    if (role == KIRO_MESSENGER_SERVER) {
        char *addr_local = NULL;
        struct sockaddr *src_addr = rdma_get_local_addr (priv->conn);

        if (!src_addr) {
            addr_local = "NONE";
        }
        else {
            addr_local = inet_ntoa (((struct sockaddr_in *)src_addr)->sin_addr);
            /*
            if(src_addr->sa_family == AF_INET)
                addr_local = &(((struct sockaddr_in*)src_addr)->sin_addr);
            else
                addr_local = &(((struct sockaddr_in6*)src_addr)->sin6_addr);
            */
        }

        if (rdma_listen (priv->conn, 0)) {
            g_critical ("Failed to put server into listening state: %s", strerror (errno));
            goto fail;
        }

        g_message ("Server bound to address %s:%s", addr_local, port);
        g_message ("Enpoint listening");

    }
    else if (role == KIRO_MESSENGER_CLIENT) {
        if (rdma_connect (priv->conn, NULL)) {
            g_critical ("Failed to establish connection to the server: %s", strerror (errno));
            goto fail;
        }

        if (0 > setup_connection (priv->conn)) {
            goto fail;
        }

        struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->conn->context);
        ibv_req_notify_cq (priv->conn->recv_cq, 0); // Make the respective Queue push events onto the channel
        if (rdma_post_recv (priv->conn, priv->conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
            g_critical ("Posting preemtive receive for connection failed: %s", strerror (errno));
            goto fail;
        }

        g_message ("Connection to %s:%s established", address, port);
    }
    else {
        g_critical ("Messenger role needs to be either KIRO_MESSENGER_SERVER or KIRO_MESSENGER_CLIENT");
        goto fail;
    }


    return 0;

fail:
    // kiro_destroy_connection would try to call rdma_disconnect on the given
    // connection. But the server never 'connects' to anywhere, so this would
    // cause a crash. We need to destroy the enpoint manually without disconnect
    priv->ec = NULL;
    struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->conn->context);
    kiro_destroy_connection_context (&ctx);
    rdma_destroy_ep (priv->conn);
    priv->conn = NULL;
    return -1;
}


void
kiro_messenger_stop (KiroMessenger *self)
{
    g_return_if_fail (self != NULL);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!priv->conn)
        return;

    //Shut down event listening
    priv->close_signal = TRUE;
    g_debug ("Event handling stopped");

    if (priv->client) {
        g_io_channel_unref (priv->rdma_ec);
        priv->rdma_ec = NULL;
        kiro_destroy_connection (&(priv->client));
    }

    // Stop the main loop and clear its memory
    g_main_loop_quit (priv->main_loop);
    g_main_loop_unref (priv->main_loop);
    priv->main_loop = NULL;

    // Ask the main thread to join (It probably already has, but we do it
    // anyways. Just in case!)
    g_thread_join (priv->main_thread);
    priv->main_thread = NULL;

    // We don't need the connection management IO channel container any more.
    // Unref and thus free it.
    g_io_channel_unref (priv->conn_ec);
    priv->conn_ec = NULL;

    priv->close_signal = FALSE;

    // kiro_destroy_connection would try to call rdma_disconnect on the given
    // connection. But the server never 'connects' to anywhere, so this would
    // cause a crash. We need to destroy the enpoint manually without disconnect
    struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->conn->context);
    kiro_destroy_connection_context (&ctx);
    rdma_destroy_ep (priv->conn);
    priv->conn = NULL;
    priv->ec = NULL;
    g_message ("Server stopped successfully");
}

