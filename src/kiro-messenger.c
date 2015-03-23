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
    enum KiroMessengerType      type;            // Store weather we are server or client

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

    guint32                     msg_id;          // Used to hold and generate message IDs
    struct pending_message      *message;        // Keep all outstanding RDMA message MRs

    GHookList                   rec_callbacks;   // List of all receive callbacks
    GHookList                   send_callbacks;  // List of all send callbacks
};


struct pending_message {
    enum {
        KIRO_MESSAGE_SEND = 0,
        KIRO_MESSAGE_RECEIVE
    } direction;
    guint32 handle;
    gboolean message_is_mine;
    struct KiroMessage *msg;
    struct kiro_rdma_mem *rdma_mem;
};


G_DEFINE_TYPE (KiroMessenger, kiro_messenger, G_TYPE_OBJECT);

KiroMessenger *
kiro_messenger_new (void)
{
    return g_object_new (KIRO_TYPE_MESSENGER, NULL);
}


void
kiro_messenger_free (KiroMessenger *self)
{
    g_return_if_fail (self != NULL);
    if (KIRO_IS_MESSENGER (self))
        g_object_unref (self);
    else
        g_warning ("Trying to use kiro_messenger_free on an object which is not a KIRO Messenger. Ignoring...");
}


static void
kiro_messenger_init (KiroMessenger *self)
{
    g_return_if_fail (self != NULL);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);
    memset (priv, 0, sizeof (&priv));
    g_hook_list_init (&(priv->rec_callbacks), sizeof (GHook));
    g_hook_list_init (&(priv->send_callbacks), sizeof (GHook));
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


gboolean
invoke_callbacks (GHook *hook, gpointer msg)
{
    KiroMessengerCallbackFunc func = (KiroMessengerCallbackFunc)(hook->func);
    return func((struct KiroMessage *)msg, hook->data);
}


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
    KiroMessengerPrivate *priv = (KiroMessengerPrivate *)data;
    struct rdma_cm_id *conn = NULL;
    if (priv->type == KIRO_MESSENGER_SERVER)
        conn = priv->client;
    else
        conn = priv->conn;

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
    struct kiro_ctrl_msg *msg_in = (struct kiro_ctrl_msg *)ctx->cf_mr_recv->mem;
    guint type = msg_in->msg_type;
    g_debug ("Received a message from the peer of type %u", type);

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
        case KIRO_REQ_RDMA:
        {
            // The client uses the peer_mri structure to tell us the length of
            // the requested message and the 'handle', which we need to reply
            // back to match the client REQ messages with our ACK messages.
            g_debug ("Peer wants to send a message of size %lu", msg_in->peer_mri.length);
            struct kiro_rdma_mem *rdma_data_in = NULL;
            struct kiro_ctrl_msg *msg_out = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
            msg_out->msg_type = KIRO_REJ_RDMA; // REJ by default. Only change if everyhing is okay

            if (priv->message) {
                g_debug ("But only one pending message is allowed");
            }
            else if (!priv->rec_callbacks.hooks) {
                g_debug ("But noone if listening for any messages");
            }
            else {
                rdma_data_in = kiro_create_rdma_memory (conn->pd, msg_in->peer_mri.length, \
                                                       IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);

                if (!rdma_data_in) {
                    g_critical ("Failed to create message MR for peer message!");
                }
                else {
                    g_debug ("Sending message MR to peer");
                    msg_out->msg_type = KIRO_ACK_RDMA;
                    msg_out->peer_mri = *rdma_data_in->mr;
                    msg_out->peer_mri.handle = msg_in->peer_mri.handle;

                    struct pending_message *pm = (struct pending_message *)g_malloc0(sizeof (struct pending_message));
                    pm->direction = KIRO_MESSAGE_RECEIVE;
                    pm->handle = msg_in->peer_mri.handle;
                    pm->msg = (struct KiroMessage *)g_malloc0 (sizeof (struct KiroMessage));
                    pm->msg->status = KIRO_MESSAGE_PENDING;
                    pm->msg->id = msg_in->peer_mri.handle;
                    pm->msg->size = msg_in->peer_mri.length;
                    pm->msg->payload = rdma_data_in->mem;
                    pm->msg->message_handled = FALSE;
                    pm->rdma_mem = rdma_data_in;
                    priv->message = pm;
                }
            }

            if (0 > send_msg (conn, ctx->cf_mr_send)) {
                g_critical ("Failed to send RDMA credentials to peer!");
                if (rdma_data_in)
                    kiro_destroy_rdma_memory (rdma_data_in);
            }
            g_debug ("RDMA message reply sent to peer");
            break;
        }
        case KIRO_REJ_RDMA:
        {
            g_debug ("Message '%u' was rejected by the peer", msg_in->peer_mri.handle);
            if (priv->message->handle != msg_in->peer_mri.handle) {
                g_debug ("Reply is for the wrong message...");
                //
                //TODO: Cancel the current message transfer? Or do nothing?
                //
                goto done;
            }
            else {
                g_debug ("Cleaning up pending message ...");
                priv->message->rdma_mem->mem = NULL; // mem points to the original message data! DON'T FREE IT JUST YET!
                kiro_destroy_rdma_memory (priv->message->rdma_mem);
                priv->message->msg->status = KIRO_MESSAGE_SEND_FAILED;
                g_hook_list_marshal_check (&(priv->send_callbacks), FALSE, invoke_callbacks, priv->message->msg);
                if (priv->message->message_is_mine && !priv->message->msg->message_handled) {
                    g_debug ("Message is owned by the messenger and noone wants to handle it. Cleaning it up...");
                    g_free (priv->message->msg->payload);
                    g_free (priv->message->msg);
                }
                g_free (priv->message);
                priv->message = NULL;
                break;
            }
        }
        case KIRO_ACK_RDMA:
        {
            g_debug ("Got RDMA credentials for message '%u' from peer", msg_in->peer_mri.handle);
            if (priv->message->handle != msg_in->peer_mri.handle) {
                g_debug ("Reply is for the wrong message...");
                //
                //TODO: Cancel the current message transfer? Or do nothing?
                //
                goto done;
            }
            else {

                if (rdma_post_write (conn, conn, priv->message->rdma_mem->mem, priv->message->rdma_mem->size, priv->message->rdma_mem->mr, 0, \
                                    (uint64_t)msg_in->peer_mri.addr, msg_in->peer_mri.rkey)) {
                    g_critical ("Failed to RDMA_WRITE to peer: %s", strerror (errno));
                    goto cleanup;
                }

                struct ibv_wc wc;
                if (rdma_get_send_comp (conn, &wc) < 0) {
                    g_critical ("No send completion for RDMA_WRITE received: %s", strerror (errno));
                    goto cleanup;
                }

                switch (wc.status) {
                    case IBV_WC_SUCCESS:
                        g_debug ("Message RDMA transfer successfull");
                        priv->message->msg->status = KIRO_MESSAGE_SEND_SUCCESS;
                        break;
                    case IBV_WC_RETRY_EXC_ERR:
                        g_critical ("Peer no longer responding");
                        priv->message->msg->status = KIRO_MESSAGE_SEND_FAILED;
                        break;
                    case IBV_WC_REM_ACCESS_ERR:
                        g_critical ("Peer has revoked access right to write data");
                        priv->message->msg->status = KIRO_MESSAGE_SEND_FAILED;
                        break;
                    default:
                        g_critical ("Could not send message data to the peer. Status %u", wc.status);
                        priv->message->msg->status = KIRO_MESSAGE_SEND_FAILED;
                }
            }

            struct kiro_ctrl_msg *msg_out = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
            msg_out->peer_mri = msg_in->peer_mri;

            if (priv->message->msg->status == KIRO_MESSAGE_SEND_SUCCESS)
                msg_out->msg_type = KIRO_RDMA_DONE;
            else
                msg_out->msg_type = KIRO_RDMA_CANCEL;

            if (0 > send_msg (conn, ctx->cf_mr_send)) {
                //
                //TODO: If this ever happens, the peer will be in an undefined
                //state. We don't know if the peer has already cleared our
                //pending message request or not. Its almost impossible to
                //recover from this... Maybe just disconnect 'broken pipe'?
                //We'll interrupt the program for now...
                //
                g_error ("Failed to send transfer status to peer!");
            }
            g_debug ("Message transfer done.");

            cleanup:
                g_debug ("Cleaning up pending message ...");
                priv->message->rdma_mem->mem = NULL; // mem points to the original message data! DON'T FREE IT JUST YET!
                kiro_destroy_rdma_memory (priv->message->rdma_mem);
                g_hook_list_marshal_check (&(priv->send_callbacks), FALSE, invoke_callbacks, priv->message->msg);
                if (priv->message->message_is_mine && !priv->message->msg->message_handled) {
                    g_debug ("Message is owned by the messenger and noone wants to handle it. Cleaning it up...");
                    g_free (priv->message->msg->payload);
                    g_free (priv->message->msg);
                }
                g_free (priv->message);
                priv->message = NULL;
                //
                //TODO: Inform the peer about failed send?
                //
                break; //case KIRO_ACK_RDMA:
        }
        case KIRO_RDMA_DONE:
        {
            g_debug ("Peer has signalled message transfer success");
            priv->message->msg->status = KIRO_MESSAGE_RECEIVED;
            g_hook_list_marshal_check (&(priv->rec_callbacks), FALSE, invoke_callbacks, priv->message->msg);
            if (priv->message->msg->message_handled != TRUE) {
                g_debug ("Noone cared for the message. Received data will be freed.");
            }
            // -- FALL THROUGH INTENTIONAL -- //
        }
        case KIRO_RDMA_CANCEL:
        {
            g_debug ("Cleaning up pending message ...");
            kiro_destroy_rdma_memory (priv->message->rdma_mem);
            g_free (priv->message);
            priv->message = NULL;
            break;
        }
        default:
            g_debug ("Message Type %i is unknown. Ignoring...", type);
    }

done:
    //Post a generic receive in order to stay responsive to any messages from
    //the peer
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
                //Don't connect this peer any more.
                //Sorry mate!
                rdma_reject (ev->id, NULL, 0);
                goto exit;
            }

            do {
                g_debug ("Got connection request from client");

                if (priv->client) {
                    g_debug ("But we already have a client. Rejecting...");
                    rdma_reject (ev->id, NULL, 0);
                    goto exit;
                }

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

                // Connection set-up successfull. Store as client
                priv->client = ev->id;

                struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->client->context);
                ibv_req_notify_cq (priv->client->recv_cq, 0); // Make the respective Queue push events onto its event channel
                if (rdma_post_recv (priv->client, priv->client, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
                    g_critical ("Posting preemtive receive for connection failed: %s", strerror (errno));
                    goto fail;
                }

                // Create a g_io_channel wrapper for the new clients receive
                // queue event channel and add a main_loop watch to it.
                priv->rdma_ec = g_io_channel_unix_new (priv->client->recv_cq_channel->fd);
                priv->rdma_ec_id = g_io_add_watch (priv->rdma_ec, G_IO_IN | G_IO_PRI, process_rdma_event, (gpointer)priv);
                //
                // main_loop now holds a reference. We don't need ours any more
                g_io_channel_unref (priv->rdma_ec);

                break;

                fail:
                    g_warning ("Failed to accept client connection: %s", strerror (errno));
                    if (errno == EINVAL)
                        g_message ("This might happen if the client pulls back the connection request before the server can handle it.");

            } while(0);
        }
        else if (ev->event == RDMA_CM_EVENT_DISCONNECTED) {
            if (priv->type == KIRO_MESSENGER_SERVER) {
                struct kiro_connection_context *ctx = (struct kiro_connection_context *) (ev->id->context);
                if (!(ctx == priv->client->context)) {
                    g_debug ("Got disconnect request from unknown client");
                    goto exit;
                }
                else {
                    g_debug ("Got disconnect request from client");
                    g_source_remove (priv->rdma_ec_id); // this also unrefs the GIOChannel of the source. Nice.
                    priv->client = NULL;
                    priv->rdma_ec = NULL;
                    priv->rdma_ec_id = 0;

                }

                // I'm pretty sure this is not true any more. But i'll keep it
                // for now...
                /* // Note: */
                /* // The ProtectionDomain needs to be buffered and freed manually. */
                /* // Each connecting client is attached with its own pd, which we */
                /* // create manually. So we also need to clean it up manually. */
                /* // This needs to be done AFTER the connection is brought down, so we */
                /* // buffer the pointer to the pd and clean it up afterwards. */
                /* struct ibv_pd *pd = ev->id->pd; */
                /* kiro_destroy_connection (& (ev->id)); */
                /* g_free (pd); */

                kiro_destroy_connection (& (ev->id));
            }
            else {
                //
                //TODO: Server has disconnected us
                //
            }

            g_debug ("Connection closed successfully.");
        }
        else if (ev->event == RDMA_CM_EVENT_ESTABLISHED) {
            g_debug ("Client connection established");
        }
        else
            g_debug ("Event type '%i' unhandled", ev->event);

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
        // Remove the IO Channel Sources from the main loop
        // This will also unref their respective GIOChannels
        g_source_remove (priv->conn_ec_id);
        priv->conn_ec_id = 0;

        g_source_remove (priv->rdma_ec_id);
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

    G_LOCK (connection_handling);
    priv->type = role;
    priv->ec = rdma_create_event_channel ();

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
        ibv_req_notify_cq (priv->conn->recv_cq, 0); // Make the respective Queue push events onto its event channel
        if (rdma_post_recv (priv->conn, priv->conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
            g_critical ("Posting preemtive receive for connection failed: %s", strerror (errno));
            goto fail;
        }

        // Create a g_io_channel wrapper for the new receive
        // queue event channel and add a main_loop watch to it.
        priv->rdma_ec = g_io_channel_unix_new (priv->conn->recv_cq_channel->fd);
        priv->rdma_ec_id = g_io_add_watch (priv->rdma_ec, G_IO_IN | G_IO_PRI, process_rdma_event, (gpointer)priv);
        //
        // main_loop now holds a reference. We don't need ours any more
        g_io_channel_unref (priv->rdma_ec);
        g_message ("Connection to %s:%s established", address, port);
    }
    else {
        g_critical ("Messenger role needs to be either KIRO_MESSENGER_SERVER or KIRO_MESSENGER_CLIENT");
        goto fail;
    }

    // NOTE:
    // We need to move the base connection to a newly created event channel
    // because the standard event channel, which is given by the call to
    // rdma_create_ep is somehow broken and will only report certain events.
    // I have no idea if I am missing something here, or if this is a bug in the
    // ib_verbs / Infiniband driver implementation, but thats just the way it is
    // for now.
    if (rdma_migrate_id (priv->conn, priv->ec)) {
        g_critical ("Was unable to migrate connection to new Event Channel: %s", strerror (errno));
        goto fail;
    }

    priv->main_loop = g_main_loop_new (NULL, FALSE);
    g_idle_add ((GSourceFunc)stop_messenger_main_loop, priv);
    priv->conn_ec = g_io_channel_unix_new (priv->ec->fd);
    priv->conn_ec_id = g_io_add_watch (priv->conn_ec, G_IO_IN | G_IO_PRI, process_cm_event, (gpointer)priv);
    priv->main_thread = g_thread_new ("KIRO Messenger main loop", start_messenger_main_loop, priv->main_loop);
    // We gave control to the main_loop (with add_watch) and don't need our ref
    // any longer
    g_io_channel_unref (priv->conn_ec);

    G_UNLOCK (connection_handling);
    return 0;

fail:
    // kiro_destroy_connection would try to call rdma_disconnect on the given
    // connection. But the server never 'connects' to anywhere, so this would
    // cause a crash. We need to destroy the enpoint manually without disconnect
    if (priv->ec)
        rdma_destroy_event_channel (priv->ec);
    priv->ec = NULL;
    struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->conn->context);
    kiro_destroy_connection_context (&ctx);
    rdma_destroy_ep (priv->conn);
    priv->conn = NULL;
    G_UNLOCK (connection_handling);
    return -1;
}


int
kiro_messenger_submit_message (KiroMessenger *self, struct KiroMessage *msg, gboolean take_ownership)
{
    g_return_val_if_fail (self != NULL, -1);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!priv->conn)
        return -1;

    struct kiro_connection_context *ctx;
    struct rdma_cm_id *conn;
    if (priv->type == KIRO_MESSENGER_SERVER) {
        ctx = (struct kiro_connection_context *)priv->client->context;
        conn = priv->client;
    }
    else {
        ctx = (struct kiro_connection_context *)priv->conn->context;
        conn = priv->conn;
    }

    struct kiro_rdma_mem *rdma_out = (struct kiro_rdma_mem *)g_malloc0 (sizeof (struct kiro_rdma_mem));
    if (!rdma_out) {
        //
        //TODO
        //
        goto fail;
    }
    rdma_out->size = msg->size;
    rdma_out->mem = msg->payload;
    if (0 > kiro_register_rdma_memory (conn->pd, &(rdma_out->mr), msg->payload, msg->size, IBV_ACCESS_LOCAL_WRITE)) {
        //
        //TODO
        //
        goto fail;
    }

    struct pending_message *pm = (struct pending_message *)g_malloc0(sizeof (struct pending_message));
    pm->direction = KIRO_MESSAGE_SEND;
    pm->message_is_mine = take_ownership;
    pm->msg = msg;
    pm->handle = priv->msg_id++;
    pm->rdma_mem = rdma_out;
    priv->message = pm;

    struct kiro_ctrl_msg *req = (struct kiro_ctrl_msg *)ctx->cf_mr_send->mem;
    req->msg_type = KIRO_REQ_RDMA;
    req->peer_mri.length = msg->size;
    req->peer_mri.handle = pm->handle;

    if (0 > send_msg (conn, ctx->cf_mr_send)) {
        //
        //TODO
        //
        goto fail;
    }
    return 0;

fail:
    return -1;
}


gulong
kiro_messenger_add_receive_callback (KiroMessenger *self, KiroMessengerCallbackFunc *func, void *user_data)
{
    g_return_val_if_fail (self != NULL, 0);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    GHook *new_hook = g_hook_alloc (&(priv->rec_callbacks));
    new_hook->data = user_data;
    new_hook->func = (GHookCheckFunc)func;
    g_hook_append (&(priv->rec_callbacks), new_hook);
    return new_hook->hook_id;
}


gboolean
kiro_messenger_remove_receive_callback (KiroMessenger *self, gulong hook_id)
{
    g_return_val_if_fail (self != NULL, FALSE);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    return g_hook_destroy (&(priv->rec_callbacks), hook_id);
}


gulong
kiro_messenger_add_send_callback (KiroMessenger *self, KiroMessengerCallbackFunc *func, void *user_data)
{
    g_return_val_if_fail (self != NULL, 0);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    GHook *new_hook = g_hook_alloc (&(priv->send_callbacks));
    new_hook->data = user_data;
    new_hook->func = (GHookCheckFunc)func;
    g_hook_append (&(priv->send_callbacks), new_hook);
    return new_hook->hook_id;
}


gboolean
kiro_messenger_remove_send_callback (KiroMessenger *self, gulong hook_id)
{
    g_return_val_if_fail (self != NULL, FALSE);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    return g_hook_destroy (&(priv->send_callbacks), hook_id);
}


void
kiro_messenger_stop (KiroMessenger *self)
{
    g_return_if_fail (self != NULL);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!priv->conn)
        return;

    //Shut down event listening
    g_debug ("Stopping event handling...");
    priv->close_signal = TRUE;

    // Wait for the main loop to stop and clear its memory
    while (g_main_loop_is_running (priv->main_loop)) {};
    g_main_loop_unref (priv->main_loop);
    priv->main_loop = NULL;

    // Ask the main thread to join (It probably already has, but we do it
    // anyways. Just in case!)
    g_thread_join (priv->main_thread);
    priv->main_thread = NULL;

    if (priv->client) {
        kiro_destroy_connection (&(priv->client));
    }
    priv->rdma_ec = NULL;
    priv->conn_ec = NULL;
    rdma_destroy_event_channel (priv->ec);
    priv->ec = NULL;
    priv->close_signal = FALSE;

    if (priv->type  == KIRO_MESSENGER_CLIENT) {
        kiro_destroy_connection (&(priv->conn));
    }
    else {
        // kiro_destroy_connection would try to call rdma_disconnect on the given
        // connection. But the server never 'connects' to anywhere, so this would
        // cause a crash. We need to destroy the endpoint manually without disconnect
        struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->conn->context);
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->conn);
        priv->conn = NULL;
    }

    g_hook_list_clear (&(priv->rec_callbacks));
    g_message ("Messenger stopped successfully");
}

