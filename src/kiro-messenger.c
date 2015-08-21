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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    gboolean                    close_signal;    // Flag used to signal event listening to stop for server shutdown
    GThread                     *main_thread;    // Main KIRO server thread
    GMainLoop                   *main_loop;      // Main loop of the server for event polling and handling

    struct rdma_cm_id           *base;           // Listening base rdma_id for server only
    struct rdma_event_channel   *ec;             // Base event channel
    GIOChannel                  *conn_ioc;       // GLib IO Channel of the eventchannel
    guint                       conn_ioc_id;     // ID of the source created by g_io_add_watch
    KiroConnectCallbackFunc     con_callback;    // Connection callback
    gpointer                    con_user_data;   // User data for the connection callback

    KiroMessage                 *message_in;     // Current received message

    GMutex                      connection_handling_lock;
    GMutex                      shutdown_lock;

    GList                       *peers;
    GList                       *rec_requests;   // Receive Requests
    GMutex                      r_queue_lock;
    gulong                      rank_counter;
    gulong                      static_counter;

    GList                       *statics;
    KiroStaticRDMAInternal      *static_request;
    GMutex                      static_rdma_lock;
};


typedef struct {
    KiroMessage                 *message_out;    // Current send message being processed
    struct kiro_rdma_mem        *rdma_mem;       // Current RDMA memory region for message
    KiroMessageCallbackFunc     callback;        // Callback to invoke upon send
    gpointer                    user_data;       // User data for the callback
    KiroRequest                 *request;        // The KiroRequest this was made from
    guint                       fail_count;      // How ofter has the message failed to be sent
} PendingMessage;


typedef struct {
    gulong                      rank;            // Rank/local-id
    struct rdma_event_channel   *ec;             // Event channel for this peer
    struct rdma_cm_id           *conn;           // Actual connection
    GIOChannel                  *rdma_ioc;       // GLib IO Channel encapsulation for the rdma event channel
    guint                       rdma_ioc_id;     // ID of the source created by g_io_add_watch, needed to remove it again
    GIOChannel                  *conn_ioc;       // GLib IO Channel of the connection management eventchannel
    guint                       conn_ioc_id;     // ID of the source created by g_io_add_watch

    GList                       *s_queue;        // Message Send Queue
    PendingMessage              *pending;        // Current pending message

    GMutex                      s_queue_lock;
    GMutex                      rdma_handling_lock;
    gboolean                    active;

    //Pointer to private data structures of the Messenger.
    //This is a workaround to make the peers be able to work independent of the
    //messenger itself.
    GList                       **peer_list;
    GList                       **rec_requests;
    GMutex                      *r_queue_lock;
} KiroPeer;


//Opaque structure (forward declared in header)
struct _KiroStaticRDMAInternal {
    struct kiro_rdma_mem *local_mem;
    struct ibv_mr remote_mem;
    gulong max_size;
    gulong peer_rank;
    gboolean valid;
};



G_DEFINE_TYPE (KiroMessenger, kiro_messenger, G_TYPE_OBJECT);


/**
 * KiroMessengerError:
 * @KIRO_MESSENGER_ERROR: Default error
 */
GQuark
kiro_messenger_get_error_quark (void)
{
    return g_quark_from_static_string ("kiro-messenger-error-quark");
}


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


//forward declare
gboolean idle_task (KiroMessengerPrivate *user_data);
gpointer start_messenger_main_loop (gpointer user_data);

static void
kiro_messenger_init (KiroMessenger *self)
{
    g_return_if_fail (self != NULL);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);
    memset (priv, 0, sizeof (&priv));

    g_mutex_init (&priv->connection_handling_lock);
    g_mutex_init (&priv->shutdown_lock);
    g_mutex_init (&priv->r_queue_lock);
    g_mutex_init (&priv->static_rdma_lock);

    g_mutex_unlock (&priv->connection_handling_lock);
    g_mutex_unlock (&priv->shutdown_lock);
    g_mutex_unlock (&priv->r_queue_lock);
    g_mutex_unlock (&priv->static_rdma_lock);

    priv->main_loop = g_main_loop_new (NULL, FALSE);
    g_idle_add_full (0, (GSourceFunc)idle_task, priv, NULL);
    priv->main_thread = g_thread_new ("KIRO Messenger main loop", start_messenger_main_loop, priv->main_loop);
}


static void
kiro_messenger_finalize (GObject *object)
{
    g_return_if_fail (object != NULL);
    KiroMessenger *self = KIRO_MESSENGER (object);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    //Clean up the server
    kiro_messenger_stop (self);

    g_main_loop_quit (priv->main_loop);
    while (g_main_loop_is_running (priv->main_loop)) {};
    g_main_loop_unref (priv->main_loop);
    priv->main_loop = NULL;

    g_thread_join (priv->main_thread);
    priv->main_thread = NULL;

    g_mutex_clear (&priv->connection_handling_lock);
    g_mutex_clear (&priv->shutdown_lock);
    g_mutex_clear (&priv->r_queue_lock);

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



/* Helper functions */
KiroContinueFlag
_internal_callback (KiroMessageStatus *status, void *user_data)
{
    KiroMessageStatus *tmp = g_malloc0 (sizeof (KiroMessageStatus));
    memcpy (tmp, status, sizeof (KiroMessageStatus));
    *(KiroMessageStatus **)user_data = tmp;
    return KIRO_CALLBACK_REMOVE;
}


static gint
_glist_comp_id (gconstpointer peer, gconstpointer id)
{
    struct rdma_cm_id *target = (struct rdma_cm_id *)id;
    KiroPeer *peer_container = (KiroPeer *)peer;

    if (peer_container->conn == target)
        return 0;

    return (gint)(peer - id);
}


KiroPeer *
find_peer_by_id (KiroMessengerPrivate *priv, struct rdma_cm_id *id)
{
    g_return_val_if_fail (priv != NULL, NULL);
    GList *e = g_list_find_custom (priv->peers, id, (GCompareFunc)_glist_comp_id);
    if (e)
        return (KiroPeer *)e->data;
    else
        return NULL;
}


static gint
_glist_comp_rank (gconstpointer peer, gconstpointer rank)
{
    gulong* target = (gulong*)rank;
    KiroPeer *peer_container = (KiroPeer *)peer;

    return (peer_container->rank - *target);
}


KiroPeer *
find_peer_by_rank (KiroMessengerPrivate *priv, gulong rank)
{
    g_return_val_if_fail (priv != NULL, NULL);
    GList *e = g_list_find_custom (priv->peers, &rank, (GCompareFunc)_glist_comp_rank);
    if (e)
        return (KiroPeer *)e->data;
    else
        return NULL;
}


struct rdma_cm_id*
create_endpoint (const char *address, const char *port, gboolean listen)
{
    struct rdma_cm_id *ep = NULL;

    struct rdma_addrinfo hints, *res_addrinfo;
    memset (&hints, 0, sizeof (hints));
    hints.ai_port_space = RDMA_PS_IB;

    if (listen)
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


static gboolean
prepare_connection (struct rdma_cm_id *conn, GError **error_out)
{
    GError *error = NULL;

    if (!conn)
        return FALSE;

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)g_try_malloc0 (sizeof (struct kiro_connection_context));

    if (!ctx) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                "Failed to create connection context");
        goto error;
    }

    ctx->cf_mr_recv = kiro_create_rdma_memory (conn->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);
    ctx->cf_mr_send = kiro_create_rdma_memory (conn->pd, sizeof (struct kiro_ctrl_msg), IBV_ACCESS_LOCAL_WRITE);

    if (!ctx->cf_mr_recv || !ctx->cf_mr_send) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                "Failed to register control message memory");
        kiro_destroy_connection_context (&ctx);
        goto error;
    }

    ctx->cf_mr_recv->size = ctx->cf_mr_send->size = sizeof (struct kiro_ctrl_msg);
    conn->context = ctx;

    if (rdma_post_recv (conn, conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                "Posting preemtive receive for new peer connection failed: %s", strerror (errno));
        kiro_destroy_connection_context (&ctx);
        goto error;
    }

    g_debug ("Connection prepared successfully");
    return TRUE;

error:
    g_propagate_error (error_out, error);
    return FALSE;
}


//forward declare... (just to keep all the helpers where they belong to
static gboolean process_rdma_event (GIOChannel *source, GIOCondition condition, gpointer data);
static gboolean process_cm_event (GIOChannel *source, GIOCondition condition, gpointer data);

KiroPeer *
create_peer (struct rdma_cm_id *conn, GError **error_out)
{
    g_return_val_if_fail (conn != NULL, NULL);
    GError *error = NULL;

    KiroPeer *peer = g_malloc0 (sizeof (KiroPeer));
    peer->conn = conn;
    peer->ec = rdma_create_event_channel ();
    if (!peer->ec) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to create new event channel for listening.");
        goto error;
    }

    if (rdma_migrate_id (peer->conn, peer->ec)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Was unable to migrate connection to new Event Channel: %s", strerror (errno));
        rdma_destroy_event_channel (peer->ec);
        goto error;
    }

    //Request notfications on the completion queue channel
    //or otherwise we won't get any feedback upon receive
    ibv_req_notify_cq (conn->recv_cq, 0);

    // Create a g_io_channel wrapper for the new peers RDMA and connection
    // management event channel and add a main_loop watch to it.
    peer->conn_ioc = g_io_channel_unix_new (peer->ec->fd);
    peer->conn_ioc_id = g_io_add_watch_full (peer->conn_ioc, 0, G_IO_IN | G_IO_PRI, process_cm_event, (gpointer)peer, NULL);
    peer->rdma_ioc = g_io_channel_unix_new (peer->conn->recv_cq_channel->fd);
    peer->rdma_ioc_id = g_io_add_watch_full (peer->rdma_ioc, -100, G_IO_IN | G_IO_PRI, process_rdma_event, (gpointer)peer, NULL);

    // main_loop now holds a reference. We don't need ours any more
    g_io_channel_unref (peer->conn_ioc);
    g_io_channel_unref (peer->rdma_ioc);
    g_mutex_init (&peer->rdma_handling_lock);
    g_mutex_init (&peer->s_queue_lock);
    return peer;

error:
    g_propagate_error (error_out, error);
    g_free (peer);
    return NULL;
}


static void
destroy_peer (gpointer peer_in)
{

    //TODO
    //Make sure to cancel all pending messages and call their send callbacks
    /* if (peer->rdma_mem) */
    /*     kiro_destroy_rdma_memory (peer->rdma_mem); */

    KiroPeer *peer = (KiroPeer *)peer_in;
    g_debug ("Deactivating peer with rank '%lu'.", peer->rank);
    peer->active = FALSE;
    while (peer->pending) {};
    g_source_remove (peer->conn_ioc_id); // this also unrefs the GIOChannel of the source. Nice.
    g_source_remove (peer->rdma_ioc_id); // this also unrefs the GIOChannel of the source. Nice.
    kiro_destroy_connection ( &(peer->conn));
    rdma_destroy_event_channel (peer->ec);


    g_free (peer);
    *peer->peer_list = g_list_remove (*peer->peer_list, peer);

    g_debug ("Peer deactivated successfully.");
}


static inline gboolean
send_msg (struct rdma_cm_id *id, struct kiro_rdma_mem *r, uint32_t imm_data)
{
    gboolean retval = TRUE;

    struct ibv_sge sge;

	sge.addr = (uint64_t) (uintptr_t) r->mem;
	sge.length = (uint32_t) r->size;
	sge.lkey = r->mr ? r->mr->lkey : 0;


	struct ibv_send_wr wr, *bad;

	wr.wr_id = (uintptr_t) id;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_SEND_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl (imm_data); // Needs to be network byte order

    g_debug ("Sending message");
    int ret = ibv_post_send (id->qp, &wr, &bad);
    if (ret) {
        retval = FALSE;
        g_debug ("ibv_post_send failed with: %i", ret);
    }
    else {
        struct ibv_wc wc;
        if (rdma_get_send_comp (id, &wc) < 0) {
            retval = FALSE;
        }
        g_debug ("WC Status: %i", wc.status);
    }
    return retval;
}


gpointer
start_messenger_main_loop (gpointer data)
{
    g_main_loop_run ((GMainLoop *)data);
    return NULL;
}


gboolean
idle_task (KiroMessengerPrivate *priv)
{
    GError *error = NULL;

    if (priv->close_signal)
        return TRUE;

    g_mutex_lock (&priv->connection_handling_lock);
    GList *le = g_list_first (priv->peers);
    while (le) {
        KiroPeer *peer = (KiroPeer *)le->data;

        g_mutex_lock (&peer->s_queue_lock);
        GList *pl = g_list_first (peer->s_queue);
        PendingMessage *pm = NULL;
        if (pl && !peer->pending) {
            peer->pending = pm = (PendingMessage *)pl->data;

            struct kiro_connection_context *ctx = (struct kiro_connection_context *)peer->conn->context;
            struct kiro_ctrl_msg *request = (struct kiro_ctrl_msg *)ctx->cf_mr_send->mem;
            KiroMessage *msg = pm->message_out;

            if (msg->size > 0) {
                g_debug ("Sending message of type '%u' and size '%lu'", msg->msg, msg->size);
                request->msg_type = KIRO_REQ_RDMA;
                request->peer_mri = *(pm->rdma_mem->mr);
                request->peer_mri.handle = GPOINTER_TO_UINT (msg);
            }
            else {
                g_debug ("Sending stub message of type '%u'", msg->msg);
                // STUB message
                request->msg_type = KIRO_MSG_STUB;
                request->peer_mri.handle = GPOINTER_TO_UINT (msg);
            }

            if (0 > send_msg (peer->conn, ctx->cf_mr_send, msg->msg)) {
                g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                             "Failed to RDMA_SEND to peer '%lu'.", pm->request->peer_rank);
                goto done;
            }
            g_debug ("RDMA_SEND successfull");
        }
done:
        if (error) {
            g_debug ("Sending message failed (Try %u): '%s'", pm->fail_count, error->message);
            if (++(pm->fail_count) >= 3) {
                g_debug ("Message sending failed after %u times. Giving up.", pm->fail_count);
                peer->s_queue = g_list_delete_link (peer->s_queue, pl);
                //TODO
                //Cancel message
            }
            g_error_free (error);
            error = NULL;
        }
        else {
            peer->s_queue = g_list_delete_link (peer->s_queue, pl);
        }

        le = g_list_next (le);
        g_mutex_unlock (&peer->s_queue_lock);
    }
    g_mutex_unlock (&priv->connection_handling_lock);

    return TRUE;
}


static gboolean
process_rdma_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source'
    // Tell the compiler to ignore it by (void)-ing it
    (void) source;
    (void) condition;
    KiroPeer *peer = (KiroPeer *)data;

    if (!g_mutex_trylock (&peer->rdma_handling_lock)) {
        g_debug ("RDMA handling will wait for the next dispatch.");
        return TRUE;
    }

    struct rdma_cm_id *conn = peer->conn;
    struct ibv_wc wc;

    gint num_comp = ibv_poll_cq (conn->recv_cq, 1, &wc);
    if (!num_comp) {
        g_critical ("RDMA event handling was triggered, but there is no completion on the queue");
        goto end_rdma_eh;
    }
    if (num_comp < 0) {
        g_critical ("Failure getting receive completion event from the queue: %s", strerror (errno));
        goto end_rdma_eh;
    }
    g_debug ("Got %i receive events from the queue", num_comp);
    void *cq_ctx;
    struct ibv_cq *cq;
    int err = ibv_get_cq_event (conn->recv_cq_channel, &cq, &cq_ctx);

    //TODO
    //Add these to a pool and ACK multiple events at once
    if (!err)
        ibv_ack_cq_events (cq, 1);

    struct kiro_connection_context *ctx = (struct kiro_connection_context *)conn->context;
    struct kiro_ctrl_msg *msg_in = (struct kiro_ctrl_msg *)ctx->cf_mr_recv->mem;
    guint type = msg_in->msg_type;
    g_debug ("Received a message from peer '%lu' of type %u",peer->rank, type);

    switch (type) {
        /* case KIRO_PING: */
        /* { */
        /*     struct kiro_ctrl_msg *msg_out = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem); */
        /*     msg_out->msg_type = KIRO_PONG; */

        /*     if (!send_msg (conn, ctx->cf_mr_send, 0)) { */
        /*         g_warning ("Failure while trying to post PONG send: %s", strerror (errno)); */
        /*         goto done; */
        /*     } */
        /*     break; */
        /* } */
        case KIRO_REQ_STATIC:
        {
            g_debug ("Peer %lu wants static RDMA of size %lu", peer->rank, msg_in->peer_mri.length);

            // Shit... the peers need to know about static resquest...
            // But that lies in priv :/  What now?

            break;
        }
        case KIRO_MSG_STUB:
        {
            g_debug ("Got a stub message from peer '%lu'.", peer->rank);
            struct kiro_ctrl_msg *reply = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
            reply->msg_type = KIRO_REJ_RDMA;
            reply->peer_mri = msg_in->peer_mri;

            if ((*peer->rec_requests)) {
                reply->msg_type = KIRO_ACK_MSG;
                g_debug ("Sending ACK message");
            }
            else
                g_debug ("But noone if listening for any messages");

            if (0 > send_msg (conn, ctx->cf_mr_send, 0)) {
                g_warning ("Failure while trying to send ACK: %s", strerror (errno));
            }

            if (reply->msg_type == KIRO_ACK_MSG) {
                g_debug ("Dispatching received message");

                g_mutex_lock (peer->r_queue_lock);
                KiroRequest *req = (g_list_first (*peer->rec_requests))->data;
                *peer->rec_requests = g_list_delete_link (*peer->rec_requests, g_list_first (*peer->rec_requests));
                g_mutex_unlock (peer->r_queue_lock);

                req->status = KIRO_MESSAGE_RECEIVED;
                req->peer_rank = peer->rank;
                req->message = (KiroMessage *)g_malloc0 (sizeof (KiroMessage));
                req->message->msg = ntohl (wc.imm_data);
                req->message->size = 0;
                req->message->payload = NULL;

                if (req->callback) {
                    (*req->callback) (req, req->user_data);
                }
            }

            break;
        }
        case KIRO_ACK_MSG:
        {
            g_debug ("Got ACK for message from peer '%lu'", peer->rank);
            g_debug ("Calling send-callback");

            KiroRequest *req = peer->pending->request;
            req->status = KIRO_MESSAGE_SEND_SUCCESS;

            if (req->callback) {
                (*req->callback) (req, req->user_data);
            }

            g_free (peer->pending);
            peer->pending = NULL;
            break;
        }
        case KIRO_REQ_RDMA:
        {
            // Peer has send us peer_mri information where to pickup the
            // message. Read it and send an KIRO_ACK_RDMA once the RDMA_READ was
            // successful
            g_debug ("Peer wants us to read a message of size %lu", msg_in->peer_mri.length);
            struct kiro_ctrl_msg *msg_out = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
            msg_out->msg_type = KIRO_REJ_RDMA; // REJ by default. Only change if everything is okay
            msg_out->peer_mri = msg_in->peer_mri;


            struct kiro_rdma_mem *rdma_data_in = NULL;
            if (!peer->active) {
                g_debug ("Peer is being disposed. Won't accept message.");
                goto reject;
            }
            else if (!(*peer->rec_requests)) {
                g_debug ("But no one is listening for any messages");
                goto reject;
            }
            else {
                rdma_data_in = kiro_create_rdma_memory (conn->pd, msg_in->peer_mri.length, IBV_ACCESS_LOCAL_WRITE);

                if (!rdma_data_in) {
                    g_critical ("Failed to create message MR for peer message!");
                    goto reject;
                }
                else {

                    if (rdma_post_read (conn, conn, rdma_data_in->mem, rdma_data_in->mr->length, rdma_data_in->mr, 0, \
                                        (uint64_t)msg_in->peer_mri.addr, msg_in->peer_mri.rkey)) {
                        g_critical ("Failed to RDMA_READ from peer '%lu': %s", peer->rank, strerror (errno));
                        goto reject;
                    }

                    struct ibv_wc send_wc;
                    if (rdma_get_send_comp (conn, &send_wc) < 0) {
                        g_critical ("No send-completion for RDMA_WRITE received: %s", strerror (errno));
                        kiro_destroy_rdma_memory (rdma_data_in);
                        goto reject;
                    }

                    switch (send_wc.status) {
                        case IBV_WC_SUCCESS:
                            g_debug ("Message RDMA read successful");
                            msg_out->msg_type = KIRO_ACK_RDMA;
                            break;
                        case IBV_WC_RETRY_EXC_ERR:
                            g_critical ("Peer '%lu' no longer responding", peer->rank);
                            goto reject;
                            break;
                        case IBV_WC_REM_ACCESS_ERR:
                            g_critical ("Peer '%lu' has revoked access right to read data", peer->rank);
                            goto reject;
                            break;
                        default:
                            g_critical ("Could not read message data from peer '%lu'. Status %u", peer->rank, send_wc.status);
                            goto reject;
                    }
                }
            }

        reject:

            //TODO
            //set the reason for reject as immediate data

            if (0 > send_msg (conn, ctx->cf_mr_send, 0)) {
                //FIXME: If this ever happens, the peer will be in an undefined
                //state. We can only recover from this, if the peer has a clever
                //enough timeout/cancellation mechanism.
                g_critical ("Failed to send RDMA reply to peer!");
            }
            else {
                g_debug ("RDMA message reply sent to peer");
            }

            if (msg_out->msg_type == KIRO_ACK_RDMA) {
                // transfer successful. Dispatch message
                g_debug ("Dispatching received message");

                g_mutex_lock (peer->r_queue_lock);
                KiroRequest *req = (g_list_first (*peer->rec_requests))->data;
                *peer->rec_requests = g_list_delete_link (*peer->rec_requests, g_list_first (*peer->rec_requests));
                g_mutex_unlock (peer->r_queue_lock);

                req->status = KIRO_MESSAGE_RECEIVED;
                req->peer_rank = peer->rank;
                req->message = (KiroMessage *)g_malloc0 (sizeof (KiroMessage));
                req->message->msg = ntohl (wc.imm_data);
                req->message->size = rdma_data_in->size;
                req->message->payload = rdma_data_in->mem;

                if (req->callback) {
                    (*req->callback) (req, req->user_data);
                }
            }
            //Disattach the payload so kiro_destroy_rdma_memory does not free it
            if (rdma_data_in)
                rdma_data_in->mem = NULL;
            kiro_destroy_rdma_memory (rdma_data_in);
            break;
        }
        case KIRO_REJ_RDMA:
        {
            g_debug ("Message '%u' was rejected by the peer", msg_in->peer_mri.handle);
            if (!peer->pending->message_out) {
                g_debug ("But there is no pending message...");
                break;
            }
            else {
                KiroRequest *req = peer->pending->request;
                req->status = KIRO_MESSAGE_REJ_WITH_NOT_LISTENING;

                if (req->callback) {
                    g_debug ("Calling send-callback");
                    (*req->callback) (req, req->user_data);
                }

                peer->pending->rdma_mem->mem = NULL;
                kiro_destroy_rdma_memory (peer->pending->rdma_mem);
                g_free (peer->pending);
                peer->pending = NULL;
                break;
            }
        }
        case KIRO_ACK_RDMA:
        {
            g_debug ("Peer has successfully received our message.");

            KiroRequest *req = peer->pending->request;
            req->status = KIRO_MESSAGE_SEND_SUCCESS;

            if (req->callback) {
                g_debug ("Calling send-callback");
                (*req->callback) (req, req->user_data);
            }

            peer->pending->rdma_mem->mem = NULL;
            kiro_destroy_rdma_memory (peer->pending->rdma_mem);
            g_free (peer->pending);
            peer->pending = NULL;
            break;
        }
        default:
            g_debug ("Message Type %i is unknown. Ignoring...", type);
    }

    //Post a generic receive in order to stay responsive to any messages from
    //the peer
    if (rdma_post_recv (conn, conn, ctx->cf_mr_recv->mem, ctx->cf_mr_recv->size, ctx->cf_mr_recv->mr)) {
        g_critical ("Posting generic receive for event handling failed: %s", strerror (errno));
        destroy_peer (peer);
    }

    ibv_req_notify_cq (conn->recv_cq, 0); // Make the respective Queue push events onto the channel

end_rdma_eh:
    g_debug ("Finished RDMA event handling");
    g_mutex_unlock (&peer->rdma_handling_lock);
    return TRUE;
}


static gboolean
handle_connection_request (KiroMessenger *messenger, struct rdma_cm_event *ev, GError **error_out)
{
    GError *error = NULL;
    GError *sub_error = NULL;

    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (messenger);

    if (TRUE == priv->close_signal) {
        //Main thread has signalled shutdown!
        //Don't connect this peer any more.
        //Sorry mate!
        g_debug ("Peer connection rejected due to shutdown request");
        rdma_reject (ev->id, NULL, 0);
        return FALSE;
    }
    g_debug ("Got connection request from peer");

    if ( -1 == kiro_attach_qp (ev->id)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Could not create a QP for the new connection: %s", strerror (errno));
        rdma_reject (ev->id, NULL, 0);
        goto error;
    }

    if (!prepare_connection (ev->id, &sub_error)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Connection setup for peer failed: '%s'", sub_error->message);
        g_error_free (sub_error);
        rdma_reject (ev->id, NULL, 0);
        goto error;
    }

    if (rdma_accept (ev->id, NULL)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                "Connection-accept for peer failed.");
        kiro_destroy_connection_context (ev->id->context);
        rdma_reject (ev->id, NULL, 0);
        goto error;
    }
    g_debug ("Peer connection established");


    // Connection set-up successfull. Create peer and store
    KiroPeer *peer = create_peer (ev->id, &sub_error);
    if (!peer) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Creation of peer structure failed: '%s'", sub_error->message);
        g_error_free (sub_error);
        rdma_destroy_ep (ev->id);
        goto error;
    }

    peer->rank = ++(priv->rank_counter);
    peer->peer_list = &(priv->peers);
    peer->rec_requests = &(priv->rec_requests);
    peer->r_queue_lock = &priv->r_queue_lock;
    peer->active = TRUE;
    priv->peers = g_list_prepend (priv->peers, peer);

    // Invoke the connection callback
    KiroContinueFlag ret = (priv->con_callback) (peer->rank, priv->con_user_data);
    if (ret == KIRO_CALLBACK_REMOVE) {
        kiro_messenger_stop_listen (messenger, &sub_error);

        // PRINT SUB ERROR
    }

    // All fine!
    g_debug ("Peer rank %lu assigned", peer->rank);
    return TRUE;

error:
    g_propagate_error (error_out, error);
    g_debug ("Peer connection failed: '%s'", error->message);
    return FALSE;
}

static gboolean
process_cm_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source' and 'condition'
    // Tell the compiler to ignore them by (void)-ing them
    (void) source;
    (void) condition;
    KiroPeer *peer = (KiroPeer *)data;

    g_debug ("CM event handler for peer '%lu' triggered", peer->rank);
    struct rdma_cm_event *active_event;

    if (0 <= rdma_get_cm_event (peer->ec, &active_event)) {
        struct rdma_cm_event *ev = g_malloc (sizeof (*active_event));
        memcpy (ev, active_event, sizeof (*active_event));
        rdma_ack_cm_event (active_event);

        if (ev->event == RDMA_CM_EVENT_DISCONNECTED) {
            if (!g_mutex_trylock (&peer->rdma_handling_lock)) {
                g_critical ("Peer disconnected in the middle of a running transmission!");
                //TODO:
                //Anything special to take care of here?
            }
            destroy_peer (peer);
        }
        else if (ev->event == RDMA_CM_EVENT_ESTABLISHED) {
            g_debug ("Peer connection for peer '%lu' established", peer->rank);
        }
        else
            g_debug ("Event type '%i' unhandled", ev->event);

        g_free (ev);
    }
    g_debug ("CM event handling done");
    return TRUE;
}


static gboolean
base_cm_event_handler (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source' and 'condition'
    // Tell the compiler to ignore them by (void)-ing them
    (void) source;
    (void) condition;
    KiroMessenger *messenger = (KiroMessenger *) data;
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (messenger);

    g_debug ("CM event handler triggered");
    if (!g_mutex_trylock (&priv->connection_handling_lock)) {
        // Unsafe to handle connection management right now.
        // Wait for next dispatch.
        g_debug ("Connection handling is busy. Waiting for next dispatch");
        return TRUE;
    }

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

        GError *error = NULL;
        if (ev->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
            if (!handle_connection_request (messenger, ev, &error)) {
                g_critical ("%s", error->message);
                g_error_free (error);
            }
        }
        else
            g_debug ("Event type '%i' not handled by base-listener", ev->event);

exit:
        g_free (ev);
    }

    g_mutex_unlock (&priv->connection_handling_lock);
    g_debug ("CM event handling done");
    return TRUE;
}


int
kiro_messenger_start_listen (KiroMessenger *self, const char *address, const char *port,
                             KiroConnectCallbackFunc con_callback, gpointer user_data,
                             GError **error_out)
{
    g_return_val_if_fail (self != NULL, -1);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    GError *error = NULL;

    if (priv->base) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Messenger is already listening.");
        g_propagate_error (error_out, error);
        return -1;
    }

    g_mutex_lock (&priv->connection_handling_lock);

    priv->base = create_endpoint (address, port, TRUE);
    if (!priv->base) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to create internal InfiniBand endpoint.");
        g_propagate_error (error_out, error);
        return -1;
    }
    g_debug ("Endpoint created");


    char *addr_local = NULL;
    struct sockaddr *src_addr = rdma_get_local_addr (priv->base);

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

    if (rdma_listen (priv->base, 0)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to set internal InfiniBand endpoint to listening state.");
        goto fail;
    }

    // NOTE:
    // We need to move the base connection to a newly created event channel
    // because the standard event channel, which is given by the call to
    // rdma_create_ep is somehow broken and will only report certain events.
    // I have no idea if I am missing something here, or if this is a bug in the
    // ib_verbs / Infiniband driver implementation, but thats just the way it is
    // for now.
    priv->ec = rdma_create_event_channel ();
    if (!priv->ec) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to create new event channel for listening.");
        goto fail;
    }

    if (rdma_migrate_id (priv->base, priv->ec)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to migrate internal InfiniBand entpoint to new event channel.");
        goto fail;
    }

    priv->con_callback = con_callback;
    priv->con_user_data = user_data;

    priv->conn_ioc = g_io_channel_unix_new (priv->ec->fd);
    priv->conn_ioc_id = g_io_add_watch_full (priv->conn_ioc, 0, G_IO_IN | G_IO_PRI, base_cm_event_handler, (gpointer)self, NULL);

    // We gave control to the main_loop (with add_watch) and don't need our ref
    // any longer
    g_io_channel_unref (priv->conn_ioc);

    g_debug ("Server bound to address %s:%s", addr_local, port);
    g_debug ("Enpoint listening");


    g_mutex_unlock (&priv->connection_handling_lock);
    return 0;

fail:
    // kiro_destroy_connection would try to call rdma_disconnect on the given
    // connection. But the server never 'connects' to anywhere, so this would
    // cause a crash. We need to destroy the enpoint manually without disconnect
    if (priv->base) {
        struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->base->context);
        kiro_destroy_connection_context (&ctx);
        rdma_destroy_ep (priv->base);
    }
    priv->base = NULL;

    if (priv->ec) {
        rdma_destroy_event_channel (priv->ec);
        priv->ec = NULL;
    }

    g_mutex_unlock (&priv->connection_handling_lock);
    g_propagate_error (error_out, error);
    return -1;
}


void
kiro_messenger_stop_listen (KiroMessenger *self, GError **error_out)
{
    g_return_if_fail (self != NULL);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!priv->base)
        return;

    //Right now, there is no error that we could handle
    (void) error_out;

    // Remove the IO Channel source from the main loop
    // This will also unref its respective GIOChannels
    g_source_remove (priv->conn_ioc_id);
    priv->conn_ioc_id = 0;
    priv->conn_ioc = NULL;

    // kiro_destroy_connection would try to call rdma_disconnect on the
    // given connection. But the servers base connection never 'connects' to
    // anywhere, so this would cause a crash in the driver. We need to
    // destroy the endpoint manually without disconnect
    struct kiro_connection_context *ctx = (struct kiro_connection_context *) (priv->base->context);
    kiro_destroy_connection_context (&ctx);
    rdma_destroy_ep (priv->base);
    priv->base = NULL;

    rdma_destroy_event_channel (priv->ec);
    priv->ec = NULL;

    priv->con_callback = NULL;
    priv->con_user_data = NULL;

    g_debug ("Messenger stopped listening");
}


void
kiro_messenger_connect (KiroMessenger *self, const gchar *addr, const gchar *port, gulong *rank, GError **error_out)
{
    g_return_if_fail (self != NULL);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    GError *error = NULL;
    GError *sub_error = NULL;

    struct rdma_cm_id *conn = create_endpoint (addr, port, FALSE);
    if (!conn) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to create internal InfiniBand endpoint.");
        goto fail;
    }

    if (rdma_connect (conn, NULL)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to connect to %s:%s! Error: '%s'", addr, port, strerror (errno));
        goto fail;
    }

    if (!prepare_connection (conn, &sub_error)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to setup endoint: '%s'", sub_error->message);
        g_error_free (sub_error);
        rdma_disconnect (conn);
        goto fail;
    }

    // Connection set-up successfull. Create peer and store
    KiroPeer *peer = create_peer (conn, &sub_error);
    if (!peer) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Creation of peer structure failed: '%s'", sub_error->message);
        g_error_free (sub_error);
        rdma_disconnect (conn);
        goto fail;
    }


    peer->rank = ++(priv->rank_counter);
    peer->peer_list = &(priv->peers);
    peer->rec_requests = &(priv->rec_requests);
    peer->r_queue_lock = &priv->r_queue_lock;
    peer->active = TRUE;
    priv->peers = g_list_prepend (priv->peers, peer);

    // All fine!
    *rank = peer->rank;
    g_debug ("Connection was assigned rank %lu", peer->rank);
    return;

fail:
    g_propagate_error (error_out, error);
    rdma_destroy_ep (conn);
}


gboolean
kiro_messenger_send (KiroMessenger *self, KiroRequest *request, GError **error_out)
{
    g_return_val_if_fail (self != NULL, FALSE);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    GError *error = NULL;

    if (!request->message) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "No KiroMessage supplied in send request.");
        g_propagate_error (error_out, error);
        return FALSE;
    }

    if (!priv->peers) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Trying to submit a message on a messenger which is not connected.");
        g_propagate_error (error_out, error);
        return FALSE;
    }

    g_mutex_lock (&priv->connection_handling_lock);

    KiroPeer *peer = find_peer_by_rank (priv, request->peer_rank);
    if (!peer) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "No peer with rank '%lu' is connected.", request->peer_rank);
        goto fail;
    }

    if (!peer->active) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Peer with rank '%lu' is no longer active.", request->peer_rank);
        goto fail;
    }

    KiroMessage *msg = request->message; //easy access

    PendingMessage *pm = g_malloc0 (sizeof (PendingMessage));
    pm->message_out = request->message;
    pm->callback = request->callback;
    pm->user_data = request->user_data;
    pm->request = request;

    if (msg->size > 0) {
        g_debug ("Registering message memory of size '%lu' and type '%u'", msg->size, msg->msg);
        struct kiro_rdma_mem *rdma_out = (struct kiro_rdma_mem *)g_malloc0 (sizeof (struct kiro_rdma_mem));
        if (!rdma_out) {
            g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                         "Failed to create RDMA memory for transmission.");
            g_free (pm);
            goto fail;
        }
        rdma_out->size = msg->size;
        rdma_out->mem = msg->payload;
        if (0 > kiro_register_rdma_memory (peer->conn->pd, &(rdma_out->mr), msg->payload, msg->size,
                                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ)) {
            g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                         "Failed to register (pin) RDMA memory for transmission.");
            g_free (pm);
            goto fail;
        }
        pm->rdma_mem = rdma_out;
    }


    g_mutex_lock (&peer->s_queue_lock);
    peer->s_queue = g_list_append (peer->s_queue, pm);
    request->status = KIRO_MESSAGE_PENDING;
    g_mutex_unlock (&peer->s_queue_lock);

    g_mutex_unlock (&priv->connection_handling_lock);

    return TRUE;

fail:
    g_mutex_unlock (&priv->connection_handling_lock);
    g_propagate_error (error_out, error);
    return FALSE;
}



gboolean
kiro_messenger_send_blocking (KiroMessenger *self, KiroMessage *msg, gulong peer_rank, GError **error_out)
{
    KiroRequest request;
    request.message = msg;
    request.peer_rank = peer_rank;
    request.callback = NULL;
    request.status = KIRO_MESSAGE_PENDING;

    if (!kiro_messenger_send (self, &request, error_out)) {
        return  FALSE;
    }
    while (request.status == KIRO_MESSAGE_PENDING) {};

    gboolean ret = (request.status == KIRO_MESSAGE_SEND_SUCCESS);

    //TODO
    //Set error according to request.status

    return ret;
}


gboolean
kiro_messenger_receive (KiroMessenger *self, KiroRequest *request)
{
    g_return_val_if_fail (self != NULL, FALSE);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    g_debug ("Starting to listen on request ID %lu", request->id);

    g_mutex_lock (&priv->r_queue_lock);
    priv->rec_requests = g_list_append (priv->rec_requests, request);
    g_mutex_unlock (&priv->r_queue_lock);

    return TRUE;
}


KiroStaticRDMA *kiro_messenger_request_static (KiroMessenger *self, gulong size, gulong peer_rank, GError **error_out)
{
    GError *error = NULL;
    g_return_val_if_fail (self != NULL, NULL);

    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!priv->peers) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Trying to perform an action on a messenger which is not connected.");
        g_propagate_error (error_out, error);
        return NULL;
    }

    g_mutex_lock (&priv->connection_handling_lock);

    KiroPeer *peer = find_peer_by_rank (priv, peer_rank);
    if (!peer) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "No peer with rank '%lu' is connected.", peer_rank);
        g_propagate_error (error_out, error);
        g_mutex_unlock (&priv->connection_handling_lock);
        return NULL;
    }

    if (!peer->active) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Peer with rank '%lu' is no longer active.", peer_rank);
        g_propagate_error (error_out, error);
        g_mutex_unlock (&priv->connection_handling_lock);
        return NULL;
    }


    struct kiro_rdma_mem *rdma_mem = kiro_create_rdma_memory (peer->conn->pd, size, IBV_ACCESS_REMOTE_WRITE
                                                              | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);

    if (!rdma_mem) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Unable to allocate RDMA memory of size %lu", size);
        g_propagate_error (error_out, error);
        g_mutex_unlock (&priv->connection_handling_lock);
        return NULL;
    }

    g_mutex_lock (&priv->static_rdma_lock);

    KiroStaticRDMA *stat = g_malloc0 (sizeof (KiroStaticRDMA));
    stat->id = priv->static_counter++;
    stat->size = size;
    stat->mem = rdma_mem->mem;
    stat->peer_rank = peer_rank;
    stat->internal = g_malloc (sizeof (KiroStaticRDMAInternal));
    stat->internal->peer_rank = peer_rank;
    stat->internal->local_mem = rdma_mem;
    stat->internal->valid = FALSE;

    priv->static_request = stat->internal;

    g_mutex_lock (&peer->rdma_handling_lock);

    g_debug ("Requesting STATIC_RDMA of size %lu from peer %lu.", size, peer_rank);
    struct kiro_connection_context *ctx = (struct kiro_connection_context *)peer->conn->context;
    struct kiro_ctrl_msg *static_request = (struct kiro_ctrl_msg *)ctx->cf_mr_send->mem;
    static_request->msg_type = KIRO_REQ_STATIC;
    static_request->peer_mri = *(stat->internal->local_mem->mr);

    if (0 > send_msg (peer->conn, ctx->cf_mr_send, stat->id)) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to RDMA_SEND to peer '%lu'.", peer_rank);
        g_mutex_unlock (&peer->rdma_handling_lock);
        goto fail;
    }

    g_mutex_unlock (&peer->rdma_handling_lock);

    g_debug ("Request sent successfully.");

    //Wait for response. The event-handler will remove this pointer
    //once the remote side has responded.
    while (priv->static_request) { usleep (100); };

    if (stat->internal->valid == FALSE) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Failed to RDMA_SEND to peer '%lu'.", peer_rank);
        goto fail;
    }

    g_mutex_unlock (&priv->static_rdma_lock);
    g_mutex_unlock (&priv->connection_handling_lock);
    return stat;


fail:
    g_free (stat->internal);
    kiro_destroy_rdma_memory (rdma_mem);
    g_free (stat);
    g_mutex_unlock (&priv->static_rdma_lock);
    g_mutex_unlock (&priv->connection_handling_lock);
    return NULL;
};


KiroStaticRDMA* kiro_messenger_accept_static (KiroMessenger *self, gulong max_size, GError **error_out)
{
    GError *error = NULL;

    g_return_val_if_fail (self != NULL, NULL);

    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!priv->peers) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Trying to perform an action on a messenger which is not connected.");
        g_propagate_error (error_out, error);
        return NULL;
    }

    g_mutex_lock (&priv->static_rdma_lock);

    KiroStaticRDMAInternal *internal = g_malloc0 (sizeof (KiroStaticRDMAInternal));
    internal->max_size = max_size;
    internal->valid = FALSE;
    priv->static_request = internal;

    g_debug ("Setting up request to wait for STATIC_RDMA with max size %lu", max_size);

    //Wait for incoming request. The event-hanlder will remove this pointer
    //once a request from a remote peer was accepted.
    while (priv->static_request) { usleep (100); };

    if (internal->valid == FALSE) {
        g_free (internal);
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Waiting for an incoming STATIC_RDMA request got interrupted.");
        g_propagate_error (error_out, error);
        return NULL;
    }

    g_debug ("STATIC_RDMA request accepted from peer %lu, with size %lu", internal->peer_rank, internal->local_mem->size);

    KiroStaticRDMA *out = g_malloc0 (sizeof (KiroStaticRDMA));
    out->mem = internal->local_mem->mem;
    out->size = internal->local_mem->size;
    out->peer_rank = internal->peer_rank;
    out->id = priv->static_counter++;
    out->internal = internal;

    return out;
};


gboolean kiro_messenger_release_static (KiroMessenger *self, KiroStaticRDMA* static_mem, GError **error_out)
{
    GError *error = NULL;

    g_return_val_if_fail (self != NULL, FALSE);

    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!static_mem->internal) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "KiroStaticRDMA struct corrupted ('internal' points to NULL). Doing nothing.");
        g_propagate_error (error_out, error);
        return FALSE;
    }

    g_mutex_lock (&priv->static_rdma_lock);

    kiro_destroy_rdma_memory (static_mem->internal->local_mem);
    g_free (static_mem->internal);
    g_free (static_mem);

    g_mutex_unlock (&priv->static_rdma_lock);
    return TRUE;
};


static gboolean
push_pull_static (KiroMessenger *self, KiroStaticRDMA* static_mem, gboolean push, GError **error_out)
{
    GError *error = NULL;

    g_return_val_if_fail (self != NULL, FALSE);

    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!static_mem->internal) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "KiroStaticRDMA struct corrupted ('internal' points to NULL). Doing nothing.");
        g_propagate_error (error_out, error);
        return FALSE;
    }

    KiroStaticRDMAInternal *internal = static_mem->internal;
    gulong peer_rank = internal->peer_rank;

    g_mutex_lock (&priv->connection_handling_lock);

    KiroPeer *peer = find_peer_by_rank (priv, peer_rank);
    if (!peer) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "No peer with rank '%lu' is connected.", peer_rank);
        g_propagate_error (error_out, error);
        g_mutex_unlock (&priv->connection_handling_lock);
        return FALSE;
    }

    if (!peer->active) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                     "Peer with rank '%lu' is no longer active.", peer_rank);
        g_propagate_error (error_out, error);
        g_mutex_unlock (&priv->connection_handling_lock);
        return FALSE;
    }

    g_mutex_lock (&peer->rdma_handling_lock);


    if (push) {
        if (rdma_post_read (peer->conn, peer->conn, internal->local_mem->mem, internal->local_mem->size, \
                            internal->local_mem->mr, 0, (uint64_t)internal->remote_mem.addr, internal->remote_mem.rkey)) {
            g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                         "Failed to post RDMA_READ: %s ", strerror (errno));
            goto fail;
        }
    }
    else {
        if (rdma_post_write (peer->conn, peer->conn, internal->local_mem->mem, internal->local_mem->size, \
                            internal->local_mem->mr, 0, (uint64_t)internal->remote_mem.addr, internal->remote_mem.rkey)) {
            g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                         "Failed to post RDMA_WRITE: %s ", strerror (errno));
            goto fail;
        }
    }

    struct ibv_wc send_wc;
    if (rdma_get_send_comp (peer->conn, &send_wc) < 0) {
        g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                "No send-completion for RDMA operation received: %s", strerror (errno));
        goto fail;
    }

    switch (send_wc.status) {
        case IBV_WC_SUCCESS:
            g_debug ("Message RDMA read successful");
            break;
        case IBV_WC_RETRY_EXC_ERR:
            g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                         "Peer '%lu' no longer responding", peer->rank);
            goto fail;
            break;
        case IBV_WC_REM_ACCESS_ERR:
            g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                         "Peer '%lu' has revoked access right to read data", peer->rank);
            goto fail;
            break;
        default:
            g_set_error (&error, KIRO_MESSENGER_ERROR, KIRO_MESSENGER_ERROR,
                         "Could not read message data from peer '%lu'. Status %u", peer->rank, send_wc.status);
            goto fail;
    }

    g_mutex_unlock (&peer->rdma_handling_lock);
    g_mutex_unlock (&priv->connection_handling_lock);
    return TRUE;

fail:
    g_mutex_unlock (&peer->rdma_handling_lock);
    g_mutex_unlock (&priv->connection_handling_lock);
    g_propagate_error (error_out, error);
    return FALSE;
};


gboolean kiro_messenger_push_static (KiroMessenger *self, KiroStaticRDMA* static_mem, GError **error_out)
{
    return push_pull_static (self, static_mem, TRUE, error_out);
};


gboolean kiro_messenger_pull_static (KiroMessenger *self, KiroStaticRDMA* static_mem, GError **error_out)
{
    return push_pull_static (self, static_mem, FALSE, error_out);
};


void
_foreach_peer_disconnect (gpointer peer_in, gpointer user_data)
{
    (void)user_data;
    KiroPeer *peer = (KiroPeer *)peer_in;
    destroy_peer (peer);
}


void
kiro_messenger_stop (KiroMessenger *self)
{
    g_return_if_fail (self != NULL);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!priv->peers && !priv->base) return;

    //Shut down event listening
    g_debug ("Stopping event handling...");
    priv->close_signal = TRUE;

    //This function will try to take the connection_handling_lock
    //so call this before we take the lock ourselves.
    kiro_messenger_stop_listen (self, NULL);

    g_mutex_lock (&priv->connection_handling_lock);
    g_mutex_lock (&priv->shutdown_lock);

    g_debug ("Disconnecting peers");
    if (priv->peers) {
        g_list_foreach (priv->peers, _foreach_peer_disconnect, NULL);
    }
    priv->peers = NULL;
    priv->rank_counter = 0;
    priv->static_counter = 0;

    priv->close_signal = FALSE;

    g_mutex_unlock (&priv->connection_handling_lock);
    g_mutex_unlock (&priv->shutdown_lock);

    g_debug ("Messenger stopped successfully");
}

