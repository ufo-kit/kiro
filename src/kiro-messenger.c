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

    // These are used in passive instance of messenger - The instance that listens for handshake
    guint                       rb_total_size;      // Total size of the ring buffer
    struct kiro_rdma_mem        *self_rb_rdma_mem;  // Pointer to RDMA accessible memory (ring buffer). This is assigned during KIRO_REQ_RDMA by the peer
    gboolean                    ring_buffer_ready;  // Indicates if the ring buffer is created and ready to poll
    void                        *rb_poll_ptr;       // Points to the head of the ring buffer
    void                        *rb_start_ptr;      // Starting address of the ring buffer. Is used when a new client is connected (after all existing clients are disconnected)
                                                    // So as to reuse an already created ring buffer
    struct kiro_rdma_mem        *self_hd_rdma_mem;  // Pinned down location of head descriptor
    short                       expecting_msg_id;   // Index starts from 1 as client tracks msg_id with index starting from 1. Is reset after every connection teardown

    // These are used in active instance of messenger - The instance that begins handshake
    unsigned int                processed_message_id;
    void                        *peer_hd_addr;      // Addr of the peer's head descriptor (of its ring buffer)
    uint32_t                    peer_hd_rkey;       // Remote key for the head descriptor
    void                        *peer_rb_head;      // Location where the peer's RB is *probably* polling. Is updated frequently
    void                        *peer_rb_tail;      // Location where client is going to write data to the RB
    uint32_t                    peer_rb_rkey;       // Remote key for the ring buffer
    void                        *peer_rb_start;     // Start of the buffer
    void                        *peer_rb_end;       // End of the buffer
    struct kiro_rdma_rb_status  *rb_status;         // A local memory location..Where remote peer's kiro_rdma_rb_status (head) is read into
    struct ibv_mr               *rb_status_mr;      // IBV MR for above
    gboolean                    head_descriptor_ready;  // Is set after receiving ACK_RDMA. Used to begin polling on rb_status
    gboolean                    peer_rb_requested;      // Flag that is set after requesting RB from peer

    guint32                     msg_id;          // Used to hold and generate message IDs
    struct pending_message      *message;        // Keep all outstanding RDMA message MRs

    GHookList                   rec_callbacks;   // List of all receive callbacks
    GHookList                   send_callbacks;  // List of all send callbacks

    GMutex                      connection_handling;
    GMutex                      rdma_handling;
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
    gboolean last_message;
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
    sleep(1); // To allow RDMA write to complete
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
    g_mutex_init (&priv->connection_handling);
    g_mutex_init (&priv->rdma_handling);
}


static void
kiro_messenger_finalize (GObject *object)
{
    g_return_if_fail (object != NULL);
    KiroMessenger *self = KIRO_MESSENGER (object);
    //Clean up the server
    kiro_messenger_stop (self);

    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);
    g_mutex_clear (&priv->connection_handling);
    g_mutex_clear (&priv->rdma_handling);

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


G_LOCK_DEFINE (close_lock);


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
send_msg (struct rdma_cm_id *id, struct kiro_rdma_mem *r, uint32_t imm_data)
{
    gboolean retval = TRUE;
    g_debug ("Sending message");

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

    if (ibv_post_send (id->qp, &wr, &bad)) {
        retval = FALSE;
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

gboolean
wait_for_rdma_write_completion(KiroMessengerPrivate *priv, int increment_tail_bytes, gboolean update_tail)
{
  struct ibv_wc wc;
  gboolean ret_val = FALSE;
  struct rdma_cm_id *conn = NULL;

  if (priv->type == KIRO_MESSENGER_SERVER)
      conn = priv->client;
  else
      conn = priv->conn;

  if (rdma_get_send_comp (conn, &wc) < 0) {
      g_critical ("No send completion for RDMA_WRITE received: %s", strerror (errno));
      return FALSE;
  }

  switch (wc.status) {
      case IBV_WC_SUCCESS:
          //g_debug ("RDMA transfer was successfull");
          priv->message->msg->status = KIRO_MESSAGE_SEND_SUCCESS;
          if(update_tail) // Should only update tail pointer when true
          {
            priv->peer_rb_tail += increment_tail_bytes; // We are using a void pointer, size in no.of bytes are incremented
            //g_debug("Ringbuffer updated..Tail pointing at : %p",priv->peer_rb_tail);
          }
          ret_val = TRUE;
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

  return ret_val;
}

void
initiate_read_peer_rb_head(KiroMessengerPrivate *priv)
{
  struct rdma_cm_id *conn = NULL;
  if (priv->type == KIRO_MESSENGER_SERVER)
      conn = priv->client;
  else
      conn = priv->conn;

  if(rdma_post_read (conn, conn->context, (void *)priv->rb_status, sizeof(struct kiro_rdma_rb_status), priv->rb_status_mr, 0, \
                    (uint64_t)priv->peer_hd_addr, priv->peer_hd_rkey)) {
        g_critical("Failed to read head status from remote %s", strerror (errno));
  }
}

gboolean
proceed_with_write(KiroMessengerPrivate *priv)
{
  gboolean ret_val = FALSE;
  if((long unsigned int)(priv->peer_rb_end - priv->peer_rb_tail) > (long unsigned int)(sizeof(struct kiro_rdma_meta_info)+priv->message->rdma_mem->size))
  {
    ret_val = TRUE;
  }
  else
  {
    // There is no more free space for meta+msg at the end region of the buffer. Below checks if peer's head has progressed
    if((long unsigned int)(priv->peer_rb_head - priv->peer_rb_start) > (long unsigned int)(sizeof(struct kiro_rdma_meta_info)+priv->message->rdma_mem->size))
    {
      // Below if check's if there is enough space for a reset_flag of size unsigned char is available
      if((long unsigned int)(priv->peer_rb_end - priv->peer_rb_tail) > (long unsigned int)(sizeof(unsigned char)))
      {
        // Beginning area of the RB is free, so write a reset_flag conveying that the next message after this reset_flag will be placed at the start location of RB
        struct rdma_cm_id *conn = NULL;
        if (priv->type == KIRO_MESSENGER_SERVER)
            conn = priv->client;
        else
            conn = priv->conn;

        unsigned char *reset_flag = (unsigned char *)g_malloc0(sizeof(unsigned char));
        *reset_flag = 77;
        struct ibv_mr *reset_flag_mr;
        kiro_register_rdma_memory(conn->pd, &reset_flag_mr, reset_flag, sizeof(unsigned char),IBV_ACCESS_LOCAL_WRITE); // Registerting MR for the reset flag

        g_debug("Writing RB reset_flag\n");
        if (rdma_post_write (conn, conn->context, (void *)reset_flag, sizeof(unsigned char), reset_flag_mr, 0, \
                            (uint64_t)priv->peer_rb_tail, priv->peer_rb_rkey)) {
            g_critical ("Failed to RDMA_WRITE reset_flag to peer: %s", strerror (errno));
            return FALSE;
        }
        wait_for_rdma_write_completion(priv, sizeof(unsigned char), FALSE);

        // Manually resetting tail pointer on active instance to the beginning of the peer's RB
        priv->peer_rb_tail = priv->peer_rb_start;
        ret_val = FALSE;
      }
      else
      {
        // This is a special case where there is no space left for writing a start_flag :/ !
        g_critical("Unhandled special case where there is no memory left for unsigned char :/");
        ;
      }
    }
  }
  return ret_val;
}

gboolean
rdma_write_message(KiroMessengerPrivate *priv)
{
  struct rdma_cm_id *conn = NULL;
  if (priv->type == KIRO_MESSENGER_SERVER)
      conn = priv->client;
  else
      conn = priv->conn;

  while( !proceed_with_write(priv)) {};

  struct kiro_rdma_meta_info *meta_info = (struct kiro_rdma_meta_info *)g_malloc0(sizeof(struct kiro_rdma_meta_info));
  meta_info->start_flag = 42;  // Indicates there is an incoming rdma message for the polling mechanism, also answer to life, the universe and everything
  if(priv->message->last_message)
    meta_info->last_message = TRUE;  // Passive instance resets and deregisters its RB after processing a payload with this flag
  else
    meta_info->last_message = FALSE;

  meta_info->rdma_done = FALSE; // This will be set to true once the actual message is transferred
  meta_info->followup_msg_size = priv->message->rdma_mem->size;
  meta_info->next_message = priv->peer_rb_tail + sizeof(struct kiro_rdma_meta_info) + priv->message->rdma_mem->size;
  meta_info->message_id = priv->msg_id;

  struct ibv_mr *meta_info_mr;

  void *meta_info_pointer_peer = priv->peer_rb_tail;

  kiro_register_rdma_memory(conn->pd, &meta_info_mr, meta_info, sizeof(struct kiro_rdma_meta_info),IBV_ACCESS_LOCAL_WRITE); // Registerting MR for the meta information

  g_debug("Writing meta_info of rdma message at %p", priv->peer_rb_tail);
  if (rdma_post_write (conn, conn->context, (void *)meta_info, sizeof(struct kiro_rdma_meta_info), meta_info_mr, 0, \
                      (uint64_t)priv->peer_rb_tail, priv->peer_rb_rkey)) {
      g_critical ("Failed to RDMA_WRITE to peer: %s", strerror (errno));
      return FALSE;
  }
  wait_for_rdma_write_completion(priv, sizeof(struct kiro_rdma_meta_info), TRUE);

  g_debug("Writing rdma message at %p", priv->peer_rb_tail);
  if (rdma_post_write (conn, conn->context, priv->message->rdma_mem->mem, priv->message->rdma_mem->size, priv->message->rdma_mem->mr, 0, \
                      (uint64_t)priv->peer_rb_tail, priv->peer_rb_rkey)) {
      g_critical ("Failed to RDMA_WRITE to peer: %s", strerror (errno));
      return FALSE;
  }
  wait_for_rdma_write_completion(priv, priv->message->rdma_mem->size, TRUE);

  g_debug("Setting rdma done flag in meta_info to true");
  // The tail pointer to the peer's ring buffer has already been updated
  meta_info->rdma_done = TRUE; // Setting this true and then writing it in the peer ring buffer
  if (rdma_post_write (conn, conn->context, (void *)meta_info, sizeof(struct kiro_rdma_meta_info), meta_info_mr, 0, \
                      (uint64_t)meta_info_pointer_peer, priv->peer_rb_rkey)) {
      g_critical ("Failed to RDMA_WRITE to peer: %s", strerror (errno));
      return FALSE;
  }
  wait_for_rdma_write_completion(priv, sizeof(struct kiro_rdma_meta_info), FALSE);  // Donot update tail pointer at this RDMA write, Parameter 2 is don't care with FALSE flag

  // After every write we request for a RDMA read
  initiate_read_peer_rb_head(priv);

  return TRUE;
}


static gboolean
process_rdma_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source'
    // Tell the compiler to ignore it by (void)-ing it
    (void) source;
    KiroMessengerPrivate *priv = (KiroMessengerPrivate *)data;

    if (!g_mutex_trylock (&priv->rdma_handling)) {
        g_debug ("RDMA handling will wait for the next dispatch.");
        return TRUE;
    }

    g_debug ("Got message on condition: %i", condition);
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
            struct kiro_ctrl_msg *msg_out = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
            msg_out->msg_type = KIRO_PONG;

            if (!send_msg (conn, ctx->cf_mr_send, 0)) {
                g_warning ("Failure while trying to post PONG send: %s", strerror (errno));
                goto done;
            }
            break;
        }
        case KIRO_MSG_STUB:
        {
            g_debug ("Got a stub message from the peer.");
            struct kiro_ctrl_msg *reply = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
            reply->msg_type = KIRO_REJ_RDMA;

            struct KiroMessage *msg_out = NULL;
            if (!priv->rec_callbacks.hooks) {
                g_debug ("But noone if listening for any messages");
            }
            else if (priv->message) {
                g_debug ("But we are currently waiting for something else");
            }
            else {
                msg_out = g_malloc0 (sizeof (struct KiroMessage));
                if (msg_out) {
                    msg_out->payload = NULL;
                    msg_out->size = 0;
                    msg_out->msg = ntohl (wc.imm_data);
                    msg_out->status = KIRO_MESSAGE_RECEIVED;

                    g_debug ("Sending ACK message");
                    reply->msg_type = KIRO_ACK_MSG;
                    reply->peer_mri.handle = msg_in->peer_mri.handle;
                }
            }

            if (0 > send_msg (conn, ctx->cf_mr_send, 0)) {
                g_warning ("Failure while trying to send ACK: %s", strerror (errno));
                if (msg_out)
                    g_free (msg_out);
                goto done;
            }

            if (msg_out) {
                g_hook_list_marshal_check (&(priv->rec_callbacks), FALSE, invoke_callbacks, msg_out);
                // Stub messages can always be cleaned up
                g_free (msg_out);
            }
            break;
      }
        case KIRO_ACK_MSG:
        {
            g_debug ("Got ACK for message '%u' from peer", msg_in->peer_mri.handle);
            if (priv->message->handle != msg_in->peer_mri.handle) {
                g_debug ("Reply is for the wrong message...");
                //
                //TODO: Cancel the current message transfer? Or do nothing?
                //
            }
            else {
                priv->message->msg->status = KIRO_MESSAGE_SEND_SUCCESS;
            }

            g_debug ("Cleaning up pending message ...");
            priv->message->rdma_mem = NULL; // MSG was a stub. There is no rdma_mem anyways
            g_hook_list_marshal_check (&(priv->send_callbacks), FALSE, invoke_callbacks, priv->message->msg);
            if (priv->message->message_is_mine && !priv->message->msg->message_handled) {
                g_debug ("Message is owned by the messenger and noone wants to handle it. Cleaning it up...");
                g_free (priv->message->msg);
            }
            g_free (priv->message);
            priv->message = NULL;
            break;
        }
        case KIRO_REQ_RDMA:
        {
            // Active instance requests for RDMA. We send our head pointer of ring buffer
            g_debug ("Peer wants to send a message of size %lu", msg_in->peer_mri.length);
            struct kiro_rdma_mem *rb_rdma_mem = NULL;
            struct kiro_rdma_mem *hd_rdma_mem = NULL;
            struct kiro_ctrl_msg *msg_out = (struct kiro_ctrl_msg *) (ctx->cf_mr_send->mem);
            msg_out->msg_type = KIRO_REJ_RDMA; // REJ by default. Only change if everyhing is okay

            if (priv->message) {
                g_debug ("But only one pending message is allowed");
            }
            else if (!priv->rec_callbacks.hooks) {
                g_debug ("But no one is listening for any messages");
            }
            else {

                // If there is already a ring buffer allocated. Use it, otherwise create one
                if(NULL == priv->rb_start_ptr)
                {
                    g_debug("Churning out a chunk of memory for peer");
                    priv->rb_total_size = msg_in->peer_mri.length*12; // arbitrarily set to X times the requested size
                    // This is the actual ring buffer memory allocation we do when an active instance requests an RDMA for the first time.
                    rb_rdma_mem = kiro_create_rdma_memory (conn->pd, priv->rb_total_size, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);
                    memset(rb_rdma_mem->mem,0,priv->rb_total_size);

                    priv->self_rb_rdma_mem = rb_rdma_mem;  // Our own copy of the newly create kiro_rdma_mem
                    priv->rb_poll_ptr = rb_rdma_mem->mem;  // Our own copy of the head. We use this to begin polling at this location of the ring buffer (mem element of structure)
                    priv->rb_start_ptr = rb_rdma_mem->mem; // Our own copy of the starting address of the ring buffer

                    /*
                    following creates a memory region where passive instance' head is available for active side to remotely read from passive side
                    This memory location is passed to active side along with ring buffer
                    */
                    g_debug("Creating a mem for active instance to read RB head");
                    hd_rdma_mem = kiro_create_rdma_memory (conn->pd, sizeof(struct kiro_rdma_rb_status), IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);
                    memset(hd_rdma_mem->mem,0,sizeof(struct kiro_rdma_rb_status));

                    // hd abbreviates is head descriptor
                    priv->self_hd_rdma_mem = hd_rdma_mem;
                }
                else
                {
                    g_debug("3R. Reusing existing ring buffer");
                    // The 3rd param is non null so a new region will not be allocated
                    kiro_register_rdma_memory(conn->pd, &priv->self_rb_rdma_mem->mr, priv->self_rb_rdma_mem->mem, priv->rb_total_size, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);
                    kiro_register_rdma_memory(conn->pd, &priv->self_hd_rdma_mem->mr, priv->self_hd_rdma_mem->mem, sizeof(struct kiro_rdma_rb_status), IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);
                    rb_rdma_mem = priv->self_rb_rdma_mem;
                    hd_rdma_mem = priv->self_hd_rdma_mem;
                }

                g_debug("RB is x%d the requested message size", (int)(priv->rb_total_size/msg_in->peer_mri.length));

                if (!rb_rdma_mem || !hd_rdma_mem) {
                    g_critical ("Failed to give RB HD info for peer !");
                }
                else {
                    g_debug ("Sending RB head ptr %p to peer", rb_rdma_mem->mem);
                    g_debug ("RB Ends at %p", rb_rdma_mem->mem+priv->rb_total_size);
                    g_debug ("Sending HD ptr %p to peer", hd_rdma_mem->mem);
                    msg_out->msg_type = KIRO_ACK_RDMA;
                    msg_out->peer_mri = *rb_rdma_mem->mr;
                    msg_out->peer_mri.handle = msg_in->peer_mri.handle;
                    msg_out->pin_hd = *hd_rdma_mem->mr;
                }
            }

            if (0 > send_msg (conn, ctx->cf_mr_send, 0)) {
                g_critical ("Failed to send RDMA credentials(ring buffer head) to peer!");
                if (rb_rdma_mem) {
                    // If we reach this point, we definitely have a pending
                    // message. Clean it up!
                    kiro_destroy_rdma_memory (rb_rdma_mem);
                    kiro_destroy_rdma_memory (hd_rdma_mem);
                    g_free (priv->message->msg);
                    g_free (priv->message);
                    priv->message = NULL;
                }
            }
            g_debug("Ready for new rdma message. Polling at %p",priv->rb_poll_ptr);
            priv->ring_buffer_ready = TRUE;
            priv->expecting_msg_id = 1;
            //g_debug ("RDMA message reply sent to peer");
            goto end_rmda_eh;
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
                if (priv->message->rdma_mem) {
                    priv->message->rdma_mem->mem = NULL; // mem points to the original message data! DON'T FREE IT JUST YET!
                    kiro_destroy_rdma_memory (priv->message->rdma_mem);
                }
                priv->message->msg->status = KIRO_MESSAGE_SEND_FAILED;
                g_hook_list_marshal_check (&(priv->send_callbacks), FALSE, invoke_callbacks, priv->message->msg);
                if (priv->message->message_is_mine && !priv->message->msg->message_handled) {
                    g_debug ("Message is owned by the messenger and noone wants to handle it. Cleaning it up...");
                    if (priv->message->msg->payload)
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
            // @TODO SASI. we no longer require to maintain which message we got RDMA ack as we will be requesting RDMA only once (for the first time).
            g_debug ("Received ring buffer head for RDMA Request");
            if (priv->message->handle != msg_in->peer_mri.handle) {
                g_debug ("Reply is for the wrong message...");
                //
                //TODO: Cancel the current message transfer? Or do nothing?
                //
                goto done;
            }
            else {
                priv->peer_rb_head = msg_in->peer_mri.addr;  // Peer head is at the beginning of the buffer during the first message
                g_debug("Peers ring buffer head is at : %p",priv->peer_rb_head);
                priv->peer_rb_tail = msg_in->peer_mri.addr;
                priv->peer_rb_rkey = msg_in->peer_mri.rkey;
                priv->peer_rb_start = msg_in->peer_mri.addr;
                priv->peer_rb_end = msg_in->peer_mri.addr + msg_in->peer_mri.length; // + Offset in bytes

                priv->peer_hd_addr = msg_in->pin_hd.addr;
                priv->peer_hd_rkey = msg_in->pin_hd.rkey;
                g_debug("Peers head desc head is at : %p",priv->peer_hd_addr);

                g_debug("Length of peer memory region is x%d the requested size", (int)(msg_in->peer_mri.length/priv->message->rdma_mem->size));

                priv->rb_status = (struct kiro_rdma_rb_status *)g_malloc0(sizeof(struct kiro_rdma_rb_status));
                kiro_register_rdma_memory(conn->pd, &priv->rb_status_mr, priv->rb_status, sizeof(struct kiro_rdma_rb_status),IBV_ACCESS_LOCAL_WRITE); // Registering MR for read head
                priv->head_descriptor_ready = TRUE;

                g_debug("Writing message %d",priv->msg_id);
                if( FALSE == rdma_write_message(priv))
                  goto cleanup; // Doesnot matter, fallthrough happens nevertheless
            }

            // This cleans up local pending message in active and releases transmit lock in the calling application
            cleanup:
                g_debug ("Cleaning up pending message (1st)...");
                priv->message->rdma_mem->mem = NULL; // mem points to the original message data! DON'T FREE IT JUST YET!
                kiro_destroy_rdma_memory (priv->message->rdma_mem);
                g_hook_list_marshal_check (&(priv->send_callbacks), FALSE, invoke_callbacks, priv->message->msg);
                if (priv->message->message_is_mine && !priv->message->msg->message_handled) {
                    g_debug ("Message is owned by the messenger and no one wants to handle it. Cleaning it up...");
                    g_free (priv->message->msg->payload);
                    g_free (priv->message->msg);
                }
                g_free (priv->message);
                priv->message = NULL;
                //
                //TODO: Inform the peer about failed send?
                //
                goto end_rmda_eh;
                break; //case KIRO_ACK_RDMA:
        }
        case KIRO_RDMA_DONE:
        {
            // @TODO SASI. This case should be implemented in the cb handler that polls ring buffer for data, so that the server doesnot further reject RDMA requests
            g_debug ("Peer has signalled message transfer success");
            priv->message->msg->status = KIRO_MESSAGE_RECEIVED;
            g_hook_list_marshal_check (&(priv->rec_callbacks), FALSE, invoke_callbacks, priv->message->msg);
            if (priv->message->msg->message_handled != TRUE) {
                g_debug ("Noone cared for the message. Received data will be freed.");
            }
            else {
                priv->message->rdma_mem->mem = NULL;
            }

            // -- FALL THROUGH INTENTIONAL -- //
        }
        case KIRO_RDMA_CANCEL:
        {
            g_debug ("Cleaning up pending message ...");
            if (priv->message) {
                kiro_destroy_rdma_memory (priv->message->rdma_mem);
                g_free (priv->message);
            }
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
    g_mutex_unlock (&priv->rdma_handling);
    return TRUE;
}


static gboolean
process_cm_event (GIOChannel *source, GIOCondition condition, gpointer data)
{
    // Right now, we don't need 'source' and 'condition'
    // Tell the compiler to ignore them by (void)-ing them
    (void) source;
    (void) condition;
    KiroMessengerPrivate *priv = (KiroMessengerPrivate *)data;

    g_debug ("CM event handler triggered");
    if (!g_mutex_trylock (&priv->connection_handling)) {
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

    g_mutex_unlock (&priv->connection_handling);
    g_debug ("CM event handling done");
    return TRUE;
}


gpointer
start_messenger_main_loop (gpointer data)
{
    g_main_loop_run ((GMainLoop *)data);
    return NULL;
}


/**
  The purpose of this is three fold:
  1: Check if there was any close_signal set
  2: Poll ring buffer at head for arrival of new data
  3: Poll head descriptor / kiro_rdma_rb_status for changes in address of RB pointed by the peer
**/
gboolean
idle_handler_of_main_loop (KiroMessengerPrivate *priv)
{
    void *temp_poll_ptr;
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
    if(priv->ring_buffer_ready)
    {
      if(*(unsigned char *)priv->rb_poll_ptr == 42)
      {
        // A new meta_info struct was written to the RB
        struct kiro_rdma_meta_info *meta_info = (struct kiro_rdma_meta_info *)priv->rb_poll_ptr;
        if(meta_info->rdma_done)
        {
          temp_poll_ptr = priv->rb_poll_ptr;
          if(meta_info->message_id == priv->expecting_msg_id)
          {
            g_debug("Payload available, processing message %d", priv->expecting_msg_id);
            priv->expecting_msg_id++;
          }
          else if(meta_info->message_id < priv->expecting_msg_id)
          {
            // Processing of ring buffer is faster than incoming message rate.
            // So return idle handler & continue to poll until we receive the message we expect
            return TRUE;
          }
          else
          {
            // This block is executed if messages in RB are not in order that is expected
            g_critical("RB: %d  EX: %d",meta_info->message_id, priv->expecting_msg_id);
            g_critical("Lost track of messages in ring buffer");
            // Prevent processing of unknown message by returning idle handler immediately
            return FALSE;
          }

          void *next_msg_ptr;
          struct pending_message *pm = (struct pending_message *)g_malloc0(sizeof (struct pending_message));
          pm->direction = KIRO_MESSAGE_RECEIVE;
          //pm->handle = msg_in->peer_mri.handle;
          pm->msg = (struct KiroMessage *)g_malloc0 (sizeof (struct KiroMessage));
          pm->msg->status = KIRO_MESSAGE_PENDING;
          pm->msg->id = meta_info->message_id;
          //pm->msg->msg = ntohl (wc.imm_data); //is in network byte order
          pm->msg->size = meta_info->followup_msg_size;
          pm->msg->payload = priv->rb_poll_ptr+sizeof(struct kiro_rdma_meta_info);  // Offset from meta_info memory location
          pm->msg->message_handled = FALSE;
          // pm->rdma_mem = rb_rdma_mem;
          priv->message = pm;

          next_msg_ptr = meta_info->next_message;

          /*
            The following was crutial part of KIRO genericMessenger.
            But, in enhancedMessenger this shouldn't be done because test application tries to free allocated RDMA memory (payload) which will result in seg fault as this memory is still a part of RB.
            And the ringbuffer once created, is supposed to be alive as long as the passive side is alive.
            Solution: We zero this message' part of ring buffer (meta_info and actual message) and then increment out poll pointer and inform peer about the updated head pointer.

            priv->message->msg->status = KIRO_MESSAGE_RECEIVED;
          */

          g_hook_list_marshal_check (&(priv->rec_callbacks), FALSE, invoke_callbacks, priv->message->msg);

          if(meta_info->last_message)
          {
            g_debug("Last message sent by the active instance");
            // This is the last message sent by the active side, since we have processed it we maually reset our RB for a new handshake
            // De-registers memory region associated to the allocated ring buffer
            ibv_dereg_mr(priv->self_rb_rdma_mem->mr);
            priv->rb_poll_ptr = priv->rb_start_ptr;
            priv->ring_buffer_ready = FALSE;
            priv->expecting_msg_id = 1;
          }
          else
          {
            priv->rb_poll_ptr = next_msg_ptr;
            g_debug("Ready for next rdma message. Polling at %p", priv->rb_poll_ptr);
          }
          struct kiro_rdma_rb_status *hd_mem = (struct kiro_rdma_rb_status *)priv->self_hd_rdma_mem->mem;
          hd_mem->processed_id = 1729;
          hd_mem->head = priv->rb_poll_ptr;

          // Cleaning up received message
          if ( TRUE != priv->message->msg->message_handled) {
              g_debug ("No one registered callbacks for this message. Received data will be freed.");
          }
          if (priv->message) {
              g_free (priv->message);
              // memset memory region of meta_info and the message with zero
              memset(temp_poll_ptr, 0, (size_t)sizeof(struct kiro_rdma_meta_info)+meta_info->followup_msg_size);
          }
          priv->message = NULL;

        }
      }
      else if (*(unsigned char *)priv->rb_poll_ptr == 77)
      {
        temp_poll_ptr = priv->rb_poll_ptr;
        // Ring buffer is wrapped back to the beginning
        g_debug("Peer notified that it has begun to put msgs at the beginning of RB");
        priv->rb_poll_ptr = priv->rb_start_ptr;
        g_debug("Resetting poll ptr. Polling at %p", priv->rb_poll_ptr);

        // Notify peer that we processed messages and reset out poll pointer to the start of ring buffer. Used to synchronize both instances
        struct kiro_rdma_rb_status *hd_mem = (struct kiro_rdma_rb_status *)priv->self_hd_rdma_mem->mem;
        hd_mem->processed_id = 1729;
        hd_mem->head = priv->rb_poll_ptr;

        memset(temp_poll_ptr, 0, (size_t)sizeof(unsigned char));
      }
    }
    if(priv->head_descriptor_ready)
    {
      /** In an ideal scenario...
          1. The client/active instance writes a message to the passive/server ring buffer
          2. The server/listener/passive processes incoming message and copies its new head pointer to the head descriptor (an memory that is remotely accessible)
          3. Simultaneously, the client/active instance of messenger initiates an rdma_read to copy it to a local memory
          4. We poll this local memory for changes.
          Note that this program flow is probably unpredictable because we are not using any synchronizations provided by IB such as completions
      **/
      // peer_rb_head is first set in ACK_RDMA.
      if(priv->rb_status->head != priv->peer_rb_head && priv->rb_status->head != NULL)
      {
        g_debug("RB Head status changed to %p",priv->peer_rb_head);
        priv->peer_rb_head = priv->rb_status->head;
        initiate_read_peer_rb_head(priv);
      }
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

    g_mutex_lock (&priv->connection_handling);
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

        g_debug ("Server bound to address %s:%s", addr_local, port);
        g_debug ("Enpoint listening");

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
        g_debug ("Connection to %s:%s established", address, port);
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
    g_idle_add ((GSourceFunc)idle_handler_of_main_loop, priv);
    priv->conn_ec = g_io_channel_unix_new (priv->ec->fd);
    priv->conn_ec_id = g_io_add_watch (priv->conn_ec, G_IO_IN | G_IO_PRI, process_cm_event, (gpointer)priv);
    priv->main_thread = g_thread_new ("KIRO Messenger main loop", start_messenger_main_loop, priv->main_loop);
    // We gave control to the main_loop (with add_watch) and don't need our ref
    // any longer
    g_io_channel_unref (priv->conn_ec);

    g_mutex_unlock (&priv->connection_handling);
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
    g_mutex_unlock (&priv->connection_handling);
    return -1;
}


int
kiro_messenger_submit_message (KiroMessenger *self, struct KiroMessage *msg, gboolean take_ownership, gboolean last_message)
{
    g_return_val_if_fail (self != NULL, -1);
    KiroMessengerPrivate *priv = KIRO_MESSENGER_GET_PRIVATE (self);

    if (!priv->conn)
        return -1;

    g_mutex_lock (&priv->rdma_handling);
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

    // If NULL it implies that we did not request for RDMA yet
    if(priv->peer_rb_requested == FALSE)
    {
      g_debug("Requesting RDMA RB, HD");
      priv->peer_rb_requested = TRUE;
      struct pending_message *pm = (struct pending_message *)g_malloc0(sizeof (struct pending_message));
      if (!pm) {
          goto fail;
      }
      pm->direction = KIRO_MESSAGE_SEND;
      pm->message_is_mine = take_ownership;
      pm->msg = msg;
      pm->handle = priv->msg_id++;
      pm->last_message = last_message;
      priv->message = pm;

      struct kiro_ctrl_msg *req = (struct kiro_ctrl_msg *)ctx->cf_mr_send->mem;

      if (msg->size > 0) {
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
          pm->rdma_mem = rdma_out;

          req->msg_type = KIRO_REQ_RDMA;
          req->peer_mri.length = msg->size;
          req->peer_mri.handle = pm->handle;
      }
      else {
          // STUB message
          req->msg_type = KIRO_MSG_STUB;
          req->peer_mri.handle = pm->handle;
      }

      if (0 > send_msg (conn, ctx->cf_mr_send, msg->msg)) {
          //
          //TODO
          //
          goto fail;
      }
      g_mutex_unlock (&priv->rdma_handling);
      return 0;
    }
    else
    {
      // @TODO SASI. RDMA details are now available, so no longer request server for another RDMA and instead simple use tail pointer to put data into the servers ring buffer
      // We already have RDMA region details of the server. So we should continue using that
      // Tail is incremented in rdma_write_message. Checking if it is possible to write into ring buffer should also probably be there

      // @TODO SASI.First part is basically repeated from above. Merge it with previous implementation
      struct pending_message *pm = (struct pending_message *)g_malloc0(sizeof (struct pending_message));
      if (!pm) {
          goto fail;
      }
      pm->direction = KIRO_MESSAGE_SEND;
      pm->message_is_mine = take_ownership;
      pm->msg = msg;
      pm->handle = priv->msg_id++;
      pm->last_message = last_message;
      priv->message = pm;

      struct kiro_rdma_mem *rdma_out = (struct kiro_rdma_mem *)g_malloc0 (sizeof (struct kiro_rdma_mem));
      if (!rdma_out) {
          goto fail;
      }
      rdma_out->size = msg->size;
      rdma_out->mem = msg->payload;
      if (0 > kiro_register_rdma_memory (conn->pd, &(rdma_out->mr), msg->payload, msg->size, IBV_ACCESS_LOCAL_WRITE)) {
          goto fail;
      }
      pm->rdma_mem = rdma_out;

      g_debug("Writing message %d",priv->msg_id);
      if( FALSE == rdma_write_message(priv))
        goto fail;

      g_debug("Consecutive message was written to the ring buffer");
      g_mutex_unlock (&priv->rdma_handling);    // Technically this is not RDMA Handling
      g_hook_list_marshal_check (&(priv->send_callbacks), FALSE, invoke_callbacks, priv->message->msg);
      // Cleaning up pending message
      g_free (priv->message); // Freeing pending_message
      priv->message = NULL;

      return 0;
    }
  fail:
      g_mutex_unlock (&priv->rdma_handling);
      return -1;

}


gulong
kiro_messenger_add_receive_callback (KiroMessenger *self, KiroMessengerCallbackFunc func, void *user_data)
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
kiro_messenger_add_send_callback (KiroMessenger *self, KiroMessengerCallbackFunc func, void *user_data)
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
    g_debug ("Messenger stopped successfully");
}
