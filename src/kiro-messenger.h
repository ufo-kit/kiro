/* Copyright (C) 2014-2015 Timo Dritschler <timo.dritschler@kit.edu>
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

#ifndef __KIRO_MESSENGER_H
#define __KIRO_MESSENGER_H

#include <stdint.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define KIRO_TYPE_MESSENGER             (kiro_messenger_get_type())
#define KIRO_MESSENGER(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj), KIRO_TYPE_MESSENGER, KiroMessenger))
#define KIRO_IS_MESSENGER(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj), KIRO_TYPE_MESSENGER))
#define KIRO_MESSENGER_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass), KIRO_TYPE_MESSENGER, KiroMessengerClass))
#define KIRO_IS_MESSENGER_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass), KIRO_TYPE_MESSENGER))
#define KIRO_MESSENGER_GET_CLASS(obj)   (G_TYPE_INSTANCE_GET_CLASS((obj), KIRO_TYPE_MESSENGER, KiroMessengerClass))


typedef struct _KiroMessenger           KiroMessenger;
typedef struct _KiroMessengerClass      KiroMessengerClass;
typedef struct _KiroMessengerPrivate    KiroMessengerPrivate;

enum {
    KIRO_MESSENGER_ERROR
} KiroMessengerError;

GQuark kiro_messenger_get_error_quark (void);
#define KIRO_MESSENGER_ERROR kiro_messenger_get_error_quark()

typedef gboolean KiroContinueFlag;
#define KIRO_CALLBACK_CONTINUE TRUE
#define KIRO_CALLBACK_REMOVE FALSE

typedef gboolean KiroCleanupFlag;
#define KIRO_CLEANUP_MESSAGE TRUE
#define KIRO_KEEP_MESSAGE FALSE

struct _KiroMessenger {

    GObject parent;

    /*< private >*/
    KiroMessengerPrivate *priv;
};


struct _KiroMessengerClass {

    GObjectClass parent_class;

};

typedef struct {
    guint32     msg;        // Space for application specific message semantics
    guint64     size;       // Size of the messages payload in bytes
    gpointer    payload;    // Pointer to the payload of the message
} KiroMessage;


typedef enum {
    KIRO_MESSAGE_PENDING = 0,
    KIRO_MESSAGE_SEND_SUCCESS,
    KIRO_MESSAGE_SEND_FAILED,
    KIRO_MESSAGE_REJ_WITH_PEER_BUSY,
    KIRO_MESSAGE_REJ_WITH_NOT_LISTENING,
    KIRO_MESSAGE_RECEIVED
} KiroMessageStatus;


//Forward declare
struct _KiroRequest;


/**
 * KiroMessageCallbackFunc: (skip)
 * @request: A pointer to the #KiroRequest for the processed message.
 * @user_data: (transfer none): The #user_data which was provided during
 * registration of this callback
 *
 *   Defines the type of a callback function used in the #KiroMessenger send and
 *   receive mechanisms.
 *
 * Note:
 *   The user might not call synchronous/blocking functions from within this
 *   callback, such as kiro_messenger_send_blocking. This will cause the messenger
 *   to lock up.
 * See also:
 *   kiro_messenger_receive, kiro_messenger_send, kiro_messenger_send_blocking
 */
typedef void (*KiroMessageCallbackFunc) (struct _KiroRequest *request, void *user_data);


typedef struct _KiroRequest {
    gulong                  id;         // Can be set by the user for easy identification
    KiroMessage             *message;
    gulong                  peer_rank;  // Rank/local-id of the destination or source
    KiroMessageCallbackFunc callback;
    gpointer                *user_data;
    KiroMessageStatus       status;
} KiroRequest;


/* GObject and GType functions */
GType        kiro_messenger_get_type            (void);


/**
 * kiro_messenger_new:
 *
 *   Creates a new, unbound #KiroMessenger and returns a pointer to it.
 *
 * Returns: (transfer full): A pointer to a new #KiroMessenger
 * See also:
 *   kiro_messenger_free
 */
KiroMessenger* kiro_messenger_new (void);


/**
 * kiro_messenger_free:
 * @messenger: The #KiroMessenger that is to be freed
 *
 *   Transitions the #KiroMessenger through all necessary shutdown routines and
 *   frees the object memory.
 *
 * See also:
 *   kiro_messenger_new
 */
void kiro_messenger_free (KiroMessenger *messenger);


/* messenger functions */

/**
 * KiroConnectCallbackFunc:
 * @connection_rank: The rank/local-id of the new connection
 * @user_data: (transfer none): The #user_data which was provided during
 * registration of this callback
 *
 *   Defines the type of a callback function which will be invoked each time
 *   a new peer connects to the messenger
 *
 * Returns: A #KiroContinueFlag. KIRO_CALLBACK_REMOVE will cause the
 * #KiroMessenger to stop listening and automatically deregister this callback.
 * See also:
 *   kiro_messenger_start_listen
 */
typedef KiroContinueFlag (*KiroConnectCallbackFunc) (gulong connection_rank, void *user_data);

/**
 * kiro_messenger_start_listen:
 * @messenger: #KiroMessenger to perform the operation on
 * @bind_addr: (allow-none): Local address to bind the messenger to
 * @bind_port: (allow-none): Local port to listen for connections
 * @connect_callback: (scope call): Callback to notify about new connection
 * @user_data: (transfer none): Data to pass to the @connect_callback on call
 * @error: (allow-none): A #GError, used for error reporting
 *
 *   The messenger will open an InfiniBand connection and listen for incoming
 *   client connections. The @connect_callback will be invoked each time a peer
 *   connects to this @messenger and will be passed the new rank/local-id of the
 *   connected peer.
 *
 * Returns: The port the messenger was bound to. -1 in case of error.
 * Notes:
 *   If the bind_addr is NULL, the messenger will bind to the first device
 *   it can find on the machine and listen across all IPs. Otherwise it
 *   will try to bind to the device associated with the given address.
 *   Address is given as a string of either a hostname or a dot-separated
 *   IPv4 address or a colon-separated IPv6 hex-address.
 *   If bind_port is NULL the messenger will choose a free port randomly
 *   and return the chosen port as return value.
 * See also:
 *   kiro_messenger_stop_listen, kiro_messenger_disconnect
 */
int kiro_messenger_start_listen (KiroMessenger *messenger,
                                 const char *bind_addr,
                                 const char *bind_port,
                                 KiroConnectCallbackFunc connect_callback,
                                 gpointer user_data,
                                 GError **error);

/**
 * kiro_messenger_stop_listen:
 * @messenger: #KiroMessenger to perform the operation on
 *
 *   Stops the given #KiroMessenger from listening so new incoming connections
 *   will be accepted. The callback which was registered upon the call to
 *   kiro_messenger_start_listen will automatically be deregisterd.
 *
 * Notes:
 *   If the given #KiroMessenger is not currently listening or %NULL, this
 *   function will simply return.
 * See also:
 *   kiro_messenger_start_listen
 */
void kiro_messenger_stop_listen (KiroMessenger *messenger, GError **error);

/**
 * kiro_messenger_connect:
 * @messenger: #KiroMessenger to perform the operation on
 * @remote_addr: Remote address to connect the messenger to
 * @remote_port: Remote port to connect the messenger to
 * @connection_rank (out): Storage for the rank/local-id of the new connection
 * @error: (allow-none): A #GError, used for error reporting
 *
 *   The messenger will open an InfiniBand connection and try to connect to the
 *   given @remote_addr and @remote_port. If successfull, the rank/local-id of
 *   the new connection will be stored in @connection_rank.
 *
 * See also:
 *   kiro_messenger_disconnect
 */
void kiro_messenger_connect (KiroMessenger *messenger,
                             const gchar *remote_addr,
                             const gchar *remote_port,
                             gulong *connection_rank,
                             GError **error);

/**
 * kiro_messenger_add_receive_callback:
 * @messenger: #KiroMessenger to perform this operation on
 * @request: A #KiroRequest for receiving a message
 *
 *   The given @request will be used to report the status of the receive
 *   request. If a callback was supplied in the request struct, that callback
 *   will be invoked upon completion of the receive request.
 *
 * See also:
 *   kiro_messenger_send_blocking
 */
gboolean kiro_messenger_receive (KiroMessenger *messenger, KiroRequest *request);


/**
 * kiro_messenger_add_send_callback:
 * @messenger: #KiroMessenger to perform this operation on
 * @request: A #KiroRequest for this message
 * @error: (allow-none): A #GError, used for error reporting
 *
 *   Tries to process the given @request struct. The @request will be used to
 *   report the status of the send request. If a callback was supplied in the
 *   request struct, that callback will be invoked upon completion of the
 *   send request.
 *
 * Returns: A #gboolean. %TRUE if the @request was submitted successfully,
 * %FALSE in case of an error. See the provided @error for details.
 *
 * Note:
 *   The @request container and all of its data will not be freed
 *   after the callbacks has been invoked. The user implementation needs to
 *   take care of this.
 * See also:
 *   kiro_messenger_send_blocking
 */
gboolean kiro_messenger_send (KiroMessenger *messenger,
                              KiroRequest *request, 
                              GError **error);

/**
 * kiro_messenger_submit_message:
 * @messenger: #KiroMessenger to use for sending the message
 * @message: The #KiroMessage to send
 * @peer_rank: Rank/local-ID of the peer to send the @message to
 * @error: (allow-none): A #GError, used for error reporting
 *
 *   Sends the given #KiroMessage to the remote peer.
 *
 * Returns: A #GBoolean. %TRUE when the @message was sent successfully, %FALSE
 * in case of a transmission failure. See the provided @error for details.
 *
 * See also:
 *   kiro_messenger_send_with_callback
 */
gboolean kiro_messenger_send_blocking (KiroMessenger *messenger,
                                       KiroMessage *message,
                                       gulong peer_rank,
                                       GError **error);

/**
 * kiro_messenger_stop:
 * @messenger: #KiroMessenger to perform the operation on
 *
 * Stops the given #KiroMessenger
 */
void kiro_messenger_stop (KiroMessenger *messenger);

G_END_DECLS

#endif //__KIRO_MESSENGER_H
