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

typedef gboolean KiroContinueFlag;
#define KIRO_CALLBACK_CONTINUE TRUE
#define KIRO_CALLBACK_REMOVE FALSE

enum KiroMessengerType {
    KIRO_MESSENGER_SERVER = 0,
    KIRO_MESSENGER_CLIENT
};

enum KiroMessageStatus {
    KIRO_MESSAGE_PENDING = 0,
    KIRO_MESSAGE_SEND_SUCCESS,
    KIRO_MESSAGE_SEND_FAILED,
    KIRO_MESSAGE_RECEIVED
};


struct _KiroMessenger {

    GObject parent;

    /*< private >*/
    KiroMessengerPrivate *priv;
};


struct _KiroMessengerClass {

    GObjectClass parent_class;

};

struct KiroMessage {
    enum KiroMessageStatus status; // Status of the message
    guint64     id;           // Unique ID of the message. This may not be changed by the user
    guint32     msg;          // Space for application specific message semantics 
    guint64     size;         // Size of the messages payload in bytes
    gpointer    payload;      // Pointer to the payload of the message
    gboolean    message_handled; // FALSE initially, TRUE once the message was handled
};


/* GObject and GType functions */

/**
 * kiro_messenger_get_type: (skip)
 * Returns: GType of KiroMessenger
 */
GType        kiro_messenger_get_type            (void);

/**
 * kiro_messenger_new - Creates a new #KiroMessenger
 * Returns: (transfer full): A pointer to a new #KiroMessenger
 * Description:
 *   Creates a new, unbound #KiroMessenger and returns a pointer to it.
 * See also:
 *   kiro_messenger_start, kiro_messenger_free
 */
KiroMessenger*  kiro_messenger_new                (void);

/**
 * kiro_messenger_free - 'Destroys' the given #KiroMessenger
 * @messenger: The #KiroMessenger that is to be freed
 * Description:
 *   Transitions the #KiroMessenger through all necessary shutdown routines and
 *   frees the object memory.
 * See also:
 *   kiro_messenger_start, kiro_messenger_new
 */
void         kiro_messenger_free                (KiroMessenger *messenger);


/* messenger functions */

/**
 * kiro_messenger_start - Starts the messenger
 * @messenger: #KiroMessenger to perform the operation on
 * @bind_addr: Local address to bind the messenger to
 * @bind_port: Local port to listen for connections
 * @role: A #KiroMessengerType which is used to decide the role of this
 * messenger
 * Returns: An integer denoting success of this function. 0 for success, -1
 * otherwise
 * Description:
 *   Starts the #KiroMessenger with the given role. When @role is given as
 *   KIRO_MESSENGER_SERVER, the messenger will open an InfiniBand connection and
 *   listen for the first client that tries to connect. When given
 *   KIRO_MESSENGER_CLIENT, the messenger will instead try to connect to to the
 *   given address.
 * Notes:
 *   If the bind_addr is NULL, the messenger will bind to the first device
 *   it can find on the machine and listen across all IPs. Otherwise it
 *   will try to bind to the device associated with the given address.
 *   Address is given as a string of either a hostname or a dot-separated
 *   IPv4 address or a colon-separated IPv6 hex-address.
 *   If bind_port is NULL the messenger will choose a free port randomly
 *   and return the chosen port as return value.
 * See also:
 *   kiro_messenger_new,
 */
int kiro_messenger_start (KiroMessenger *messenger, const char *bind_addr, const char *bind_port, enum KiroMessengerType role);


/**
 * KiroReceiveCallbackFunc - Function type for sync callbacks of a
 * #KiroMessenger
 * @message: A pointer to the #KiroMessage that was received and/or sent
 * @user_data: (transfer none): The #user_data which was provided during
 * registration of this callback
 * Returns: A #KiroContinueFlag
 * Description:
 *   Defines the type of a callback function which will be invoked every time
 *   the messenger has sent received a message
 * Note:
 *   Returning %FALSE or %KIRO_CALLBACK_REMOVE will automatically remove the callback
 *   from the internal callback list. Return %TRUE or %KIRO_CALLBACK_CONTINUE if you
 *   want to keep the callback active.
 *   A receive callback function needs to set the "message_handled" flag correctly, if
 *   it has taken the messages content and set the "payload" pointer in the
 *   message struct to %NULL. Otherweise, the messages payload will be freed
 *   once all callbacks were invoked.
 *   In case of a send callback, the message payload will only be freed if the
 *   'take_ownership' flag in the 'kiro_messenger_submit_message' was set to
 *   %TRUE. If the flag was set to %FALSE, the "message_handled" flag is ignored
 *   by the callback mechanism and the message payload will never be freed.
 *   If the callback receives a message which already has the "message_handled"
 *   flag set, it can safely ignore the message and just return.
 *   A callback may under no circumstances free the given message construct.
 * See also:
 *   kiro_messenger_add_receive_callback, kiro_messenger_remove_receive_callback,
 *   kiro_messenger_add_send_callback, kiro_messenger_remove_send_callback
 */
typedef KiroContinueFlag (*KiroMessengerCallbackFunc)   (struct KiroMessage *message, void *user_data);

/**
 * kiro_messenger_add_receive_callback - Register a callback function for
 * receive events
 * @messenger: #KiroMessenger to perform this operation on
 * @callback: Pointer to a #KiroReceiveCallbackFunc that will be invoked when a
 * messege is received
 * @user_data: Pointer to user data that will be passed to the callback function
 * Returns: Integer ID of the registered callback.
 * Description:
 *   Registers the given callback function to be invoked every time the
 *   messenger receives a message.
 * See also:
 *   kiro_messenger_remove_receive_callback
 */
gulong kiro_messenger_add_receive_callback (KiroMessenger *messenger, KiroMessengerCallbackFunc callback, void *user_data);

/**
 * kiro_messenger_remove_receive_callback - Removes a previously registered
 * callback
 * @messenger: #KiroMessenger to perform this operation on
 * @callback_id: ID of the callback that should be removed
 * Returns: %TRUE on success %FALSE in case of error
 * Description:
 *   Removes the callback with the given ID.
 * See also:
 *   kiro_messenger_add_receive_callback
 */
gboolean kiro_messenger_remove_receive_callback (KiroMessenger *messenger, gulong callback_id);

/**
 * kiro_messenger_add_send_callback - Register a callback function for
 * send completions
 * @messenger: #KiroMessenger to perform this operation on
 * @callback: Pointer to a #KiroSendCallbackFunc that will be invoked when a
 * messege was sent
 * @user_data: Pointer to user data that will be passed to the callback function
 * Returns: Integer ID of the registered callback.
 * Description:
 *   Registers the given callback function to be invoked every time the
 *   messenger has sent a message.
 * Note:
 *   A pointer to the message that has been sent will be passed to all
 *   registered callbacks one by one. The message container and all of its data
 *   will not be feed after all callbacks have been invoked. The user
 *   implementation needs to take care of this.
 * See also:
 *   kiro_messenger_remove_send_callback
 */
gulong kiro_messenger_add_send_callback (KiroMessenger *messenger, KiroMessengerCallbackFunc callback, void *user_data);

/**
 * kiro_messenger_remove_send_callback - Removes a previously registered
 * callback
 * @messenger: #KiroMessenger to perform this operation on
 * @callback_id: ID of the callback that should be removed
 * Returns: %TRUE on success %FALSE in case of error
 * Description:
 *   Removes the callback with the given ID.
 * See also:
 *   kiro_messenger_send_receive_callback
 */
gboolean kiro_messenger_remove_send_callback (KiroMessenger *messenger, gulong callback_id);

/**
 * kiro_messenger_submit_message - send the given message to the remote side
 * @messenger: #KiroMessenger to use for sending the message
 * @message: Pointer to a #KiroMessage for sending
 * @take_ownership: Decide if the messenger will take ownership of the messag
 * data or the ownership should stay with the caller.
 * Returns: 0 on success, -1 in case of error
 * Description:
 *   Sends the given #KiroMessage to the remote side. The 'status' field of the
 *   #KiroMessage struct will be set to KIRO_MESSAGE_SEND_SUCCESS, once the
 *   message has been sent successfully, or to KIRO_MESSAGE_SEND_FAILED in case
 *   of an error.
 *   If the @take_ownership flag was set to %TRUE, the KiroMessenger will take
 *   ownership of the KiroMessage struct. It will then attempt to free it,
 *   including the message payload, after the message was sent, in accordance to
 *   the status of the "message_handled" flag of the created message, after all
 *   send callbacks have returned.  This mechanism can be used to realize a
 *   fire-and-forget mechanism where the user can be sure the message will be
 *   cleaned up automatically after it has been sent. If the @take_ownership is
 *   set to %FALSE, the KiroMessenger will not take ownership of the message and
 *   the message and its payload will not be freed after the callbacks have been
 *   invoked, regardles of the "message_handled" flag. The caller stays
 *   responsible to clean up the message sooner or later.
 * Note:
 *   After the message was sent to the remote side, all of the registeres send
 *   callbacks will be invoked, regardless if the message was sent successfully
 *   or not. The status field in the message struct can be checked to see if the
 *   message was sent successfully.
 */
int kiro_messenger_submit_message (KiroMessenger *messenger, struct KiroMessage *message, gboolean take_ownership);

/**
 * kiro_messenger_stop - Stops the messenger
 * @messenger: #KiroMessenger to perform the operation on
 * Description:
 *   Stops the given #KiroMessenger
 * See also:
 *   kiro_messenger_start
 */
void kiro_messenger_stop (KiroMessenger *messenger);

G_END_DECLS

#endif //__KIRO_MESSENGER_H
