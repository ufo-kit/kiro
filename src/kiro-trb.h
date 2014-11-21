/* Copyright (C) 2014 Timo Dritschler <timo.dritschler@kit.edu>
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
 * SECTION: kiro-trb
 * @Short_description: KIRO 'Transmittable Ring Buffer'
 * @Title: KiroTrb
 *
 * KiroTrb implements a 'Transmittable Ring Buffer' that holds all necessary information
 * about its content inside itself, so its data can be exchanged between different
 * instances of the KiroTrb Class and/or sent over a network.
 */

#ifndef __KIRO_TRB_H
#define __KIRO_TBR_H

#include <stdint.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define KIRO_TYPE_TRB             (kiro_trb_get_type())
#define KIRO_TRB(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj), KIRO_TYPE_TRB, KiroTrb))
#define KIRO_IS_TRB(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj), KIRO_TYPE_TRB))
#define KIRO_TRB_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass), KIRO_TYPE_TRB, KiroTrbClass))
#define KIRO_IS_TRB_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass), KIRO_TYPE_TRB))
#define KIRO_TRB_GET_CLASS(obj)   (G_TYPE_INSTANCE_GET_CLASS((obj), KIRO_TYPE_TRB, KiroTrbClass))


typedef struct _KiroTrb           KiroTrb;
typedef struct _KiroTrbClass      KiroTrbClass;
typedef struct _KiroTrbPrivate    KiroTrbPrivate;


struct _KiroTrb {

    GObject parent;

};


/**
 * IbvConnectorInterface:
 *
 * Base interface for IbvConnectors.
 */

struct _KiroTrbClass {

    GObjectClass parent_class;

};


struct KiroTrbInfo {

    /* internal information about the buffer */
    uint64_t buffer_size_bytes;  // Size in bytes INCLUDING this header
    uint64_t element_size;       // Size in bytes of one single element
    uint64_t offset;             // Current Offset to access the 'oldest' element (in element count!)

} __attribute__ ((packed));


/* GObject and GType functions */
/**
 * kiro_trb_get_type: (skip)
 * Returns: GType of #KiroTrb
 */
GType       kiro_trb_get_type           (void);

/**
 * kiro_trb_new - Creates a new #KiroTrb
 * Returns: (transfer full): A pointer to a new #KiroTrb
 * Description:
 *   Creates a new, unshaped #KiroTrb and returns a pointer to it.
 * See also:
 *   kiro_trb_free, kiro_trb_reshape
 */
KiroTrb*    kiro_trb_new                (void);

/**
 * kiro_trb_free - 'Destroys' the given #KiroTrb
 * @trb: (transfer none): The #KiroTrb that is to be freed
 * Description:
 *   Clears all underlying memory and frees the object memory. 
 * Note:
 *   The internal memory is also freed when calling this function. If you want
 *   to continue using the raw @trb memory after call this function, you need to
 *   memcpy() its content using the information optained from
 *   kiro_trb_get_raw_buffer and kiro_trb_get_raw_size. 
 * See also:
 *   kiro_trb_new
 */
void        kiro_trb_free               (KiroTrb *trb);


/* trb functions */

/**
 * kiro_trb_get_element_size:
 * Returns the element size in bytes
 * @trb: #KiroTrb to perform the operation on
 * Description:
 *   Returns the size of the individual elements in the buffer
 * See also:
 *   kiro_trb_reshape, kiro_trb_adopt, kiro_trb_clone
 */
uint64_t kiro_trb_get_element_size (KiroTrb *trb);

/**
 * kiro_trb_get_max_elements:
 * Returns the capacity of the buffer
 * @trb: #KiroTrb to perform the operation on
 * Description:
 *   Returns the mximal number of elements that can be stored in
 *   the buffer
 * See also:
 *   kiro_trb_get_element_size, kiro_trb_reshape, kiro_trb_adopt,
 *   kiro_trb_clone
 */
uint64_t kiro_trb_get_max_elements (KiroTrb *trb);


/**
 * kiro_trb_get_raw_size:
 * Returns the size of the buffer memory
 * @trb: #KiroTrb to perform the operation on
 * Description:
 *   Returns the size of the buffers internal memory
 * Notes:
 *   The returned size is given INCLUDING the header on top of the
 *   buffers internal memory
 * See also:
 *   kiro_trb_reshape, kiro_trb_adopt,
 *   kiro_trb_clone
 */
uint64_t kiro_trb_get_raw_size (KiroTrb *trb);


/**
 * kiro_trb_get_raw_buffer:
 * @trb: #KiroTrb to perform the operation on
 * Description:
 *   Returns a pointer to the memory structure of the given buffer.
 * Returns: (transfer none): a pointer to the buffer memory
 * Notes:
 *   The returned pointer points to the beginning of the internal
 *   memory of the buffer, including all header information. The
 *   user is responsible to ensure the consistency of any data
 *   written to the memory and should call 'kiro_trb_refesh' in
 *   case any header information was changed.
 *   The pointed to memory might become invalid at any time by
 *   concurrent access to the TRB, reshaping, adopting or cloning
 *   a new memory block.
 *   Under no circumstances might the memory pointed to by the returned
 *   pointer be 'freed' by the user!
 *   If this function is called on a buffer that is not yet setup,
 *   a NULL pointer is returned instead.
 * See also:
 *   kiro_trb_refesh, kiro_trb_reshape, kiro_trb_adopt, kiro_trb_clone
 */
void* kiro_trb_get_raw_buffer (KiroTrb *trb);


/**
 * kiro_trb_get_element:
 * @trb: #KiroTrb to perform the operation on
 * @index: Index of the element in the buffer to access
 * Description:
 *   Returns a pointer to the element in the buffer at the given index.
 * Returns: (transfer none): a pointer to the element at the given index.
 * Notes:
 *   The returned pointer to the element is only guaranteed to be valid
 *   immediately after the function call. The user is responsible to
 *   ensure that no data is written to the returned memory. The
 *   element pointed to might become invalid at any time by any concurrent
 *   access to the buffer wraping around and overwriting the element or
 *   changing the buffer memory entirely.
 *   Under no circumstances might the memory pointed to by the returned
 *   pointer be 'freed' by the user!
 *   If this function is called on a buffer that is not yet setup,
 *   a NULL pointer is returned instead.
 * See also:
 *   kiro_trb_get_element_size, kiro_trb_get_raw_buffer
 */
void* kiro_trb_get_element (KiroTrb *trb, uint64_t index);


/**
 * kiro_trb_dma_push:
 * Gives DMA to the next element and pushes the buffer
 * @trb: #KiroTrb to perform the operation on
 * Description:
 *   Returns a pointer to the next element in the buffer and increases
 *   all internal counters and meta data as if an element was pushed
 *   onto the buffer.
 * Returns: (transfer none): Pointer to the bginning of element memory
 * Notes:
 *   The returned pointer to the element is only guaranteed to be valid
 *   immediately after the function call. The user is responsible to
 *   ensure that no more data is written than 'element_size'. The
 *   element pointed to might become invalid at any time by any concurrent
 *   access to the buffer wraping around and overwriting the element or
 *   changing the buffer memory entirely.
 *   Under no circumstances might the memory pointed to by the returned
 *   pointer be 'freed' by the user!
 *   If this function is called on a buffer that is not yet setup,
 *   a NULL pointer is returned instead.
 * See also:
 *   kiro_trb_push, kiro_trb_get_element_size, kiro_trb_get_raw_buffer
 */
void* kiro_trb_dma_push (KiroTrb *trb);


/**
 * kiro_trb_flush:
 * Flushes the buffer
 * @trb: #KiroTrb to perform the operation on
 * Description:
 *   Flushes the internal buffer so the buffer is 'empty' again.
 * Notes:
 *   The underlying memory is not cleared, freed or rewritten.
 *   Only the header is rewritten and the internal pointer and
 *   counter structures get reset to zero.
 * See also:
 *   kiro_trb_reshape, kiro_trb_adopt, kiro_trb_clone
 */
void kiro_trb_flush (KiroTrb *trb);


/**
 * kiro_trb_purge:
 * Completely resets the Buffer
 * @trb: #KiroTrb to perform the operation on
 * @free_memory: True = internal memory will be free()'d,
 *               False = internal memory will be 'orphaned'
 * Description:
 *   Resets all internal structures so the TRB becomes
 *   'uninitialized' again.
 * Notes:
 *   Depending on the 'free_memory' argument, any currently
 *   held internal memory either gets free()'d or is simply
 *   unreferenced and therfore 'orphaned'.
 * See also:
 *   kiro_trb_reshape, kiro_trb_adopt, kiro_trb_clone
 */
void kiro_trb_purge (KiroTrb *trb, gboolean free_memory);


/**
 * kiro_trb_is_setup:
 * Returns the setup status of the buffer
 * @trb: #KiroTrb to perform the operation on
 * Description:
 *   Returns an integer designating of the buffer is ready to
 *   be used or needs to be 'reshaped' before it can accept data
 * Notes:
 *   A return value of 0 designates that the buffer is not ready
 *   to be used. Values greater than 0 designate that the buffer
 *   is setup properly and is ready to accept data.
 * See also:
 *   kiro_trb_reshape, kiro_trb_adopt, kiro_trb_clone
 */
int kiro_trb_is_setup (KiroTrb *trb);


/**
 * kiro_trb_reshape:
 * Reallocates internal memory and structures
 * @trb: #KiroTrb to perform the operation on
 * @element_size: Individual size of the elements to store in bytes
 * @element_count: Maximum number of elements to be stored
 * Description:
 *   (Re)Allocates internal memory for the given ammount of elements
 *   at the given individual size
 * Notes:
 *   If this function gets called when the buffer already has internal
 *   memory (buffer is setup), that memory gets freed automatically.
 *   If the function fails (Negative return value) none of the old
 *   memory and data structures get changed.
 * See also:
 *   kiro_trb_is_setup, kiro_trb_reshape, kiro_trb_adopt, kiro_trb_clone
 */
int kiro_trb_reshape (KiroTrb *trb, uint64_t element_size, uint64_t element_count);


/**
 * kiro_trb_clone:
 * Clones the given memory into the internal memory
 * @trb: #KiroTrb to perform the operation on
 * @source: Pointer to the source memory to clone from
 * Description:
 *   Interprets the given memory as a pointer to another KIRO TRB and
 *   tries to copy that memory into its own.
 * Notes:
 *   The given memory is treated as a correct KIRO TRB memory block,
 *   including a consistend memory header. That header is read and
 *   then cloned into the internal memory according to the headers
 *   information.
 *   If the given memory is not a consistent KIRO TRB memory block,
 *   the behavior of this function is undefined.
 *   Returns 0 if the buffer was cloned and -1 if memory allocation
 *   failed.
 * See also:
 *   kiro_trb_reshape, kiro_trb_adopt
 */
int kiro_trb_clone (KiroTrb *trb, void *source);


/**
 * kiro_trb_push:
 * Adds an element into the buffer
 * @trb: #KiroTrb to perform the operation on
 * @source: Pointer to the memory of the element to add
 * Description:
 *   Copies the given element and adds it into the buffer
 * Notes:
 *   This function will read n-Bytes from the given address according
 *   to the setup element_size. The read memory is copied directly
 *   into the internal memory structure.
 *   Returns 0 on success, -1 on failure.
 *   In case of failure, no internal memory will change as if the
 *   call to kiro_trb_push has never happened.
 * See also:
 *   kiro_trb_dma_push, kiro_trb_get_element_size, kiro_trb_clone,
 *   kiro_trb_adopt
 */
int kiro_trb_push (KiroTrb *trb, void *source);


/**
 * kiro_trb_refresh:
 * Re-reads the TRBs memory header
 * @trb: #KiroTrb to perform the operation on
 * Description:
 *   Re-reads the internal memory header and sets up all pointers
 *   and counters in accordance to these information
 * Notes:
 *   This function is used in case the TRBs memory got changed
 *   directly (For example, by a DMA operation) to make the TRB
 *   aware of the changes to its memory. Only the buffers memory
 *   header is examined and changes are made according to these
 *   informations.
 * See also:
 *   kiro_trb_get_raw_buffer, kiro_trb_push_dma, kiro_trb_adopt
 */
void kiro_trb_refresh (KiroTrb *trb);


/**
 * kiro_trb_adopt:
 * Adopts the given memory into the TRB
 * @trb: #KiroTrb to perform the operation on
 * @source: Pointer to the source memory to adopt
 * Description:
 *   Interprets the given memory as a pointer to another KIRO TRB and
 *   takes ownership over the memory.
 * Notes:
 *   The given memory is treated as a correct KIRO TRB memory block,
 *   including a consistend memory header. That header is read and
 *   the TRB sets up all internal structures in accordance to that
 *   header.
 *   If the given memory is not a consistent KIRO TRB memory block,
 *   the behavior of this function is undefined.
 *   The TRB takes full ownership of the given memory and may free
 *   it at will.
 * See also:
 *   kiro_trb_clone, kiro_trb_reshape
 */
void kiro_trb_adopt (KiroTrb *trb, void *source);

G_END_DECLS

#endif //__KIRO_TRB_H
