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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifndef __KIRO_RDMA_H__
#define __KIRO_RDMA_H__


#include <rdma/rdma_cma.h>

  
struct kiro_connection_context {
        
    // Information and necessary structurs
    uint32_t                identifier;             // Unique Identifier for this connection (Application Specific)
    struct kiro_rdma_mem    *cf_mr_recv;            // Control-Flow Memory Region Receive
    struct kiro_rdma_mem    *cf_mr_send;            // Control-Flow Memory Region Send
    struct kiro_rdma_mem    *rdma_mr;               // Memory Region for RDMA Operations

    struct ibv_mr           peer_mr;                // RDMA Memory Region Information of the peer
    
    enum {
        KIRO_IDLE,
        KIRO_MRI_REQUESTED,                         // Memory Region Information Requested
        KIRO_RDMA_ESTABLISHED,                      // MRI Exchange complete. RDMA is ready
        KIRO_RDMA_ACTIVE                            // RDMA Operation is being performed
    } rdma_state;
    
};


struct kiro_connection {
    
    uint32_t                identifier;
    struct rdma_cm_id       *id;
    struct kiro_connection  *next;
    
};


struct kiro_ctrl_msg {
    
    enum {
        KIRO_REQ_RDMA,                              // Requesting RDMA Access to/from the peer
        KIRO_ACK_RDMA,                              // acknowledge RDMA Request and provide Memory Region Information
        KIRO_REJ_RDMA                               // RDMA Request rejected :(  (peer_mri will be invalid)
    } msg_type;
    
    struct ibv_mr peer_mri;
    
};


struct kiro_rdma_mem {
    
    void            *mem;   // Pointer to the beginning of the memory block           
    struct ibv_mr   *mr;    // Memory Region associated with the memory
    size_t          size;   // Size in Bytes of the memory block

};


static int kiro_register_rdma_memory (struct ibv_pd *pd, struct ibv_mr **mr, void *mem, size_t mem_size, int access)
{
    
    if(mem_size == 0)
    {    
        printf("Cant allocate memory of size '0'.\n");
        return -1;
    }
    
    void *mem_handle = mem;
    
    if(!mem_handle)
        mem_handle = malloc(mem_size);
        
    if(!mem_handle)
    {
        printf("Failed to allocate memory [Register Memory].");
        return -1;
    }        
    
    *mr = ibv_reg_mr(pd, mem_handle, mem_size, access);
    if(!(*mr))
    {
        // Memory Registration failed
        printf("Failed to register memory region!\n");
        free(mem_handle);
        return -1;
    }
    
    return 0;
}


static struct kiro_rdma_mem* kiro_create_rdma_memory (struct ibv_pd *pd, size_t mem_size, int access)
{
    if(mem_size == 0)
    {    
        printf("Cant allocate memory of size '0'.\n");
        return NULL;
    }
    
    struct kiro_rdma_mem *krm = (struct kiro_rdma_mem *)calloc(1, sizeof(struct kiro_rdma_mem));
    if(!krm)
    {
        printf("Failed to create new KIRO RDMA Memory.\n");
        return NULL;
    }
    
    if(kiro_register_rdma_memory(pd, &(krm->mr), krm->mem, mem_size, access))
    {
        free(krm);
        return NULL;
    }
    
    if(!krm->mem)
        krm->mem = krm->mr->addr;
    
    
    return krm;
    
}


static void kiro_destroy_rdma_memory (struct kiro_rdma_mem *krm)
{
    if(!krm)
        return;
        
    if(krm->mr)
        ibv_dereg_mr(krm->mr);
        
    if(krm->mem)
        free(krm->mem);
        
    free(krm);
    krm = NULL;
}


static void kiro_destroy_connection_context (struct kiro_connection_context **ctx)
{
    if(!ctx)
        return;
    
    if(!(*ctx))
        return;
        
    if((*ctx)->cf_mr_recv)
        kiro_destroy_rdma_memory((*ctx)->cf_mr_recv);
    if((*ctx)->cf_mr_send)
        kiro_destroy_rdma_memory((*ctx)->cf_mr_send);
        
    //The RDMA-Memory Region normally contains allocated memory from the USER that has
    //just been 'registered' for RDMA. DON'T free it! Just deregister it. The user is
    //responsible for freeing this memory. 
    if((*ctx)->rdma_mr)
    {
        if((*ctx)->rdma_mr->mr)
            ibv_dereg_mr((*ctx)->rdma_mr->mr);
           
        free((*ctx)->rdma_mr);
        (*ctx)->rdma_mr = NULL;
    }

    free(*ctx);
    *ctx = NULL;
}


static void kiro_destroy_connection (struct kiro_connection **conn)
{
    if(!(*conn))
        return;
    
    if(!(*conn)->id)
        return;
        
    rdma_disconnect((*conn)->id);
    struct kiro_connection_context *ctx = (struct kiro_connection_context *)((*conn)->id->context);
    if(ctx)
        kiro_destroy_connection_context(&ctx);
        
    rdma_destroy_ep((*conn)->id);
    free(*conn);
    *conn = NULL;
}


#endif //__KIRO_RDMA_H__   