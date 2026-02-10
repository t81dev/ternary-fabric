#ifndef FABRIC_NET_H
#define FABRIC_NET_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Enhanced Simulated RDMA layer for multi-node TFMBS.
 * Mimics ibverbs-style Queue Pairs and Completion Queues.
 */

typedef struct {
    uint64_t wr_id;
    void* addr;
    size_t length;
} tfmbs_rdma_wr_t;

typedef struct {
    uint64_t wr_id;
    int status;
} tfmbs_rdma_wc_t;

int fabric_net_init(int node_id);
int fabric_net_post_send(int dst_node, tfmbs_rdma_wr_t* wr);
int fabric_net_poll_cq(tfmbs_rdma_wc_t* wc, int max_entries);

int fabric_net_send(int dst_node, const void* buf, size_t len);
int fabric_net_recv(void* buf, size_t len);
void fabric_net_cleanup();

#endif
