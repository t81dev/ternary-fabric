#ifndef FABRIC_NET_H
#define FABRIC_NET_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Simulated RDMA layer for multi-node TFMBS.
 */

int fabric_net_init(int node_id);
int fabric_net_send(int dst_node, const void* buf, size_t len);
int fabric_net_recv(void* buf, size_t len);
void fabric_net_cleanup();

#endif
