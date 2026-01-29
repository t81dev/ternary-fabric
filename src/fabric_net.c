#include "fabric_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>

static int g_node_id = -1;
static int g_listen_fd = -1;

int fabric_net_init(int node_id) {
    g_node_id = node_id;
    char path[108];
    snprintf(path, sizeof(path), "/tmp/tfmbs_node_%d.sock", node_id);
    unlink(path);

    g_listen_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (g_listen_fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path)-1);

    if (bind(g_listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(g_listen_fd);
        return -1;
    }

    printf("[TFMBS-Net] Node %d initialized (RDMA simulation via %s)\n", node_id, path);
    return 0;
}

int fabric_net_send(int dst_node, const void* buf, size_t len) {
    char path[108];
    snprintf(path, sizeof(path), "/tmp/tfmbs_node_%d.sock", dst_node);

    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path)-1);

    ssize_t sent = sendto(fd, buf, len, 0, (struct sockaddr*)&addr, sizeof(addr));
    close(fd);

    if (sent < 0) {
        if (getenv("TFMBS_DEBUG")) perror("[TFMBS-Net] sendto failed");
        return -1;
    }
    return 0;
}

int fabric_net_recv(void* buf, size_t len) {
    if (g_listen_fd < 0) return -1;
    ssize_t recvd = recv(g_listen_fd, buf, len, 0);
    return (recvd < 0) ? -1 : 0;
}

void fabric_net_cleanup() {
    if (g_listen_fd >= 0) {
        close(g_listen_fd);
        char path[108];
        snprintf(path, sizeof(path), "/tmp/tfmbs_node_%d.sock", g_node_id);
        unlink(path);
    }
}
