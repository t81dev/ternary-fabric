#ifndef TFMBS_DRIVER_H
#define TFMBS_DRIVER_H

/**
 * @brief Mock driver interface for Phase 10.
 */
int tfmbs_dev_open(const char* path, int flags);
int tfmbs_dev_ioctl(int fd, unsigned long request, void* arg);
int tfmbs_dev_close(int fd);

#endif
