// Probe the actual pinned image/kernel/seccomp combination, without model I/O.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <unistd.h>
int main(){io_uring_params p{};const long fd=syscall(__NR_io_uring_setup,1,&p);
    if(fd<0)std::printf("io_uring_setup: unavailable errno=%d %s; pread threads selected\n",errno,std::strerror(errno));
    else {std::printf("io_uring_setup: available features=%u; pread threads selected for this packet\n",p.features);close(int(fd));}
}
