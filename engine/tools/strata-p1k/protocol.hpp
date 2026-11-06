// Host-only handshake. No changes to inference arithmetic or device kernels.
#pragma once
#include <cstdlib>
#include <string>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
inline bool p1k_n0(){const char* v=std::getenv("P1K_N0");return v && std::string(v)=="1";}
inline void p1k_gate(const char* event,const char* release){
    const char* root=std::getenv("P1K_GATE_DIR");if(!root)return;
    namespace fs=std::filesystem;fs::path dir(root);
    int fd=::open((dir/event).c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)throw std::runtime_error("exclusive payload marker");
    if(::write(fd,"{}\n",3)!=3){::close(fd);throw std::runtime_error("payload marker write");}::close(fd);
    auto until=std::chrono::steady_clock::now()+std::chrono::seconds(120);
    while(!fs::is_regular_file(dir/release)){
        if(fs::exists(dir/"stop")||std::chrono::steady_clock::now()>=until)throw std::runtime_error("payload gate expired/stopped");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
