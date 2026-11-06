// MIT; CPU I/O only. Both native controls consume exactly the same frozen F32 bytes.
#pragma once
#include <bit>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
inline std::vector<float> fp8_input(int argc,char** argv) {
    if(argc!=3 || std::endian::native!=std::endian::little) throw std::runtime_error("input.f32 output.f32 required, little endian only");
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);
    const std::streamoff bytes=f.tellg();
    if(bytes<=0 || bytes>64*1024*1024 || bytes%(512*4)) throw std::runtime_error("bounded complete 512-value vectors required");
    std::vector<float> x(size_t(bytes)/4); f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(x.data()),bytes)) throw std::runtime_error("input read failed");
    for(float v:x) if(!std::isfinite(v)) throw std::runtime_error("nonfinite control input");
    return x;
}
inline void fp8_output(const char* path,const std::vector<float>& x) {
    for(float v:x) if(!std::isfinite(v)) throw std::runtime_error("nonfinite control output");
    std::ofstream f(path,std::ios::binary);
    if(!f.write(reinterpret_cast<const char*>(x.data()),x.size()*4)) throw std::runtime_error("output write failed");
    // Give the unchanged device-fd auditor time to observe the actual allocation.
    std::this_thread::sleep_for(std::chrono::seconds(2));
}
