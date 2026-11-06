// MIT. Opt-in host inputs, before quantization; consumer: strata-mmvq-accuracy.
#pragma once
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::wiring {
class ActivationCapture {
 std::FILE* file_=nullptr;
 std::vector<bool> seen_;
 uint32_t dim_=0;
 std::mutex mutex_;
public:
 ActivationCapture(uint32_t layers,uint32_t dim):dim_(dim){
  const char* path=std::getenv("DS4_MMVQ_ACTIVATIONS");
  if(!path)return;
  if(!layers||layers>4096||!dim||!path[0]||path[0]!='/'||std::endian::native!=std::endian::little)
   throw std::runtime_error("invalid MMVQ capture dimensions/path/endianness");
  // At most one record per layer, compatible with the consumer's 4096 limit.
  seen_.resize(layers,false);
  file_=std::fopen(path,"wbx");
  if(!file_)throw std::runtime_error("MMVQ activation capture requires fresh output");
 }
 ActivationCapture(const ActivationCapture&)=delete;
 ActivationCapture& operator=(const ActivationCapture&)=delete;
 ~ActivationCapture(){if(file_)std::fclose(file_);}
 void record(uint32_t layer,std::span<const float> x){
  if(!file_)return;
  std::lock_guard<std::mutex> lock(mutex_);
  if(layer>=seen_.size()||x.size()!=dim_)throw std::runtime_error("MMVQ capture extent");
  if(seen_[layer])return;
  for(float v:x)if(!std::isfinite(v))throw std::runtime_error("MMVQ capture nonfinite input");
  const uint32_t header[]={layer,dim_};
  if(std::fwrite(header,sizeof(uint32_t),2,file_)!=2||
     std::fwrite(x.data(),sizeof(float),x.size(),file_)!=x.size()||std::fflush(file_))
   throw std::runtime_error("MMVQ activation capture write");
  seen_[layer]=true;
 }
};
}
