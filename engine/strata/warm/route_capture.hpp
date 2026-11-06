// MIT. Teacher-only bounded host route trace; no additional device readback.
#pragma once
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <span>
namespace strata::wiring {
class RouteCapture {
 std::FILE* file_=nullptr;
 uint32_t steps_=0,layers_=0,picks_=0,records_=0;
 static void need(bool ok){if(!ok)throw std::runtime_error("invalid/failing MMVQ route capture");}
public:
 RouteCapture(){
  const char* path=std::getenv("DS4_MMVQ_ROUTES");if(!path)return;
  const char* count=std::getenv("DS4_MMVQ_ROUTE_STEPS");need(count&&*count);
  char* end=nullptr;const auto n=std::strtoul(count,&end,10);
  need(end&&!*end&&n>0&&n<=1024&&path[0]=='/'&&std::endian::native==std::endian::little);
  steps_=uint32_t(n);file_=std::fopen(path,"wbx");need(file_);
 }
 RouteCapture(const RouteCapture&)=delete;
 RouteCapture& operator=(const RouteCapture&)=delete;
 ~RouteCapture(){if(file_)std::fclose(file_);}
 void record(uint32_t position,uint32_t layer,uint32_t layers,std::span<const int> ids,std::span<const float> weights){
  if(!file_)return;
  need(layers>0&&layers<=4096&&uint64_t(layers)*steps_<=65536&&ids.size()>0&&ids.size()<=32&&weights.size()==ids.size());
  if(!records_){layers_=layers;picks_=uint32_t(ids.size());}
  need(layers==layers_&&ids.size()==picks_&&position<steps_&&layer<layers_&&records_==position*layers_+layer);
  const uint32_t header[]={position,layer,picks_};
  need(std::fwrite(header,4,3,file_)==3);
  for(size_t i=0;i<ids.size();++i){
   need(ids[i]>=0&&std::isfinite(weights[i]));
   for(size_t j=0;j<i;++j)need(ids[j]!=ids[i]);
   const uint32_t id=uint32_t(ids[i]);
   need(std::fwrite(&id,4,1,file_)==1&&std::fwrite(&weights[i],4,1,file_)==1);
  }
  need(!std::fflush(file_));++records_;
 }
};
}
