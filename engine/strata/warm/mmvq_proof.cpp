// CPU proof of real layout/decoder independently against the raw scalar decoder.
#include "mmvq_project.hpp"
#include "mmvq_metrics.hpp"
#include <cassert>
#include <cstdio>
#include <random>
namespace mq=strata::wiring::mmvq;
namespace m=strata::ds4::moe;
static void half(std::vector<uint8_t>& b,int type,uint16_t h){
 if(type!=2){std::memcpy(b.data(),&h,2);return;}
 for(int j=0;j<4;++j){uint16_t s=uint16_t((m::u16(b.data()+48+2*j)&4095)|(((h>>(4*j))&15)<<12));std::memcpy(b.data()+48+2*j,&s,2);}
}
int main(){auto tables=mq::codebooks();std::mt19937 rng(0x10165);size_t checked=0;
 const m::Type types[]={m::Type::iq2_xxs,m::Type::iq2_xs,m::Type::iq1_m};
 for(int t=0;t<3;++t){std::vector<uint8_t> raw(m::block_bytes(types[t]));for(auto& v:raw)v=uint8_t(rng());half(raw,t,0x3555);
  // All IQ2_XXS index/sign pairs, all IQ2_XS encoded words, all IQ1 indices,
  // deltas and six-bit scale pairs. Each occupies every grid slot/subblock.
  int cases=t==0?256*128:t==1?512*128:2048*2*64;
  for(int c=0;c<cases;++c){for(int sb=0;sb<8;++sb){
   if(t==0){for(int j=0;j<4;++j)raw[2+8*sb+j]=uint8_t(c%256);uint32_t aux=uint32_t((c+sb)%16)<<28;for(int j=0;j<4;++j)aux|=uint32_t(c/256)<<(7*j);std::memcpy(raw.data()+6+8*sb,&aux,4);}
   else if(t==1){for(int j=0;j<4;++j){uint16_t code=uint16_t(c);std::memcpy(raw.data()+2+8*sb+2*j,&code,2);}raw[66+sb]=uint8_t(c+sb);}
   else{unsigned index=(c/128)%2048,negative=(c/64)%2;for(int j=0;j<4;++j)raw[4*sb+j]=uint8_t(index);unsigned nibble=(index>>8)|(negative<<3);raw[32+2*sb]=raw[33+2*sb]=uint8_t(nibble|(nibble<<4));uint16_t sc=uint16_t((c%64)|((c%64)<<6));std::memcpy(raw.data()+48+2*(sb/2),&sc,2);}
  }half(raw,t,0x3555);auto soa=mq::reorder(t,1,256,raw);assert(mq::restore(t,1,256,soa)==raw);
   for(int sb=0;sb<8;++sb){std::array<int8_t,32>x;for(int j=0;j<32;++j)x[j]=int8_t((j*73+c+sb)%256-128);auto decoded=m::decode(types[t],raw.data(),sb);int lo=0,hi=0;for(int j=0;j<16;++j){lo+=decoded.q[j]*x[j];hi+=decoded.q[j+16]*x[j+16];}auto got=t==0?mq::subdot<0>(soa.data(),8,sb,x.data(),tables.data()):t==1?mq::subdot<1>(soa.data(),8,sb,x.data(),tables.data()):mq::subdot<2>(soa.data(),8,sb,x.data(),tables.data());assert(got.integer==decoded.s[0]*lo+decoded.s[1]*hi);assert(std::bit_cast<uint32_t>(got.scale)==std::bit_cast<uint32_t>(decoded.d));++checked;}
  }
  // Every finite scale encoding including signed zero and subnormals.
  std::array<int8_t,32>x{};
  for(unsigned h=0;h<65536;++h){half(raw,t,uint16_t(h));if((h&0x7c00)==0x7c00){bool rejected=false;try{mq::reorder(t,1,256,raw);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);continue;}auto soa=mq::reorder(t,1,256,raw);auto got=t==0?mq::subdot<0>(soa.data(),8,0,x.data(),tables.data()):t==1?mq::subdot<1>(soa.data(),8,0,x.data(),tables.data()):mq::subdot<2>(soa.data(),8,0,x.data(),tables.data());assert(std::bit_cast<uint32_t>(got.scale)==std::bit_cast<uint32_t>(m::half(uint16_t(h))*0.125f));++checked;}
 }
 for(auto dims:{std::pair(0,256),std::pair(1,32),std::pair(1,0)}){bool bad=false;try{mq::reorder(0,dims.first,dims.second,{});}catch(...){bad=true;}assert(bad);}
 assert(!mq::metrics(std::array<float,2>{1,2},std::array<float,2>{1,1}).expert_ok());
 bool nonfinite=false;try{mq::metrics(std::array<float,1>{INFINITY},std::array<float,1>{1});}catch(...){nonfinite=true;}assert(nonfinite);
 std::printf("PASS: %zu integer/scale comparisons; exhaustive grid/sign/delta/scale fields, lossless byte roundtrip, all finite fp16 scales, rejection of invalid input\n",checked);
}
