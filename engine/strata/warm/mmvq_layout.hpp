// MIT. Lossless, same-byte-count SoA encoding for the opt-in warm bank only.
#pragma once
#include "../sycl/moe/core.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>
namespace strata::wiring::mmvq {
namespace m=ds4::moe;
inline int kind(std::string_view t){if(t=="IQ2_XXS")return 0;if(t=="IQ2_XS")return 1;if(t=="IQ1_M")return 2;throw std::invalid_argument("MMVQ type");}
inline size_t extent(int t,int n,int k){
 if(t<0||t>2||n<=0||k<=0||k%256)throw std::invalid_argument("MMVQ geometry");
 return m::product(m::product(size_t(n),size_t(k/256)),t==0?66:t==1?74:56);
}
// A field's elements are consecutive across all 32-value subblocks/rows.
// IQ2: two u32 planes, optional u8 scales, then f16 superblock scales.
// IQ1: four u8 index and two u8 high-bit planes, then four u16 scale planes.
inline std::vector<uint8_t> reorder(int t,int n,int k,std::span<const uint8_t> raw){
 const size_t bytes=extent(t,n,k),blocks=size_t(n)*size_t(k/256),sub=blocks*8;
 if(raw.size()!=bytes)throw std::invalid_argument("MMVQ raw extent");
 std::vector<uint8_t> out(bytes);
 for(size_t b=0;b<blocks;++b){const auto* p=raw.data()+b*(t==0?66:t==1?74:56);
  uint16_t d=t==2?uint16_t((m::u16(p+48)>>12)|((m::u16(p+50)>>8)&0xf0)|((m::u16(p+52)>>4)&0xf00)|(m::u16(p+54)&0xf000)):m::u16(p);
  if(!std::isfinite(m::half(d)))throw std::invalid_argument("MMVQ nonfinite weight scale");
  if(t!=2){std::memcpy(out.data()+(t==0?8:9)*sub+2*b,p,2);
   for(int i=0;i<8;++i){size_t s=b*8+i;std::memcpy(out.data()+4*s,p+2+8*i,4);std::memcpy(out.data()+4*sub+4*s,p+6+8*i,4);if(t==1)out[8*sub+s]=p[66+i];}
  }else{for(int i=0;i<8;++i){size_t s=b*8+i;for(int j=0;j<4;++j)out[size_t(j)*sub+s]=p[4*i+j];for(int j=0;j<2;++j)out[size_t(4+j)*sub+s]=p[32+2*i+j];}
   for(int j=0;j<4;++j)std::memcpy(out.data()+6*sub+2*(size_t(j)*blocks+b),p+48+2*j,2);
  }
 }
 return out;
}
inline std::vector<uint8_t> restore(int t,int n,int k,std::span<const uint8_t> soa){
 const size_t bytes=extent(t,n,k),blocks=size_t(n)*size_t(k/256),sub=blocks*8;
 if(soa.size()!=bytes)throw std::invalid_argument("MMVQ SoA extent");
 std::vector<uint8_t> out(bytes);
 for(size_t b=0;b<blocks;++b){auto* p=out.data()+b*(t==0?66:t==1?74:56);
  if(t!=2){std::memcpy(p,soa.data()+(t==0?8:9)*sub+2*b,2);for(int i=0;i<8;++i){size_t s=b*8+i;std::memcpy(p+2+8*i,soa.data()+4*s,4);std::memcpy(p+6+8*i,soa.data()+4*sub+4*s,4);if(t==1)p[66+i]=soa[8*sub+s];}}
  else{for(int i=0;i<8;++i){size_t s=b*8+i;for(int j=0;j<4;++j)p[4*i+j]=soa[size_t(j)*sub+s];for(int j=0;j<2;++j)p[32+2*i+j]=soa[size_t(4+j)*sub+s];}for(int j=0;j<4;++j)std::memcpy(p+48+2*j,soa.data()+6*sub+2*(size_t(j)*blocks+b),2);}
 }
 return out;
}
constexpr size_t table_words=2*(256+512+4096);
inline std::array<uint32_t,table_words> codebooks(){
 std::array<uint32_t,table_words> a{};
 for(size_t i=0;i<256+512;++i){const uint64_t v=i<256?m::tables::iq2xxs_grid[i]:m::tables::iq2xs_grid[i-256];a[2*i]=uint32_t(v);a[2*i+1]=uint32_t(v>>32);}
 for(unsigned i=0;i<2048;++i)for(unsigned negative=0;negative<2;++negative){uint64_t v=0;unsigned grid=m::tables::iq1s_grid_gpu[i];
  for(unsigned j=0;j<8;++j){int symbol=(grid>>(8*(j&3)+(j>=4?4:0)))&15;int q=8*(symbol-1)+(negative?-1:1);v|=uint64_t(uint8_t(q))<<(8*j);}
  size_t z=2*(256+512+2*i+negative);a[z]=uint32_t(v);a[z+1]=uint32_t(v>>32);
 }
 return a;
}
}
