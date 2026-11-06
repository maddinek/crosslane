// CPU-only exhaustive component proof of the actual opt-in header.
#include "packed_strip.hpp"
#include <random>
#include <cstdio>
#include <cstring>
using namespace strata::wiring;
using namespace strata::ds4::row2::iq2;
namespace f=e::fake;
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T,int N>void same(e::simd<T,N> a,e::simd<T,N> b,const char* name){need(std::memcmp(a.a.data(),b.a.data(),sizeof(T)*N)==0,name);}
template<int Entries>void init(sycl::queue& q,const uint16_t* table){
 q.submit([&](sycl::handler& h){h.parallel_for(sycl::nd_range<1>(16,16),
  sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
  [=](sycl::nd_item<1> it){warm_strip::table_init<Entries>(table,int(it.get_local_linear_id()));});});
}
int main()try{
 for(uint32_t i=0;i<65536;i+=32){U h(i,1);same(half(h).template bit_cast_view<uint32_t>().read(),warm_strip::half_fast(h).template bit_cast_view<uint32_t>().read(),"half bits");}
 size_t words=0;
 for(int xs=0;xs<2;++xs)for(int grid=0;grid<(xs?512:256);++grid)for(int sign=0;sign<128;++sign)for(int half4=0;half4<2;++half4){
  U code(xs?p1k_iq2xs::compact[grid]:compact_codebook[grid]),s=parity_sign(U(uint32_t(sign)));
  same(p1k_packed::iq2_weight_word(code,s,half4),warm_strip::signed_word(warm_strip::magnitudes(code,half4),s,half4),"packed magnitude/sign word");++words;
 }
 f::allow(compact_codebook.data(),sizeof(compact_codebook));f::allow(p1k_iq2xs::compact.data(),sizeof(p1k_iq2xs::compact));
 sycl::queue q;std::mt19937 rng(0x8b650008);size_t tiles=0,dots=0;
 for(int xs=0;xs<2;++xs){if(xs)init<512>(q,p1k_iq2xs::compact.data());else init<256>(q,compact_codebook.data());
  for(int k:{256,512,768,1024,1280,2304,5120})for(int n:{1,2,3,33})for(int align=0;align<16;++align){
   const size_t ld=size_t(k/256)*(xs?74:66);std::vector<uint8_t> storage(size_t(n)*ld+align);uint8_t* w=storage.data()+align;
   // Register ONLY actual matrix bytes: aligned-down overreads must fail.
   const auto old=f::regions;f::regions.clear();f::allow(w,size_t(n)*ld);
   for(size_t i=0;i<size_t(n)*ld;++i)w[i]=uint8_t(rng());
   for(int row=0;row<n;++row)for(int b=0;b<k/256;++b){uint16_t h=uint16_t(b%3==0?0x8001:b%3==1?0x3555:0xb555);std::memcpy(w+row*ld+b*(xs?74:66),&h,2);}
   f::allow(compact_codebook.data(),sizeof(compact_codebook));f::allow(p1k_iq2xs::compact.data(),sizeof(p1k_iq2xs::compact));
   for(int row=0;row<n;++row)for(int base=0;base<k/32;base+=32){
    e::simd<int8_t,1024>x;for(int i=0;i<1024;++i)x[i]=int8_t(int(rng()%256)-128);
    e::simd<int32_t,256>xw=x.template bit_cast_view<int32_t>().read();
    if(xs){auto a=p1k_iq2xs::packed(w+row*ld,base,k),b=warm_strip::weights<74,320>(w,ld,n,row,base,k);same(a,b,"XS tile bounds/layout");same(p1k_iq2xs::dot(a,x,p1k_iq2xs::compact.data()),warm_strip::dot<true>(b,xw),"XS subdot");}
    else{auto a=packed(w+row*ld,base,k),b=warm_strip::weights<66,288>(w,ld,n,row,base,k);same(a,b,"XXS tile bounds/layout");same(p1k_packed::iq2_dot(a,x,compact_codebook.data()),warm_strip::dot<false>(b,xw),"XXS subdot");}
    ++tiles;dots+=32;
   }f::regions=old;
  }
 }
 need(f::blocks>0&&f::slm_reads>0&&f::slm_writes==48&&f::barriers==32,"unused candidate path");
 std::printf("PASS: 65536 half encodings, %zu grid/sign words, %zu bounded/aligned tiles, %zu bit-exact lane subdots; block_load=%zu slm_read=%zu barriers=%zu\n",words,tiles,dots,f::blocks,f::slm_reads,f::barriers);
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
