// MIT. Opt-in IQ2 strip: exact original lane/FMA/XOR order, raw tensor layout.
#pragma once
#include "packed_project.hpp"
namespace strata::wiring::warm_strip {
using namespace ds4::row2::iq2;
// Four independent 2-bit symbols -> four magnitude bytes. No byte carries.
ESIMD_INLINE U magnitudes(U code,int half4) {
 U c=(code>>uint32_t(8*half4))&255u;
 U bytes=(c&3u)|((c&12u)<<6)|((c&48u)<<12)|((c&192u)<<18);
 return U(0x08080808u)+17u*bytes+((bytes>>1)&0x01010101u);
}
ESIMD_INLINE I signed_word(U mag,U signs,int half4) {
 U s=(signs>>uint32_t(4*half4))&15u;
 U bits=(s&1u)|((s&2u)<<7)|((s&4u)<<14)|((s&8u)<<21);
 U mask=bits*255u;
 // Each magnitude is 8,25,43: a negated byte never carries to its neighbour.
 U word=(mag^mask)+(mask&0x01010101u);
 return word.template bit_cast_view<int32_t>().read();
}
ESIMD_INLINE F half_fast(U h) {
 U sign=(h&0x8000u)<<16,mant=h&1023u,exp=(h>>10)&31u;
 U norm=sign|((exp+112u)<<23)|(mant<<13);
 norm.merge(sign|0x7f800000u|(mant<<13),exp==31u);
 U shift=ex::lzd<uint32_t>(mant)-21u; // 1..11, including mantissa zero
 U sub=sign|((113u-shift)<<23)|(((mant<<shift)&1023u)<<13);
 sub.merge(sign,mant==0u);norm.merge(sub,exp==0u);
 return norm.template bit_cast_view<float>().read();
}
template<int Entries>
ESIMD_INLINE void table_init(const uint16_t* compact,int local) {
 e::slm_init<Entries*8>();
 // All work-items reach the barrier, including rows beyond the output tail.
 if(local<Entries/32){
  U lane(0,1);U index=lane+uint32_t(local*32);
  U code=e::gather<uint16_t,32>(compact,index*2u,e::simd<uint16_t,32>(1),e::simd<uint16_t,32>(0));
  e::slm_block_store<uint32_t,32>(uint32_t(local*128),magnitudes(code,0));
  e::slm_block_store<uint32_t,32>(uint32_t(Entries*4+local*128),magnitudes(code,1));
 }
 e::barrier();
}
// Align down only within the actual matrix. Transactions never cross its end.
// The final tile/row and arbitrary pointer alignments use bounded byte gathers.
template<int Block,int Width>
ESIMD_INLINE e::simd<uint8_t,Width> weights(const uint8_t* w,size_t ld,int n,int row,int base,int k) {
 const uint8_t* tile=w+size_t(row)*ld+size_t(base/8)*Block;
 const uintptr_t first=reinterpret_cast<uintptr_t>(w),last=first+size_t(n)*ld;
 const uintptr_t address=reinterpret_cast<uintptr_t>(tile),aligned=address&~uintptr_t(15);
 const uint16_t prefix=uint16_t(address-aligned);
 e::simd<uint8_t,384> buf=0;U lane(0,1);
#pragma unroll
 for(int part=0;part<3;++part){
  const uintptr_t at=aligned+size_t(part)*128;
  if(at>=first&&at<=last&&last-at>=128){
#ifdef STRATA_IQ2_HOST_TEST
   auto words=e::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(at),e::overaligned_tag<16>{});
#else
   auto words=e::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(at),
    e::properties{e::alignment<16>,e::cache_hint_L1<e::cache_hint::cached>,e::cache_hint_L2<e::cache_hint::cached>});
#endif
   buf.template select<128,1>(part*128)=words.template bit_cast_view<uint8_t>().read();
  }else{
#pragma unroll
   for(int chunk=0;chunk<4;++chunk){
    // Use w as the valid base, not an out-of-object aligned-down pointer.
    U rel=lane+uint32_t(part*128+chunk*32);
    const uint32_t offset=uint32_t(address-first);
    auto valid=(rel+offset>=U(prefix))&(rel+offset-prefix<uint32_t(last-first));
    U off=rel+offset-prefix;
    buf.template select<32,1>(part*128+chunk*32)=e::gather<uint8_t,32>(w,off,valid,e::simd<uint8_t,32>(0));
   }
  }
 }
 e::simd<uint8_t,Width> out=0;
 const uint32_t remaining=uint32_t(k/256-base/8)*Block;
#pragma unroll
 for(int part=0;part<Width/32;++part){
  U off=lane+uint32_t(part*32);
  e::simd<uint8_t,32> v=buf.template select<32,1>(part*32+prefix);
  e::simd<uint8_t,32> z=0;z.merge(v,(off<remaining)&(off<uint32_t(Block*4)));
  out.template select<32,1>(part*32)=z;
 }
 return out;
}
// Four constant-stride regions replace a per-lane indirect register gather.
// Each region is one superblock's eight original logical lanes.
template<int Block,int Width>
ESIMD_INLINE U field(e::simd<uint8_t,Width>& bytes,int offset) {
 e::simd<uint8_t,32> out;
#pragma unroll
 for(int b=0;b<4;++b)out.template select<8,1>(b*8)=bytes.template select<8,8>(b*Block+2+offset).read();
 return U(out);
}
template<int Block,int Width>
ESIMD_INLINE U header(e::simd<uint8_t,Width>& bytes,int offset) {
 U out;
#pragma unroll
 for(int b=0;b<4;++b)out.template select<8,1>(b*8)=e::simd<uint32_t,8>(bytes[b*Block+offset]);
 return out;
}
template<bool XS,int Width>
ESIMD_INLINE F dot(e::simd<uint8_t,Width> bytes,e::simd<int32_t,256>& xw) {
 constexpr int block=XS?74:66,entries=XS?512:256;
 U aux=0;
 if constexpr(!XS){
  U a0=field<block>(bytes,4),a1=field<block>(bytes,5);
  U a2=field<block>(bytes,6),a3=field<block>(bytes,7);
  aux=a0|(a1<<8)|(a2<<16)|(a3<<24);
 }
 I lo=0,hi=0;
#pragma unroll
 for(int group=0;group<4;++group){
  U index,signs;
  if constexpr(XS){
   U a0=field<block>(bytes,2*group);
   U a1=field<block>(bytes,2*group+1);
   U word=a0|(a1<<8);index=word&511u;signs=parity_sign(word>>9);
  }else{index=field<block>(bytes,group);signs=parity_sign(aux>>uint32_t(7*group));}
  I sum=group<2?lo:hi;
#pragma unroll
  for(int half4=0;half4<2;++half4){
   U mag=e::slm_gather<uint32_t,32>(index*4u+uint32_t(half4*entries*4));
   I activation=xw.template select<32,8>(2*group+half4);
   sum=e::dp4a<int32_t>(sum,signed_word(mag,signs,half4),activation);
  }
  if(group<2)lo=sum;else hi=sum;
 }
 I s0,s1;
 if constexpr(XS){
  e::simd<uint8_t,32> raw;
#pragma unroll
  for(int b=0;b<4;++b)raw.template select<8,1>(b*8)=bytes.template select<8,1>(b*74+66).read();
  U s=raw;s0=2u*(s&15u)+1u;s1=2u*(s>>4)+1u;
 }
 else{s0=2u*(aux>>28)+1u;s1=s0;}
 U h0=header<block>(bytes,0),h1=header<block>(bytes,1);
 F d=half_fast(h0|(h1<<8))*0.125f;
 F integer=s0*lo+s1*hi;
 return d*integer;
}
template<bool XS>
ESIMD_INLINE void strip2(const uint8_t* w,size_t ld,const int8_t* xq,const float* xd,
 int n,int k,int first,float* y){
 e::simd<float,64> accum=0.0f;U lane(0,1);
 for(int base=0;base<k/32;base+=32){
  auto x=input_load(xq,base,k);e::simd<int32_t,256> xw=x.template bit_cast_view<int32_t>().read();
  auto active=lane<uint32_t(k/32-base);
  F scales=e::gather<float,32>(xd+base,lane*4u,active,F(0));
#pragma unroll
  for(int r=0;r<2;++r)if(first+r<n){
   auto bytes=weights<XS?74:66,XS?320:288>(w,ld,n,first+r,base,k);
   F subdot=dot<XS>(bytes,xw),old=accum.template select<32,1>(r*32);
   F next=ex::fma<float,32>(scales,subdot,old);old.merge(next,active);
   accum.template select<32,1>(r*32)=old;
  }
 }
#pragma unroll
 for(int r=0;r<2;++r)if(first+r<n){F a=accum.template select<32,1>(r*32);a=xor_tree(a);float v=1.0f*float(a[0]);y[first+r]=v+0.0f;}
}
}
