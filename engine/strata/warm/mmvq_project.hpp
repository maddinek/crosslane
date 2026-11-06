// MIT. MMVQ lane geometry, planar weight fields, grouped selected experts.
// Packed-byte vector dot idiom: llama.cpp e14a779, dpct/helper.hpp.
// The source's Intel MIT / LLVM exception notices are retained in research/.
#pragma once
#include "mmvq_layout.hpp"
#include "grouped_project.hpp"
namespace strata::wiring::mmvq {
inline uint32_t word(const uint8_t* p){uint32_t v;std::memcpy(&v,p,4);return v;}
inline uint16_t halfword(const uint8_t* p){uint16_t v;std::memcpy(&v,p,2);return v;}
inline unsigned sign_bits(unsigned v){v&=127;return v|((std::popcount(v)&1u)<<7);}
inline uint32_t signed_word(uint32_t v,unsigned bits){
 unsigned mask=((bits&1)|((bits&2)<<7)|((bits&4)<<14)|((bits&8)<<21))*255u;
 return (v^mask)+(mask&0x01010101u); // nonzero IQ2 magnitudes: no byte carry
}
inline int dot4(uint32_t a,uint32_t b,int sum){
#ifdef STRATA_IQ2_HOST_TEST
 for(int j=0;j<4;++j)sum+=int(int8_t(a>>(8*j)))*int(int8_t(b>>(8*j)));return sum;
#else
 // Same signed packed-byte vector idiom as upstream. Inspect native ISA for
 // dp4a; this expression alone is not proof of the compiler's lowering.
 auto va=sycl::vec<uint32_t,1>(a).template as<sycl::vec<int8_t,4>>().template convert<int32_t>();
 auto vb=sycl::vec<uint32_t,1>(b).template as<sycl::vec<int8_t,4>>().template convert<int32_t>();
 return sum+va[0]*vb[0]+va[1]*vb[1]+va[2]*vb[2]+va[3]*vb[3];
#endif
}
struct Dot {int integer;float scale;};
template<int Type>inline Dot subdot(const uint8_t* w,size_t sub,size_t s,const int8_t* x,const uint32_t* table){
 unsigned lo=0,hi=0,sc=0,dh=0;const size_t b=s/8,blocks=sub/8;
 if constexpr(Type!=2){lo=word(w+4*s);hi=word(w+4*sub+4*s);sc=Type==0?hi>>28:w[8*sub+s];dh=halfword(w+(Type==0?8:9)*sub+2*b);}
 else{unsigned h[4];for(int j=0;j<4;++j)h[j]=halfword(w+6*sub+2*(size_t(j)*blocks+b));dh=(h[0]>>12)|((h[1]>>8)&0xf0)|((h[2]>>4)&0xf00)|(h[3]&0xf000);sc=(h[(s%8)/2]>>(6*(s&1)))&63;}
 int sums[2]={0,0};
#pragma unroll
 for(int j=0;j<4;++j){unsigned index,sign=0;
  if constexpr(Type==0){index=(lo>>(8*j))&255;sign=sign_bits(hi>>(7*j));}
  else if constexpr(Type==1){unsigned code=((j<2?lo:hi)>>(16*(j&1)))&65535;index=256+(code&511);sign=sign_bits(code>>9);}
  else{unsigned h=w[size_t(4+j/2)*sub+s]>>(4*(j&1));index=256+512+2*(unsigned(w[size_t(j)*sub+s])|((h&7)<<8))+((h>>3)&1);}
  uint32_t a=table[2*index],c=table[2*index+1];if constexpr(Type!=2){a=signed_word(a,sign);c=signed_word(c,sign>>4);}
  sums[j/2]=dot4(a,word(reinterpret_cast<const uint8_t*>(x)+8*j),sums[j/2]);
  sums[j/2]=dot4(c,word(reinterpret_cast<const uint8_t*>(x)+8*j+4),sums[j/2]);
 }
 const int s0=Type==0?2*int(sc)+1:Type==1?2*int(sc&15)+1:2*int(sc&7)+1;
 const int s1=Type==0?s0:Type==1?2*int(sc>>4)+1:2*int(sc>>3)+1;
#ifdef STRATA_IQ2_HOST_TEST
 const float d=m::half(uint16_t(dh));
#else
 const float d=float(sycl::bit_cast<sycl::half>(uint16_t(dh)));
#endif
 return {s0*sums[0]+s1*sums[1],d*0.125f};
}
template<int Type,bool GateUp,bool Exact=false,int Rows=4>sycl::event project(sycl::queue& q,WarmPackedBatch a,int count,int n,int k,float* y,float* up,const int8_t* x,const float* d,const uint32_t* table){
 static_assert(Rows==1||Rows==4);
 const size_t strips=(size_t(n)+Rows-1)/Rows,sub=size_t(n)*size_t(k/32);
 return q.parallel_for(sycl::nd_range<1>(strips*size_t(count)*(GateUp?2:1)*Rows*32,Rows*32),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]]{
  const size_t part=it.get_group_linear_id()/strips;const int row=int(it.get_group_linear_id()%strips)*Rows+int(it.get_local_linear_id()/32),lane=int(it.get_local_linear_id()%32);
  if(row>=n)return; // uniform within the absent subgroup
  const size_t expert=GateUp?part/2:part;const bool is_up=GateUp&&(part&1);
  const uint8_t* w=GateUp?(is_up?a.up[expert]:a.gate[expert]):a.down[expert];
  const int8_t* qx=x+(GateUp?0:expert*size_t(k));const float* dx=d+(GateUp?0:expert*size_t(k/32));
  float sum=0;
  if constexpr(Exact){
   // Match packed-strip: round weight*integer BEFORE the activation-scale
   // FMA; each logical lane owns kb=lane+32*j. No reassociation/fast math.
   for(int kb=lane;kb<k/32;kb+=32){auto v=subdot<Type>(w,sub,size_t(row)*(k/32)+kb,qx+32*kb,table);
    const float dot=v.scale*float(v.integer);sum=sycl::fma(dx[kb],dot,sum);
   }
  }else{
   float even=0,odd=0;
   for(int kb=2*lane;kb<k/32;kb+=64){auto v=subdot<Type>(w,sub,size_t(row)*(k/32)+kb,qx+32*kb,table);even=sycl::fma(v.scale*dx[kb],float(v.integer),even);
   if(kb+1<k/32){auto z=subdot<Type>(w,sub,size_t(row)*(k/32)+kb+1,qx+32*(kb+1),table);odd=sycl::fma(z.scale*dx[kb+1],float(z.integer),odd);}
   }
   sum=even+odd;
  }
  auto sg=it.get_sub_group();for(int offset=16;offset;offset>>=1)sum+=sycl::permute_group_by_xor(sg,sum,offset);
  if(!lane)(is_up?up:y)[expert*size_t(n)+row]=Exact?sum+0.0f:sum;
 });
}
template<bool GateUp>sycl::event dispatch(sycl::queue& q,std::string_view type,WarmPackedBatch a,int count,int n,int k,float* y,float* up,const int8_t* x,const float* d,const uint32_t* table,bool exact_order=false){
 const int t=kind(type);extent(t,n,k);require(count>0&&count<=kPicks&&y&&x&&d&&table&&(!GateUp||up),"MMVQ launch extent");
 for(int i=0;i<count;++i)require(GateUp?(a.gate[i]&&a.up[i]):bool(a.down[i]),"MMVQ null weight");
 if(exact_order){
  if(t==0)return project<0,GateUp,true>(q,a,count,n,k,y,up,x,d,table);
  if(t==1)return project<1,GateUp,true>(q,a,count,n,k,y,up,x,d,table);
  return project<2,GateUp,true>(q,a,count,n,k,y,up,x,d,table);
 }
 if(t==0)return project<0,GateUp>(q,a,count,n,k,y,up,x,d,table);
 if(t==1)return project<1,GateUp>(q,a,count,n,k,y,up,x,d,table);
 return project<2,GateUp>(q,a,count,n,k,y,up,x,d,table);
}
// Scratch is shared with the existing grouped workspace; no ABI/routing change.
inline WarmGroupedWorkspace::Events enqueue(sycl::queue& q,WarmGroupedWorkspace& w,WarmPackedBatch a,int count,std::string_view gu,std::string_view dn,float lim,const uint32_t* table,bool exact_order=false,sycl::event* swiglu_event=nullptr){
 require(count>0&&count<=w.capacity&&std::isfinite(lim),"MMVQ count/limit");
 WarmGroupedWorkspace::Events e;e.input=ds4::moe::quantize(q,w.x,size_t(w.dim),w.dim,1,w.quant,w.scale);
 e.gate_up=dispatch<true>(q,gu,a,count,w.ff,w.dim,w.gate,w.up,w.quant,w.scale,table,exact_order);
 auto swiglu=ds4::row2::swiglu(q,w.gate,w.up,w.ff*count,lim,w.hidden);if(swiglu_event)*swiglu_event=swiglu;
 e.hidden_quant=ds4::moe::quantize(q,w.hidden,size_t(w.ff),w.ff,count,w.quant,w.scale);
 e.down=dispatch<false>(q,dn,a,count,w.dim,w.ff,w.y,nullptr,w.quant,w.scale,table,exact_order);return e;
}
}
