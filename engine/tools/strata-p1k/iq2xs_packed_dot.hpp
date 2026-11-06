// MIT. Raw IQ2_XS packed dot; preserves the scalar 32-lane arithmetic order.
// Adapted from crosslane's accepted IQ2_XXS packed-dot implementation.
#pragma once
#include "iq2_packed_dot.hpp"
namespace p1k_iq2xs {
using namespace strata::ds4::row2::iq2;
inline constexpr auto compact=[] {
    std::array<uint16_t,512> result{};
    for(int i=0;i<512;++i)result[i]=compact_entry(strata::ds4::moe::tables::iq2xs_grid[i]);
    return result;
}();
static_assert(sizeof(compact)==1024);
static_assert([]{for(int i=0;i<512;++i)for(int j=0;j<8;++j){
    unsigned c=(compact[i]>>(2*j))&3u;
    if(c>2||8+17*c+(c>>1)!=((strata::ds4::moe::tables::iq2xs_grid[i]>>(8*j))&255u))return false;
}return true;}());
// Four 74-byte superblocks = 296 bytes. Mask the final 24 bytes of the
// 320-byte tile AND every absent complete superblock; no row padding assumed.
ESIMD_INLINE e::simd<uint8_t,320> packed(const uint8_t* row,int base,int k){
    e::simd<uint8_t,320> result=0;U byte(0,1);
    const uint32_t start=uint32_t(base/8)*74u,remaining=uint32_t(k/256)*74u-start;
#pragma unroll
    for(int part=0;part<10;++part){
        U off=byte+uint32_t(part*32);auto mask=(off<remaining)&(off<296u);
        result.template select<32,1>(part*32)=e::gather<uint8_t,32>(row+start,off,mask,e::simd<uint8_t,32>(0));
    }
    return result;
}
ESIMD_INLINE F dot(e::simd<uint8_t,320> bytes,e::simd<int8_t,1024> x,const uint16_t* table){
    e::simd<int32_t,256> xw=x.template bit_cast_view<int32_t>().read();
    U lane(0,1);Index block=(lane/8u)*74u;Index p=block+2u+(lane%8u)*8u;
    I lo=0,hi=0;
#pragma unroll
    for(int group=0;group<4;++group){
        U b0=bytes.template iselect<32>(Index(p+uint16_t(2*group)));
        U b1=bytes.template iselect<32>(Index(p+uint16_t(2*group+1)));
        U word=b0|(b1<<8);U index=word&511u;U signs=parity_sign(word>>9);
#ifdef STRATA_IQ2_HOST_TEST
        auto grid=e::gather<uint16_t,32>(table,index*2u,e::simd<uint16_t,32>(1),e::simd<uint16_t,32>(0));
#else
        auto grid=e::gather<uint16_t,32>(table,index*2u,
            e::properties{e::cache_hint_L1<e::cache_hint::cached>,e::cache_hint_L2<e::cache_hint::cached>});
#endif
        I sum=group<2?lo:hi;
#pragma unroll
        for(int half4=0;half4<2;++half4)
            sum=e::dp4a<int32_t>(sum,p1k_packed::iq2_weight_word(U(grid),signs,half4),p1k_packed::activation_word(xw,group,half4));
        if(group<2)lo=sum;else hi=sum;
    }
    U scale=bytes.template iselect<32>(Index(block+66u+lane%8u));
    I s0=2u*(scale&15u)+1u,s1=2u*(scale>>4)+1u;
    U h0=bytes.template iselect<32>(block),h1=bytes.template iselect<32>(Index(block+1u));
    F d=half(h0|(h1<<8))*0.125f;
    F integer=s0*lo+s1*hi; // <= 5,459,968 in magnitude; exactly representable F32
    return d*integer; // rounded subdot BEFORE the original activation-scale FMA
}
ESIMD_INLINE void strip(const uint8_t* w,size_t ld,const int8_t* xq,const float* xd,
                        int k,int row,float* y,const uint16_t* table){
    F acc=0.0f;U lane(0,1);
    for(int base=0;base<k/32;base+=32){
        auto x=input_load(xq,base,k);auto active=lane<uint32_t(k/32-base);
        F scales=e::gather<float,32>(xd+base,lane*uint32_t(sizeof(float)),active,F(0));
        F subdot=dot(packed(w+size_t(row)*ld,base,k),x,table);
        F next=ex::fma<float,32>(scales,subdot,acc);acc.merge(next,active);
    }
    acc=xor_tree(acc);float value=1.0f*float(acc[0]);y[row]=value+0.0f;
}
inline sycl::event local4(sycl::queue& q,const uint8_t* w,size_t ld,int n,int k,
        float* y,const int8_t* xq,const float* xd,const uint16_t* table){
    return q.submit([&](sycl::handler& cgh){
        cgh.parallel_for(sycl::nd_range<1>((size_t(n)+3)/4*4,4),
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                int row=int(it.get_group_linear_id())*4+int(it.get_local_linear_id());if(row>=n)return;
                strip(w,ld,xq,xd,k,row,y,table);
            });
    });
}
// Same separate gate/up/down operations as the existing whole-expert path.
inline sycl::event gate_up(sycl::queue& q,const uint8_t* w,size_t ld,int n,int k,
        float* y,const int8_t* xq,const float* xd,const uint16_t* table){
    return local4(q,w,ld,n,k,y,xq,xd,table);
}
}
