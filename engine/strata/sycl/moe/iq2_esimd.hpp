// MIT; crosslane contributors. Raw IQ2_XXS bytes, original row2 arithmetic.
// CPU tests compile THIS implementation with an explicit, bounds-checking ESIMD fake.
#pragma once
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include "tables.hpp"
#include "iq2_codebook.hpp"
#include <type_traits>
namespace strata::ds4::row2::iq2 {
namespace e=sycl::ext::intel::esimd;
namespace ex=sycl::ext::intel::experimental::esimd;
using U=e::simd<uint32_t,32>;
using I=e::simd<int32_t,32>;
using F=e::simd<float,32>;
using Index=e::simd<uint16_t,32>;

// Integer reconstruction, including half subnormals/signed zero/Inf/NaN.
// No FP16 arithmetic or host/device conversion intrinsic is substituted.
ESIMD_INLINE F half(U h) {
    U sign=(h&0x8000u)<<16, mant=h&1023u, exp=(h>>10)&31u;
    U norm=sign|((exp+112u)<<23)|(mant<<13);
    norm.merge(sign|0x7f800000u|(mant<<13),exp==31u);
    U submant=mant, subexp=113u;
#pragma unroll
    for(int j=0;j<10;++j) {
        auto shift=(submant!=0u)&((submant&1024u)==0u);
        submant.merge(submant<<1,shift);subexp.merge(subexp-1u,shift);
    }
    U sub=sign|(subexp<<23)|((submant&1023u)<<13);
    sub.merge(sign,mant==0u);norm.merge(sub,exp==0u);
    return norm.template bit_cast_view<float>().read();
}

// Four complete 66B blocks cover one iteration of the 32 original logical lanes.
// Consecutive byte gathers make no dword/block alignment promise (594B down rows).
// The ninth 32B vector is masked at byte264, or earlier at the last whole block.
ESIMD_INLINE e::simd<uint8_t,288> packed(const uint8_t* row,int base,int k) {
    e::simd<uint8_t,288> result=0;
    U byte(0,1);const uint32_t start=uint32_t(base/8)*66u;
    const uint32_t remaining=uint32_t(k/256)*66u-start;
#pragma unroll
    for(int part=0;part<9;++part) {
        U off=byte+uint32_t(part*32);
        auto mask=(off<remaining)&(off<264u);
        auto data=e::gather<uint8_t,32>(row+start,off,mask,e::simd<uint8_t,32>(0));
        result.template select<32,1>(part*32)=data;
    }
    return result;
}

// Contiguous input tile, reused by independent rows in a strip. Only existing
// kb chunks are loaded. No padded input, packed-weight repack, or split-K.
ESIMD_INLINE e::simd<int8_t,1024> input(const int8_t* xq,int base,int k) {
    e::simd<int8_t,1024> result=0;U byte(0,1);
#pragma unroll
    for(int part=0;part<32;++part) {
        U off=byte+uint32_t(part*32);
        auto data=e::gather<int8_t,32>(xq+base*32,off,off<uint32_t(k-base*32),e::simd<int8_t,32>(0));
        result.template select<32,1>(part*32)=data;
    }
    return result;
}

// Load-only candidate. Alignment is checked on the ACTUAL address, including
// column and iteration offsets; malloc_device<int8_t> alone promises no 16B
// alignment. Each transaction is exactly 128 existing bytes. A 256-value down
// tail issues two transactions, never the other six. Misaligned/short chunks
// retain the original masked byte gathers. No weight, scale or math change.
ESIMD_INLINE e::simd<int8_t,1024> input_load(const int8_t* xq,int base,int k) {
    e::simd<int8_t,1024> result=0;
    const int remaining=k-base*32;
    const int8_t* tile=xq+base*32;
    U byte(0,1);
#pragma unroll
    for(int part=0;part<8;++part) {
        const int start=part*128;
        if(start<remaining) {
            const int8_t* ptr=tile+start;
            if(remaining-start>=128 && (reinterpret_cast<uintptr_t>(ptr)&15u)==0u) {
                auto words=e::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(ptr),e::overaligned_tag<16>{});
                result.template select<128,1>(start)=words.template bit_cast_view<int8_t>().read();
            } else {
#pragma unroll
                for(int chunk=0;chunk<4;++chunk) {
                    U off=byte+uint32_t(start+chunk*32);
                    auto data=e::gather<int8_t,32>(tile,off,off<uint32_t(remaining),e::simd<int8_t,32>(0));
                    result.template select<32,1>(start+chunk*32)=data;
                }
            }
        }
    }
    return result;
}

ESIMD_INLINE U parity_sign(U v) {
    v=v&127u;U parity=v^(v>>4);parity=parity^(parity>>2);parity=parity^(parity>>1);
    return v|((parity&1u)<<7);
}

// Every vector element is ONE original logical lane, not one decoded weight.
// Integer products/sums are exact; see iq-port/PROOF.md (including int8 -128).
template<class Grid=uint64_t>
ESIMD_INLINE F dot(e::simd<uint8_t,288> bytes,e::simd<int8_t,1024> x,
                   e::simd<Grid,256>& codebook) {
    static_assert(std::is_same_v<Grid,uint64_t>||std::is_same_v<Grid,uint16_t>);
    U lane(0,1);Index p=(lane/8u)*66u+2u+(lane%8u)*8u;
    U a0=bytes.template iselect<32>(Index(p+4u));
    U a1=bytes.template iselect<32>(Index(p+5u));
    U a2=bytes.template iselect<32>(Index(p+6u));
    U a3=bytes.template iselect<32>(Index(p+7u));
    U aux=a0|(a1<<8)|(a2<<16)|(a3<<24);
    I lo=0,hi=0;
#pragma unroll
    for(int group=0;group<4;++group) {
        Index index=bytes.template iselect<32>(Index(p+uint16_t(group)));
        auto grid=codebook.template iselect<32>(index);
        U signs=parity_sign(aux>>uint32_t(7*group));
#pragma unroll
        for(int j=0;j<8;++j) {
            I q;
            if constexpr(std::is_same_v<Grid,uint16_t>) {
                U c=(grid>>uint32_t(2*j))&uint16_t(3);
                q=8u+17u*c+(c>>1);
            } else q=(grid>>uint32_t(8*j))&uint64_t(255);
            q.merge(-q,((signs>>uint32_t(j))&1u)!=0u);
            I activation=x.template iselect<32>(Index(lane*32u+uint32_t(8*group+j)));
            if(group<2)lo=lo+q*activation;else hi=hi+q*activation;
        }
    }
    I scale=2u*(aux>>28)+1u;
    Index b=(lane/8u)*66u;
    U h0=bytes.template iselect<32>(b),h1=bytes.template iselect<32>(Index(b+1u));
    F d=half(h0|(h1<<8))*0.125f;
    F integer=scale*lo+scale*hi;
    return d*integer; // explicit rounded sub-dot BEFORE the lane FMA
}

ESIMD_INLINE F xor_tree(F a) {
    Index lane(0,1);
#pragma unroll
    for(int offset=16;offset;offset>>=1) {
        F old=a;
        a=old+old.template iselect<32>(Index(lane^uint16_t(offset)));
    }
    return a;
}

template<int Rows,bool Load=false,bool Compact=false>
ESIMD_INLINE void strip(const uint8_t* w,size_t ld,const int8_t* xq,const float* xd,
                        int n,int k,int first,float alpha,float* y) {
    static_assert(Rows==1||Rows==2||Rows==4);
    static_assert(!Load||Rows==1,"load-only arm has exactly one row per work-item");
    static_assert(!Compact||Rows==1,"compact-codebook arm has exactly one row per work-item");
    e::simd<float,32*Rows> accum=0.0f;
    // One immutable original 256-entry table, shared by rows in this work-item.
    using Grid=std::conditional_t<Compact,uint16_t,uint64_t>;
    e::simd<Grid,256> codebook;
#pragma unroll
    for(int i=0;i<256;++i) {
        if constexpr(Compact)codebook[i]=compact_codebook[i];
        else codebook[i]=moe::tables::iq2xxs_grid[i];
    }
    U lane(0,1);
    for(int base=0;base<k/32;base+=32) {
        e::simd<int8_t,1024> x;
        if constexpr(Load)x=input_load(xq,base,k);else x=input(xq,base,k);
        auto active=lane<uint32_t(k/32-base);
        F scales=e::gather<float,32>(xd+base,lane*uint32_t(sizeof(float)),active,F(0));
#pragma unroll
        for(int row=0;row<Rows;++row)if(first+row<n) {
            auto bytes=packed(w+size_t(first+row)*ld,base,k);
            F old=accum.template select<32,1>(row*32);
            F subdot=dot(bytes,x,codebook);
            F next=ex::fma<float,32>(scales,subdot,old);
            old.merge(next,active); // nonexistent kb MUST NOT execute a logical FMA
            accum.template select<32,1>(row*32)=old;
        }
    }
#pragma unroll
    for(int row=0;row<Rows;++row)if(first+row<n) {
        F a=accum.template select<32,1>(row*32);a=xor_tree(a);
        float v=alpha*float(a[0]);y[first+row]=v+0.0f;
    }
}
}
