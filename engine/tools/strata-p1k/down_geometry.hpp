// MIT; microbenchmark-only geometry/table delivery. Default engine untouched.
#pragma once
#include "engine/strata/sycl/moe/iq2_esimd.hpp"
namespace p1k_down {
using namespace strata::ds4::row2::iq2;
namespace m=strata::ds4::moe;
namespace r=strata::ds4::row2;
// IQ1 table's eight ternary values are packed as eight 2-bit symbols.
inline constexpr auto iq1_compact=[] {
    std::array<uint16_t,2048> result{};
    for(int i=0;i<2048;++i)for(int j=0;j<8;++j){
        unsigned c=(m::tables::iq1s_grid_gpu[i]>>(8*(j&3)+(j>=4?4:0)))&15u;
        if(c>2)throw "non-ternary IQ1 grid";
        result[i]=uint16_t(result[i]|(c<<(2*j)));
    }return result;
}();
static_assert([] {for(int i=0;i<2048;++i)for(int j=0;j<8;++j)
    if(((iq1_compact[i]>>(2*j))&3u)!=((m::tables::iq1s_grid_gpu[i]>>(8*(j&3)+(j>=4?4:0)))&15u))return false;
    return true;}());
// Preserve the entire original runtime format dispatch in both geometry arms.
// Only IQ1_M's table representation/delivery changes in decode_cached.
namespace scalar {
using namespace strata::ds4::moe;
inline Sub decode_cached(Type t,const uint8_t* row,int kb,const uint16_t* table) {
    const int ib=kb&7;
    const auto* b=row+size_t(block_values(t)==32?kb:kb/8)*block_bytes(t);
    Sub s;
    if(t==Type::q8_0){s.d=half(u16(b));for(int j=0;j<32;++j)s.q[j]=int8_t(b[2+j]);}
    else if(t==Type::iq4_nl){s.d=half(u16(b));for(int j=0;j<32;++j)s.q[j]=tables::kvalues_iq4nl[j<16?b[2+j]&15:b[2+j-16]>>4];}
    else if(t==Type::iq2_xxs || t==Type::iq2_xs) {
        const auto* p=b+2+8*ib;const unsigned aux=u32(p+4);
        for(int l=0;l<4;++l) {
            const unsigned v=u16(p+2*l);
            const uint64_t grid=t==Type::iq2_xxs?tables::iq2xxs_grid[p[l]]:tables::iq2xs_grid[v&511];
            const unsigned sg=signs(t==Type::iq2_xxs?aux>>(7*l):v>>9);
            for(int j=0;j<8;++j){int q=int((grid>>(8*j))&255);s.q[8*l+j]=(sg>>j)&1?-q:q;}
        }
        s.d=half(u16(b))*0.125f;
        if(t==Type::iq2_xxs)s.s[0]=s.s[1]=2*int(aux>>28)+1;
        else {s.s[0]=2*int(b[66+ib]&15)+1;s.s[1]=2*int(b[66+ib]>>4)+1;}
    } else if(t==Type::iq2_s) {
        s.d=half(u16(b))*0.125f;s.s[0]=2*int(b[74+ib]&15)+1;s.s[1]=2*int(b[74+ib]>>4)+1;
        for(int l=0;l<4;++l){unsigned idx=b[2+4*ib+l]|((unsigned(b[66+ib])<<(8-2*l))&0x300);
            auto grid=tables::iq2s_grid[idx];unsigned sg=b[34+4*ib+l];
            for(int j=0;j<8;++j){int q=int((grid>>(8*j))&255);s.q[8*l+j]=(sg>>j)&1?-q:q;}}
    } else if(t==Type::iq1_s) {
        unsigned h=u16(b+34+2*ib);s.d=half(u16(b))*0.125f;s.s[0]=s.s[1]=2*int((h>>12)&7)+1;
        for(int l=0;l<4;++l){unsigned idx=b[2+4*ib+l]|(((h>>(3*l))&7)<<8);unsigned grid=tables::iq1s_grid_gpu[idx];
            for(int j=0;j<8;++j){int q=int((grid>>(8*(j&3)+(j>=4?4:0)))&15);s.q[8*l+j]=8*(q-1)+((h&0x8000)?-1:1);}}
    } else if(t==Type::iq1_m) {
        const unsigned sc=u16(b+48+2*(ib>>1));
        const uint16_t d=uint16_t((u16(b+48)>>12)|((u16(b+50)>>8)&0xf0)|((u16(b+52)>>4)&0xf00)|(u16(b+54)&0xf000));
        s.d=half(d)*0.125f;s.s[0]=2*int((sc>>(6*(ib&1)))&7)+1;s.s[1]=2*int((sc>>(6*(ib&1)+3))&7)+1;
        for(int l=0;l<4;++l){unsigned h=b[32+2*ib+l/2];unsigned idx=b[4*ib+l]|((h<<(l&1?4:8))&0x700);
            uint16_t grid=table[idx];int delta=(h&(l&1?128:8))?-1:1;
            for(int j=0;j<8;++j){int q=int((grid>>(2*j))&3);s.q[8*l+j]=8*(q-1)+delta;}
        }
    } else if(t==Type::q4_k || t==Type::q5_k) {
        int d,m;scale_min(b+4,ib,d,m);s.s[0]=s.s[1]=d;s.d=half(u16(b));s.m[0]=s.m[1]=half(u16(b+2))*float(m);
        const auto* lo=b+(t==Type::q5_k?48:16)+32*(ib>>1);
        for(int j=0;j<32;++j){s.q[j]=(lo[j]>>(4*(ib&1)))&15;if(t==Type::q5_k)s.q[j]|=((b[16+j]>>ib)&1)<<4;}
    } else if(t==Type::q6_k) {
        int h=ib>>2,quarter=ib&3;const auto* lo=b+64*h+(quarter&1?32:0);
        for(int j=0;j<32;++j)s.q[j]=int(((lo[j]>>(4*(quarter>>1)))&15)|(((b[128+32*h+j]>>(2*quarter))&3)<<4))-32;
        s.s[0]=int8_t(b[192+8*h+2*quarter]);s.s[1]=int8_t(b[193+8*h+2*quarter]);s.d=half(u16(b+208));
    }
    return s;
}
template<class M> inline float lane_cached(Type t,const uint8_t* w,int k,const float* x,const int8_t* xq,const float* xd,int l,bool vec,const uint16_t* table) {
    float a=0;
    if(t==Type::bf16) {
        if(vec)for(int i=8*l;i<k;i+=256)for(int j=0;j<8;++j)a=M::fma(bf16(u16(w+2*(i+j))),x[i+j],a);
        else for(int i=l;i<k;i+=32)a=M::fma(bf16(u16(w+2*i)),x[i],a);
    } else if(t==Type::f32) {
        for(int i=4*l;i<k;i+=128)for(int j=0;j<4;++j)a=M::fma(std::bit_cast<float>(u32(w+4*(i+j))),x[i+j],a);
    } else if(t==Type::f16) {
        // HIP's `acc += f.x*x0 + f.y*x1`: contracts the inner expression then the outer add.
        for(int i=8*l;i<k;i+=256)for(int j=0;j<8;j+=2){float hi=half(u16(w+2*(i+j+1)))*x[i+j+1];a=a+M::fma(half(u16(w+2*(i+j))),x[i+j],hi);}
    } else for(int kb=l;kb<k/32;kb+=32){Sub s=decode_cached(t,w,kb,table);
        if(t==Type::q8_0){int dot=0;for(int j=0;j<32;++j)dot+=s.q[j]*xq[32*kb+j];a=M::fma(s.d*xd[kb],float(dot),a);}
        else {float dot=sub_dot<M>(s,xq+32*kb,t==Type::q4_k || t==Type::q5_k);a=M::fma(xd[kb],dot,a);}
    }
    return a;
}
}
template<class Math> inline float iq1_lane_cached(const uint8_t* row,int k,
        const int8_t* xq,const float* xd,int lane,const uint16_t* table) {
    return scalar::lane_cached<Math>(m::Type::iq1_m,row,k,nullptr,xq,xd,lane,true,table);
}
ESIMD_INLINE F cached_dot(e::simd<uint8_t,288> bytes,e::simd<int8_t,1024> x,
                   const uint16_t* codebook) {
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
        #ifdef STRATA_IQ2_HOST_TEST
        auto grid=e::gather<uint16_t,32>(codebook,U(index)*2u,e::simd<uint16_t,32>(1),e::simd<uint16_t,32>(0));
#else
        auto grid=e::gather<uint16_t,32>(codebook,U(index)*2u,
            e::properties{e::cache_hint_L1<e::cache_hint::cached>,e::cache_hint_L2<e::cache_hint::cached>});
#endif
        U signs=parity_sign(aux>>uint32_t(7*group));
#pragma unroll
        for(int j=0;j<8;++j) {
            I q;
            {
                U c=(grid>>uint32_t(2*j))&uint16_t(3);
                q=8u+17u*c+(c>>1);
            }
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

ESIMD_INLINE void cached_strip(const uint8_t* w,size_t ld,const int8_t* xq,
        const float* xd,int k,int row,float* y,const uint16_t* table) {
    F acc=0.0f;U lane(0,1);
    for(int base=0;base<k/32;base+=32){
        auto x=input_load(xq,base,k);auto active=lane<uint32_t(k/32-base);
        F scales=e::gather<float,32>(xd+base,lane*uint32_t(sizeof(float)),active,F(0));
        auto bytes=packed(w+size_t(row)*ld,base,k);
        F dot=cached_dot(bytes,x,table);
        F next=ex::fma<float,32>(scales,dot,acc);acc.merge(next,active);
    }
    acc=xor_tree(acc);float value=1.0f*float(acc[0]);y[row]=value+0.0f;
}
#if !defined(P1K_DOWN_HOST_TEST) && !defined(STRATA_WARM_PACKED)
template<bool Cached>
sycl::event iq2_local4(sycl::queue& q,const uint8_t* w,size_t ld,int n,int k,
        float* y,const int8_t* xq,const float* xd,const uint16_t* table) {
    return q.submit([&](sycl::handler& cgh){
        cgh.parallel_for(sycl::nd_range<1>((size_t(n)+3)/4*4,4),
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                int row=int(it.get_group_linear_id())*4+int(it.get_local_linear_id());if(row>=n)return;
                if constexpr(Cached)cached_strip(w,ld,xq,xd,k,row,y,table);
                else strip<1,true,true>(w,ld,xq,xd,n,k,row,1.0f,y);
            });
    });
}
template<bool Cached>
sycl::event iq1_local4(sycl::queue& q,m::Type type,const uint8_t* w,size_t ld,int n,int k,
        float* y,const int8_t* xq,const float* xd,const uint16_t* table) {
    // Four original subgroup32 row threads, not four SPMD lanes total.
    return q.parallel_for(sycl::nd_range<1>((size_t(n)+3)/4*128,128),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            int row=int(it.get_group_linear_id())*4+int(it.get_local_linear_id()/32);
            int lane=int(it.get_local_linear_id()%32);if(row>=n)return;
            float a;
            if constexpr(Cached)a=scalar::lane_cached<r::Math>(type,w+size_t(row)*ld,k,nullptr,xq,xd,lane,true,table);
            else a=m::lane<r::Math>(type,w+size_t(row)*ld,k,nullptr,xq,xd,lane,true);
            auto sg=it.get_sub_group();for(int o=16;o;o>>=1)a+=sycl::permute_group_by_xor(sg,a,o);
            if(!lane){float value=1.0f*a;y[row]=value+0.0f;}
        });
}
inline sycl::event launch(sycl::queue& q,std::string_view mode,m::Type type,
        const uint8_t* w,size_t ld,int n,int k,float* y,const int8_t* xq,const float* xd,
        const uint16_t* iq2_table,const uint16_t* iq1_table) {
    if(type!=m::Type::iq2_xxs&&type!=m::Type::iq1_m)throw std::invalid_argument("down micro quant");
    if(mode=="down-range")return r::launch(q,type,w,ld,nullptr,k,n,k,1,1.0f,y,n,xq,xd,true,r::IqPath::esimd1_load_cb_grf256);
    if(mode!="down-local4"&&mode!="down-cached")throw std::invalid_argument("down micro mode");
    bool cached=mode=="down-cached";
    if(type==m::Type::iq2_xxs)return cached?iq2_local4<true>(q,w,ld,n,k,y,xq,xd,iq2_table):iq2_local4<false>(q,w,ld,n,k,y,xq,xd,iq2_table);
    return cached?iq1_local4<true>(q,type,w,ld,n,k,y,xq,xd,iq1_table):iq1_local4<false>(q,type,w,ld,n,k,y,xq,xd,iq1_table);
}
#endif
}
