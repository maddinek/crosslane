// Strata c62d5e3 / ggml layouts, MIT; licenses/ contains the upstream notices.
#pragma once
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <string>
namespace strata::ds4::moe {
enum class Type { bf16, f32, f16, q8_0, iq1_m, iq1_s, iq2_s, iq2_xs, iq2_xxs, iq4_nl, q4_k, q5_k, q6_k };
inline Type type(std::string_view s) {
    if(s=="BF16") return Type::bf16;
    if(s=="F32") return Type::f32;
    if(s=="F16") return Type::f16;
    if(s=="Q8_0") return Type::q8_0;
    if(s=="IQ1_M") return Type::iq1_m;
    if(s=="IQ2_XS") return Type::iq2_xs;
    if(s=="IQ1_S") return Type::iq1_s;
    if(s=="IQ2_S") return Type::iq2_s;
    if(s=="IQ2_XXS") return Type::iq2_xxs;
    if(s=="IQ4_NL") return Type::iq4_nl;
    if(s=="Q4_K") return Type::q4_k;
    if(s=="Q5_K") return Type::q5_k;
    if(s=="Q6_K") return Type::q6_k;
    throw std::invalid_argument("row2 unsupported dtype: "+std::string(s));
}
inline int ggml_type(Type t) {
    switch(t) {case Type::q4_k:return 12;case Type::q5_k:return 13;case Type::q6_k:return 14;
    case Type::iq1_s:return 19;case Type::iq2_s:return 22;case Type::iq2_xxs:return 16;case Type::iq2_xs:return 17;case Type::iq4_nl:return 20;case Type::iq1_m:return 29;
    default:throw std::invalid_argument("row2 CPU expert requires an admitted ggml block type");}
}
inline int block_values(Type t) {
    switch(t) {case Type::bf16:case Type::f16:case Type::f32:return 1;
    case Type::q8_0:case Type::iq4_nl:return 32;default:return 256;}
}
inline size_t block_bytes(Type t) {
    switch(t) {case Type::bf16:case Type::f16:return 2;case Type::f32:return 4;case Type::q8_0:return 34;
    case Type::iq1_s:return 50;case Type::iq2_s:return 82;case Type::iq1_m:return 56;case Type::iq2_xs:return 74;case Type::iq2_xxs:return 66;case Type::iq4_nl:return 18;
    case Type::q4_k:return 144;case Type::q5_k:return 176;case Type::q6_k:return 210;}
    return 0; // all callers validate Type before device submission
}
inline size_t row_bytes(Type t,int k) {
    if(k<=0 || k%block_values(t)) throw std::invalid_argument("row2 partial weight block");
    return size_t(k/block_values(t))*block_bytes(t);
}
inline size_t product(size_t a,size_t b) {
    if(b && a>std::numeric_limits<size_t>::max()/b) throw std::invalid_argument("row2 extent overflow");
    return a*b;
}
inline uint16_t u16(const uint8_t* p) {return uint16_t(p[0]) | uint16_t(p[1])<<8;}
inline uint32_t u32(const uint8_t* p) {return uint32_t(u16(p)) | uint32_t(u16(p+2))<<16;}
inline float half(uint16_t h) {
    uint32_t s=uint32_t(h&0x8000)<<16, m=h&1023, e=(h>>10)&31;
    if(!e) {if(!m)return std::bit_cast<float>(s);e=113;while(!(m&1024)){m<<=1;--e;}m&=1023;}
    else e=e==31?255:e+112;
    return std::bit_cast<float>(s|(e<<23)|(m<<13));
}
inline float bf16(uint16_t h) {return std::bit_cast<float>(uint32_t(h)<<16);}
}
