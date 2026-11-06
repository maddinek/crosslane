// MIT; exact register encoding of the original immutable IQ2_XXS magnitudes.
#pragma once
#include <array>
#include "tables.hpp"
namespace strata::ds4::row2::iq2 {
constexpr uint16_t compact_entry(uint64_t grid) {
    uint16_t result=0;
    for(int j=0;j<8;++j) {
        unsigned value=unsigned((grid>>(8*j))&255u);
        // Invalid upstream magnitudes fail constant evaluation; never truncate.
        if(value!=8&&value!=25&&value!=43)throw "unknown IQ2_XXS magnitude";
        unsigned symbol=value==8?0:value==25?1:2;
        result=uint16_t(result|(symbol<<(2*j)));
    }
    return result;
}
inline constexpr auto compact_codebook=[] {
    std::array<uint16_t,256> result{};
    for(int i=0;i<256;++i)result[i]=compact_entry(moe::tables::iq2xxs_grid[i]);
    return result;
}();
static_assert(sizeof(compact_codebook)==512);
static_assert([] {
    for(int i=0;i<256;++i)for(int j=0;j<8;++j) {
        unsigned c=(compact_codebook[i]>>(2*j))&3u;
        if(c>2||8+17*c+(c>>1)!=((moe::tables::iq2xxs_grid[i]>>(8*j))&255u))return false;
    }
    return true;
}(),"all 256x8 compact magnitudes must match the original table");
}
