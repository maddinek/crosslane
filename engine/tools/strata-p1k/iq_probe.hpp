// MIT; deterministic correctness-only inputs. No timing-input substitution.
#pragma once
#include <vector>
#include <stdexcept>
#include <string_view>
inline std::vector<float> iq_probe(const std::vector<float>& original,std::string_view kind) {
    if(kind=="data")return original;
    if(kind!="zero"&&kind!="tie")throw std::invalid_argument("unknown IQ correctness input");
    std::vector<float> x(original.size(),0.0f);
    if(kind=="tie")for(size_t i=0;i<x.size();++i)
        x[i]=i%32==0?127.0f:float(int(i%31)-15)+0.5f;
    // Every32-value block has absmax127 => Q8 scale1 and exact half-integer ties.
    return x;
}
