// MIT; crosslane contributors. Explicit opt-in; no environment dispatch.
#pragma once
#include <stdexcept>
#include <string_view>
namespace strata::ds4::row2 {
enum class IqPath { control, esimd1, esimd2, esimd4, spmd4, spmd8, esimd1_load, esimd1_cb, esimd1_load_cb, esimd1_load_cb_grf256 };
inline IqPath iq_path(std::string_view name) {
    if(name=="control")return IqPath::control;
    if(name=="esimd1")return IqPath::esimd1;
    if(name=="esimd1-load")return IqPath::esimd1_load;
    if(name=="esimd1-cb")return IqPath::esimd1_cb;
    if(name=="esimd1-load-cb")return IqPath::esimd1_load_cb;
    if(name=="esimd1-load-cb-grf256")return IqPath::esimd1_load_cb_grf256;
    if(name=="esimd2")return IqPath::esimd2;
    if(name=="esimd4")return IqPath::esimd4;
    if(name=="spmd4")return IqPath::spmd4;
    if(name=="spmd8")return IqPath::spmd8;
    throw std::invalid_argument("unknown row2 IQ path");
}
inline void check_iq_path(IqPath p) {
    switch(p) {case IqPath::control:case IqPath::esimd1:case IqPath::esimd2:
    case IqPath::esimd4:case IqPath::spmd4:case IqPath::spmd8:case IqPath::esimd1_load:
    case IqPath::esimd1_cb:case IqPath::esimd1_load_cb:case IqPath::esimd1_load_cb_grf256:return;}
    throw std::invalid_argument("invalid row2 IQ path");
}
}
