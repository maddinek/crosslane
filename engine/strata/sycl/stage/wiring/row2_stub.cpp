
// the real implementation of sycl_row2.hpp. Links the SYCL model DSO; any call fails loudly.
#include "sycl_row2.hpp"
namespace strata::ds4::row2 {
namespace { [[noreturn]] void todo(const char* what) { throw std::runtime_error(std::string("TODO(row2): ") + what); } }
DenseMat upload(sycl::queue&, const wiring::WeightSource&, const std::string& name) { todo(("upload " + name).c_str()); }
void release(sycl::queue& q, DenseMat& m) noexcept { if (m.raw) { try { sycl::free(m.raw, q); } catch (...) {} m.raw = nullptr; } }
sycl::event mv(sycl::queue&, std::span<const MvJob>, IqPath) { todo("mv"); }
sycl::event gemv_f32(sycl::queue&, const float*, const float*, int, int, int, float*, int) { todo("gemv_f32"); }
sycl::event gemv_f16_rows(sycl::queue&, const uint16_t*, const float*, size_t, int, int, int, float, float*, size_t) {
    todo("gemv_f16_rows");
}
sycl::event swiglu(sycl::queue&, const float*, const float*, int, float, float*) { todo("swiglu"); }
std::unique_ptr<ExpertCompute> make_experts(sycl::queue&, const wiring::WeightSource&, uint32_t, uint32_t,
                                            const std::string&, int, const std::string&, bool, IqPath) { todo("make_experts"); }
}  // namespace strata::ds4::row2
