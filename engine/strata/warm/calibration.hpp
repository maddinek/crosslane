// MIT. Recover equal-length pass means from the existing cumulative event API.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace strata::wiring {
constexpr size_t kWarmCalibrationWarmups=2,kWarmCalibrationPasses=3,kWarmCalibrationTokens=32;
inline double warm_median(std::vector<double> values){
 if(values.size()<3)throw std::runtime_error("warm median needs at least three passes");
 for(double n:values)if(!std::isfinite(n)||n<0)throw std::runtime_error("invalid warm calibration cost");
 std::sort(values.begin(),values.end());const auto n=values.size();
 return n%2?values[n/2]:(values[n/2-1]+values[n/2])/2;
}
struct WarmCalibration {
 std::vector<std::vector<double>> passes;
 std::vector<double> previous;
 void add(const std::vector<double>& cumulative){
  if(cumulative.empty()||(!previous.empty()&&previous.size()!=cumulative.size()))throw std::runtime_error("warm calibration shape");
  const auto count=passes.size();std::vector<double> pass(cumulative.size());
  for(size_t i=0;i<pass.size();++i){
   pass[i]=cumulative[i]*double(count+1)-(count?previous[i]*double(count):0);
   if(!std::isfinite(cumulative[i])||!std::isfinite(pass[i])||pass[i]<0)throw std::runtime_error("invalid warm pass cost");
  }
  passes.push_back(std::move(pass));previous=cumulative;
 }
 std::vector<double> medians()const{
  if(passes.size()<kWarmCalibrationWarmups+3)throw std::runtime_error("insufficient warm calibration passes");
  std::vector<double> result(previous.size());
  for(size_t i=0;i<result.size();++i){std::vector<double> values;for(size_t p=kWarmCalibrationWarmups;p<passes.size();++p)values.push_back(passes[p][i]);result[i]=warm_median(std::move(values));}
  return result;
 }
};
}
