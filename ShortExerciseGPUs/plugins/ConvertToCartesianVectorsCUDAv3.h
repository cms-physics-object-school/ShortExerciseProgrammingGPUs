#ifndef ConvertToCartesianVectorsCUDAv3_h
#define ConvertToCartesianVectorsCUDAv3_h

namespace CUDAv3 {

  struct CylindricalVectorSoA {
    float* rho;
    float* eta;
    float* phi;
  };

  struct CartesianVectorSoA {
    float* x;
    float* y;
    float* z;
  };

  void convertWrapper(CylindricalVectorSoA const& cylindrical, CartesianVectorSoA const& cartesian, size_t size);

}  // namespace CUDAv3

#endif  // ConvertToCartesianVectorsCUDAv3_h
