#ifndef ConvertToCartesianVectorsCUDAv2_h
#define ConvertToCartesianVectorsCUDAv2_h

namespace CUDAv2 {

  struct CylindricalVector {
    float rho;
    float eta;
    float phi;
  };

  struct CartesianVector {
    float x;
    float y;
    float z;
  };

  void convertWrapper(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size);

}  // namespace CUDAv2

#endif  // ConvertToCartesianVectorsCUDAv2_h
