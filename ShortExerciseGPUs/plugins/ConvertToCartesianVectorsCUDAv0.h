#ifndef ConvertToCartesianVectorsCUDAv0_h
#define ConvertToCartesianVectorsCUDAv0_h

namespace CUDAv0 {

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

}  // namespace CUDAv0

#endif  // ConvertToCartesianVectorsCUDAv0_h
