#ifndef ConvertToCartesianVectorsCUDAv1_h
#define ConvertToCartesianVectorsCUDAv1_h

namespace CUDAv1 {

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

}  // namespace CUDAv1

#endif  // ConvertToCartesianVectorsCUDAv1_h
