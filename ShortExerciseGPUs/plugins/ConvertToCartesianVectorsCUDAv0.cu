// system include files
#include <cmath>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "ConvertToCartesianVectorsCUDAv0.h"

namespace CUDAv0 {

  __global__ void convertKernel(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    auto firstElement = threadIdx.x + blockIdx.x * blockDim.x;
    auto gridSize = blockDim.x * gridDim.x;

    for (size_t i = firstElement; i < size; i += gridSize) {
      cartesian[i].x = cylindrical[i].rho * std::cos(cylindrical[i].phi);
      cartesian[i].y = cylindrical[i].rho * std::sin(cylindrical[i].phi);
      cartesian[i].z = cylindrical[i].rho * std::sinh(cylindrical[i].eta);
    }
  }

  void convertWrapper(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    convertKernel<<<1, 1>>>(cylindrical, cartesian, size);
    cudaCheck(cudaGetLastError());
  }

}  // namespace CUDAv0
