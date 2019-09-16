// system include files
#include <cmath>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "ConvertToCartesianVectorsCUDAv2.h"

namespace CUDAv2 {

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
    auto blockSize = 512;                                // somewhat arbitrary for the moment
    auto gridSize = (size + blockSize - 1) / blockSize;  // round up to cover the sample size
    convertKernel<<<gridSize, blockSize>>>(cylindrical, cartesian, size);
    cudaCheck(cudaGetLastError());
  }

}  // namespace CUDAv2
