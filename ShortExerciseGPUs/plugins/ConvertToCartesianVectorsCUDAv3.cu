// system include files
#include <cmath>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "ConvertToCartesianVectorsCUDAv3.h"

namespace CUDAv3 {

  __global__ void convertKernel(CylindricalVectorSoA cylindrical, CartesianVectorSoA cartesian, size_t size) {
    auto firstElement = threadIdx.x + blockIdx.x * blockDim.x;
    auto gridSize = blockDim.x * gridDim.x;

    for (size_t i = firstElement; i < size; i += gridSize) {
      cartesian.x[i] = cylindrical.rho[i] * std::cos(cylindrical.phi[i]);
      cartesian.y[i] = cylindrical.rho[i] * std::sin(cylindrical.phi[i]);
      cartesian.z[i] = cylindrical.rho[i] * std::sinh(cylindrical.eta[i]);
    }
  }

  void convertWrapper(CylindricalVectorSoA const& cylindrical, CartesianVectorSoA const& cartesian, size_t size) {
    auto blockSize = 512;                                // somewhat arbitrary for the moment
    auto gridSize = (size + blockSize - 1) / blockSize;  // round up to cover the sample size
    convertKernel<<<gridSize, blockSize>>>(cylindrical, cartesian, size);
    cudaCheck(cudaGetLastError());
  }

}  // namespace CUDAv3
