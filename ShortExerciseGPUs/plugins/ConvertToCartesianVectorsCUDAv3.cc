// system include files
#include <cmath>
#include <memory>
#include <vector>

// CMSSW include files
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "ConvertToCartesianVectorsCUDAv3.h"
using namespace CUDAv3;

class ConvertToCartesianVectorsCUDAv3 : public edm::stream::EDProducer<> {
public:
  explicit ConvertToCartesianVectorsCUDAv3(const edm::ParameterSet&);
  ~ConvertToCartesianVectorsCUDAv3();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<std::vector<math::RhoEtaPhiVectorF>> input_;
  CylindricalVectorSoA cpu_input_;
  CylindricalVectorSoA gpu_input_;
  CartesianVectorSoA cpu_product_;
  CartesianVectorSoA gpu_product_;
  uint32_t elements_;
};

ConvertToCartesianVectorsCUDAv3::ConvertToCartesianVectorsCUDAv3(const edm::ParameterSet& config)
    : input_(consumes<std::vector<math::RhoEtaPhiVectorF>>(config.getParameter<edm::InputTag>("input"))),
      elements_(config.getParameter<uint32_t>("initialSize")) {
  produces<std::vector<math::XYZVectorF>>();

  // allocate memory on the CPU for the cylindrical and cartesian vectors
  cudaCheck(cudaMallocHost(&cpu_input_.rho, sizeof(float) * elements_));
  cudaCheck(cudaMallocHost(&cpu_input_.eta, sizeof(float) * elements_));
  cudaCheck(cudaMallocHost(&cpu_input_.phi, sizeof(float) * elements_));
  cudaCheck(cudaMallocHost(&cpu_product_.x, sizeof(float) * elements_, cudaHostAllocWriteCombined));
  cudaCheck(cudaMallocHost(&cpu_product_.y, sizeof(float) * elements_, cudaHostAllocWriteCombined));
  cudaCheck(cudaMallocHost(&cpu_product_.z, sizeof(float) * elements_, cudaHostAllocWriteCombined));

  // allocate memory on the GPU for the cylindrical and cartesian vectors
  cudaCheck(cudaMalloc(&gpu_input_.rho, sizeof(float) * elements_));
  cudaCheck(cudaMalloc(&gpu_input_.eta, sizeof(float) * elements_));
  cudaCheck(cudaMalloc(&gpu_input_.phi, sizeof(float) * elements_));
  cudaCheck(cudaMalloc(&gpu_product_.x, sizeof(float) * elements_));
  cudaCheck(cudaMalloc(&gpu_product_.y, sizeof(float) * elements_));
  cudaCheck(cudaMalloc(&gpu_product_.z, sizeof(float) * elements_));
}

ConvertToCartesianVectorsCUDAv3::~ConvertToCartesianVectorsCUDAv3() {
  // free the CPU memory
  cudaCheck(cudaFreeHost(cpu_input_.rho));
  cudaCheck(cudaFreeHost(cpu_input_.eta));
  cudaCheck(cudaFreeHost(cpu_input_.phi));
  cudaCheck(cudaFreeHost(cpu_product_.x));
  cudaCheck(cudaFreeHost(cpu_product_.y));
  cudaCheck(cudaFreeHost(cpu_product_.z));

  // free the GPU memory
  cudaCheck(cudaFree(gpu_input_.rho));
  cudaCheck(cudaFree(gpu_input_.eta));
  cudaCheck(cudaFree(gpu_input_.phi));
  cudaCheck(cudaFree(gpu_product_.x));
  cudaCheck(cudaFree(gpu_product_.y));
  cudaCheck(cudaFree(gpu_product_.z));
}

void ConvertToCartesianVectorsCUDAv3::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<std::vector<math::XYZVectorF>>(elements);

  // if necessary, reallocate memory on the GPU for the cylindrical and cartesian vectors
  if (elements > elements_) {
    elements_ = elements;

    // free the CPU memory
    cudaCheck(cudaFreeHost(cpu_input_.rho));
    cudaCheck(cudaFreeHost(cpu_input_.eta));
    cudaCheck(cudaFreeHost(cpu_input_.phi));
    cudaCheck(cudaFreeHost(cpu_product_.x));
    cudaCheck(cudaFreeHost(cpu_product_.y));
    cudaCheck(cudaFreeHost(cpu_product_.z));

    // free the GPU memory
    cudaCheck(cudaFree(gpu_input_.rho));
    cudaCheck(cudaFree(gpu_input_.eta));
    cudaCheck(cudaFree(gpu_input_.phi));
    cudaCheck(cudaFree(gpu_product_.x));
    cudaCheck(cudaFree(gpu_product_.y));
    cudaCheck(cudaFree(gpu_product_.z));

    // allocate memory on the CPU for the cylindrical and cartesian vectors
    cudaCheck(cudaMallocHost(&cpu_input_.rho, sizeof(float) * elements_));
    cudaCheck(cudaMallocHost(&cpu_input_.eta, sizeof(float) * elements_));
    cudaCheck(cudaMallocHost(&cpu_input_.phi, sizeof(float) * elements_));
    cudaCheck(cudaMallocHost(&cpu_product_.x, sizeof(float) * elements_, cudaHostAllocWriteCombined));
    cudaCheck(cudaMallocHost(&cpu_product_.y, sizeof(float) * elements_, cudaHostAllocWriteCombined));
    cudaCheck(cudaMallocHost(&cpu_product_.z, sizeof(float) * elements_, cudaHostAllocWriteCombined));

    // allocate memory on the GPU for the cylindrical and cartesian vectors
    cudaCheck(cudaMalloc(&gpu_input_.rho, sizeof(float) * elements_));
    cudaCheck(cudaMalloc(&gpu_input_.eta, sizeof(float) * elements_));
    cudaCheck(cudaMalloc(&gpu_input_.phi, sizeof(float) * elements_));
    cudaCheck(cudaMalloc(&gpu_product_.x, sizeof(float) * elements_));
    cudaCheck(cudaMalloc(&gpu_product_.y, sizeof(float) * elements_));
    cudaCheck(cudaMalloc(&gpu_product_.z, sizeof(float) * elements_));
  }

  // copy the input data from the CMSSW product to the CPU buffer
  for (uint32_t i = 0; i < elements_; ++i) {
    cpu_input_.rho[i] = input[i].rho();
    cpu_input_.eta[i] = input[i].eta();
    cpu_input_.phi[i] = input[i].phi();
  }

  // copy the input data to the GPU
  cudaCheck(cudaMemcpy(gpu_input_.rho, cpu_input_.rho, sizeof(float) * elements_, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(gpu_input_.eta, cpu_input_.eta, sizeof(float) * elements_, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(gpu_input_.phi, cpu_input_.phi, sizeof(float) * elements_, cudaMemcpyHostToDevice));

  // convert the vectors from cylindrical to cartesian coordinates, on the GPU
  convertWrapper(gpu_input_, gpu_product_, elements_);

  // copy the result from the GPU
  cudaCheck(cudaMemcpy(cpu_product_.x, gpu_product_.x, sizeof(float) * elements_, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(cpu_product_.y, gpu_product_.y, sizeof(float) * elements_, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(cpu_product_.z, gpu_product_.z, sizeof(float) * elements_, cudaMemcpyDeviceToHost));

  // copy the result from the CPU buffer to the CMSSW product
  for (uint32_t i = 0; i < elements_; ++i) {
    (*product)[i].SetCoordinates(cpu_product_.x[i], cpu_product_.y[i], cpu_product_.z[i]);
  }

  event.put(std::move(product));
}

void ConvertToCartesianVectorsCUDAv3::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  desc.add<uint32_t>("initialSize", 1000);
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectorsCUDAv3);
