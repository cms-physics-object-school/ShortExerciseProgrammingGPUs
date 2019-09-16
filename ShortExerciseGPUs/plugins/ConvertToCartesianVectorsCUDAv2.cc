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

#include "ConvertToCartesianVectorsCUDAv2.h"
using namespace CUDAv2;

class ConvertToCartesianVectorsCUDAv2 : public edm::stream::EDProducer<> {
public:
  explicit ConvertToCartesianVectorsCUDAv2(const edm::ParameterSet&);
  ~ConvertToCartesianVectorsCUDAv2();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<std::vector<math::RhoEtaPhiVectorF>> input_;
  CylindricalVector* gpu_input_;
  CartesianVector* gpu_product_;
  uint32_t elements_;
};

ConvertToCartesianVectorsCUDAv2::ConvertToCartesianVectorsCUDAv2(const edm::ParameterSet& config)
    : input_(consumes<std::vector<math::RhoEtaPhiVectorF>>(config.getParameter<edm::InputTag>("input"))),
      elements_(config.getParameter<uint32_t>("initialSize")) {
  produces<std::vector<math::XYZVectorF>>();

  // allocate memory on the GPU for the cylindrical and cartesian vectors
  cudaCheck(cudaMalloc(&gpu_input_, sizeof(CylindricalVector) * elements_));
  cudaCheck(cudaMalloc(&gpu_product_, sizeof(CartesianVector) * elements_));
}

ConvertToCartesianVectorsCUDAv2::~ConvertToCartesianVectorsCUDAv2() {
  // free the GPU memory
  cudaCheck(cudaFree(gpu_input_));
  cudaCheck(cudaFree(gpu_product_));
}

void ConvertToCartesianVectorsCUDAv2::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<std::vector<math::XYZVectorF>>(elements);

  // if necessary, reallocate memory on the GPU for the cylindrical and cartesian vectors
  if (elements > elements_) {
    elements_ = elements;
    cudaCheck(cudaFree(gpu_input_));
    cudaCheck(cudaFree(gpu_product_));
    cudaCheck(cudaMalloc(&gpu_input_, sizeof(CylindricalVector) * elements_));
    cudaCheck(cudaMalloc(&gpu_product_, sizeof(CartesianVector) * elements_));
  }

  // copy the input data to the GPU
  cudaCheck(cudaMemcpy(gpu_input_, input.data(), sizeof(CylindricalVector) * elements, cudaMemcpyHostToDevice));

  // convert the vectors from cylindrical to cartesian coordinates, on the GPU
  convertWrapper(gpu_input_, gpu_product_, elements);

  // copy the result from the GPU
  cudaCheck(cudaMemcpy(product->data(), gpu_product_, sizeof(CartesianVector) * elements, cudaMemcpyDeviceToHost));

  event.put(std::move(product));
}

void ConvertToCartesianVectorsCUDAv2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  desc.add<uint32_t>("initialSize", 1000);
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectorsCUDAv2);
