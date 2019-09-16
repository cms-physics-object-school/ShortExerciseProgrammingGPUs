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

#include "ConvertToCartesianVectorsCUDAv1.h"
using namespace CUDAv1;

class ConvertToCartesianVectorsCUDAv1 : public edm::stream::EDProducer<> {
public:
  explicit ConvertToCartesianVectorsCUDAv1(const edm::ParameterSet&);
  ~ConvertToCartesianVectorsCUDAv1() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<std::vector<math::RhoEtaPhiVectorF>> input_;
};

ConvertToCartesianVectorsCUDAv1::ConvertToCartesianVectorsCUDAv1(const edm::ParameterSet& config)
    : input_(consumes<std::vector<math::RhoEtaPhiVectorF>>(config.getParameter<edm::InputTag>("input"))) {
  produces<std::vector<math::XYZVectorF>>();
}

void ConvertToCartesianVectorsCUDAv1::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<std::vector<math::XYZVectorF>>(elements);

  // allocate memory on the GPU for the cylindrical and cartesian vectors
  CylindricalVector* gpu_input;
  CartesianVector* gpu_product;
  cudaCheck(cudaMalloc(&gpu_input, sizeof(CylindricalVector) * elements));
  cudaCheck(cudaMalloc(&gpu_product, sizeof(CartesianVector) * elements));

  // copy the input data to the GPU
  cudaCheck(cudaMemcpy(gpu_input, input.data(), sizeof(CylindricalVector) * elements, cudaMemcpyHostToDevice));

  // convert the vectors from cylindrical to cartesian coordinates, on the GPU
  convertWrapper(gpu_input, gpu_product, elements);

  // copy the result from the GPU
  cudaCheck(cudaMemcpy(product->data(), gpu_product, sizeof(CartesianVector) * elements, cudaMemcpyDeviceToHost));

  // free the GPU memory
  cudaCheck(cudaFree(gpu_input));
  cudaCheck(cudaFree(gpu_product));

  event.put(std::move(product));
}

void ConvertToCartesianVectorsCUDAv1::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectorsCUDAv1);
