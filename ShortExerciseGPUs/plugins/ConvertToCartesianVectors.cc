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

class ConvertToCartesianVectors : public edm::stream::EDProducer<> {
public:
  explicit ConvertToCartesianVectors(const edm::ParameterSet&);
  ~ConvertToCartesianVectors() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<std::vector<math::RhoEtaPhiVectorF>> input_;
};

ConvertToCartesianVectors::ConvertToCartesianVectors(const edm::ParameterSet& config)
    : input_(consumes<std::vector<math::RhoEtaPhiVectorF>>(config.getParameter<edm::InputTag>("input"))) {
  produces<std::vector<math::XYZVectorF>>();
}

void ConvertToCartesianVectors::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<std::vector<math::XYZVectorF>>(elements);

  // convert the vectors from cylindrical to cartesian coordinates
  for (unsigned int i = 0; i < elements; ++i) {
    (*product)[i].SetCoordinates(input[i].rho() * std::cos(input[i].phi()),
                                 input[i].rho() * std::sin(input[i].phi()),
                                 input[i].rho() * std::sinh(input[i].eta()));
  }

  event.put(std::move(product));
}

void ConvertToCartesianVectors::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectors);
