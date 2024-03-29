// system include files
#include <memory>
#include <random>
#include <vector>

// CMSSW include files
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

class GenerateCylindricalVectors : public edm::stream::EDProducer<> {
public:
  explicit GenerateCylindricalVectors(const edm::ParameterSet&);
  ~GenerateCylindricalVectors() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginStream(edm::StreamID) override{};
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override{};

  std::mt19937 generator_;
  std::lognormal_distribution<float> genPt_;
  std::uniform_real_distribution<float> genEta_;
  std::uniform_real_distribution<float> genPhi_;
  uint32_t size_;
};

GenerateCylindricalVectors::GenerateCylindricalVectors(const edm::ParameterSet& config)
    : generator_(std::random_device()()),
      genPt_(3, 0.6),
      genEta_(-5., +5.),
      genPhi_(0., 2 * M_PI),
      size_(config.getParameter<uint32_t>("size")) {
  produces<std::vector<math::RhoEtaPhiVectorF>>();
}

void GenerateCylindricalVectors::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto product = std::make_unique<std::vector<math::RhoEtaPhiVectorF>>(size_);
  for (auto& p : *product) {
    p.SetCoordinates(genPt_(generator_), genEta_(generator_), genPhi_(generator_));
  }
  event.put(std::move(product));
}

void GenerateCylindricalVectors::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<uint32_t>("size", 1000)->setComment("number of generated elements");
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(GenerateCylindricalVectors);
