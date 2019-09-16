// system include files
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// CMSSW include files
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class CompareCartesianVectors : public edm::one::EDAnalyzer<> {
public:
  explicit CompareCartesianVectors(const edm::ParameterSet&);
  ~CompareCartesianVectors() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  bool compare(math::XYZVectorF const& first, math::XYZVectorF const& second) const;

  edm::EDGetTokenT<std::vector<math::XYZVectorF>> first_;
  edm::EDGetTokenT<std::vector<math::XYZVectorF>> second_;
  double error_;
};

CompareCartesianVectors::CompareCartesianVectors(const edm::ParameterSet& config)
    : first_(consumes<std::vector<math::XYZVectorF>>(config.getParameter<edm::InputTag>("first"))),
      second_(consumes<std::vector<math::XYZVectorF>>(config.getParameter<edm::InputTag>("second"))),
      error_(config.getParameter<double>("error")) {}

void CompareCartesianVectors::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  auto const& first = event.get(first_);
  auto const& second = event.get(second_);
  if (first.size() != second.size()) {
    std::cout << "The two collections have different size: " << first.size() << " and " << second.size() << "."
              << std::endl;
    return;
  }
  bool consistent = true;
  for (unsigned int i = 0; i < first.size(); ++i) {
    if (not compare(first[i], second[i])) {
      if (consistent) {
        std::cout << "Found inconsistent elements:" << std::endl;
        consistent = false;
      }
      std::cout << std::setprecision(9) << "(" << first[i].x() << ", " << first[i].y() << ", " << first[i].z()
                << ")  vs  (" << second[i].x() << ", " << second[i].y() << ", " << second[i].z() << ")" << std::endl;
    }
  }
  if (consistent) {
    std::cout << "All elements are consistent." << std::endl;
  }
}

bool CompareCartesianVectors::compare(math::XYZVectorF const& first, math::XYZVectorF const& second) const {
  if (std::abs(first.x() - second.x()) > std::abs(first.x() + second.x()) * error_)
    return false;
  if (std::abs(first.y() - second.y()) > std::abs(first.y() + second.y()) * error_)
    return false;
  if (std::abs(first.z() - second.z()) > std::abs(first.z() + second.z()) * error_)
    return false;
  return true;
}

void CompareCartesianVectors::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("first", edm::InputTag("firstCartesianVectors"));
  desc.add<edm::InputTag>("second", edm::InputTag("secondCartesianVectors"));
  desc.add<double>("error", 1.e-6);
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(CompareCartesianVectors);
