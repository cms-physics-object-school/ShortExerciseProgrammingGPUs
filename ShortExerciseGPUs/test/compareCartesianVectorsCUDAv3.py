import FWCore.ParameterSet.Config as cms

process = cms.Process("PRINT")

process.options = cms.untracked.PSet(
  numberOfThreads = cms.untracked.uint32( 1 ),
  numberOfStreams = cms.untracked.uint32( 1 ),
  wantSummary = cms.untracked.bool( False )
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("file:cylindricalVectors.root"),
)

process.convertToCartesianVectors = cms.EDProducer('ConvertToCartesianVectorsCUDAv3',
  input = cms.InputTag('generateCylindricalVectors')
)

process.convertToCartesianVectorsReference = cms.EDProducer('ConvertToCartesianVectorsReference',
  input = cms.InputTag('generateCylindricalVectors')
)

process.compareCartesianVectors = cms.EDAnalyzer('CompareCartesianVectors',
  first = cms.InputTag('convertToCartesianVectors'),
  second = cms.InputTag('convertToCartesianVectorsReference'),
  error = cms.double(1.e-7)
)

process.path = cms.Path(process.convertToCartesianVectors + process.convertToCartesianVectorsReference + process.compareCartesianVectors)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)
