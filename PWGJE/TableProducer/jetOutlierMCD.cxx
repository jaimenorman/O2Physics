// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Task to produce a table joinable to the collision tables for MC Detector level 
// track or jet pt relative to the pt-hard of the event, for outlier rejection purposes
//
/// \author Jaime Norman <jaime.norman@cern.ch>

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"
#include "Framework/O2DatabasePDGPlugin.h"

#include "PWGJE/DataModel/Jet.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#include "Framework/runDataProcessing.h"

template <typename MCDetectorLevelJetsTable>

struct JetOutlierMCDTask {
  Produces<aod::JCollisionsOutliers> collisionOutlierTable;

  void processDummy(aod::JetCollisions const&)
  {
  }
  PROCESS_SWITCH(JetOutlierMCDTask, processDummy, "Dummy process", true);

  void processMCDetectorLevelEventOutlier(soa::Join<aod::JetCollisions, aod::JMcCollisionLbs>::iterator const& collision, MCDetectorLevelJetsTable const& jets, aod::JetTracks const& tracks, aod::JetMcCollisions const&)
  {
    float ptHard = collision.ptHard();
    // jets
    float jetLeadPt = 0.;
    for(auto const& jet : jets ) {
      float pt = jet.pt();
      if(pt > jetLeadPt) {
        jetLeadPt = pt;
      }
    }
    // tracks
    float trackLeadPt = 0.;
    for(auto const& track : tracks ) {
      float pt = track.pt();
      if(pt > trackLeadPt) {
        trackLeadPt = pt;
      }
    }
    float leadJetPtHatFraction = -1.;
    float leadTrackPtHatFraction = -1.;
    if(ptHard > 1E-5) {
      leadJetPtHatFraction = jetLeadPt/ptHard;
      leadTrackPtHatFraction = trackLeadPt/ptHard;
    }
    LOG(info) << "collision" << collision.globalIndex() << " pt hat = " << ptHard << "jet lead pt fraction = " << leadJetPtHatFraction << " track lead pt fraction = " << leadTrackPtHatFraction;
    // auto collision = jet.template collision_as<soa::Join<aod::JetCollisions, aod::JMcCollisionLbs>>();
    collisionOutlierTable(collision.globalIndex(), leadJetPtHatFraction, leadTrackPtHatFraction);
  }
  PROCESS_SWITCH(JetOutlierMCDTask, processMCDetectorLevelEventOutlier, "Fill event outlier tables for detector level MC jets and tracks", false);

};

using ChargedMCJetsOutlier = JetOutlierMCDTask<aod::ChargedMCDetectorLevelJets>;
using NeutralMCJetsOutlier = JetOutlierMCDTask<aod::NeutralMCDetectorLevelJets>;
using FullMCJetsOutlier = JetOutlierMCDTask<aod::FullMCDetectorLevelJets>;
using D0ChargedMCJetsOutlier = JetOutlierMCDTask<aod::D0ChargedMCDetectorLevelJets>;
using DplusChargedMCJetsOutlier = JetOutlierMCDTask<aod::DplusChargedMCDetectorLevelJets>;
using LcChargedMCJetsOutlier = JetOutlierMCDTask<aod::LcChargedMCDetectorLevelJets>;
using BplusChargedMCJetsOutlier = JetOutlierMCDTask<aod::BplusChargedMCDetectorLevelJets>;
using V0ChargedMCJetsOutlier = JetOutlierMCDTask<aod::V0ChargedMCDetectorLevelJets>;

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  std::vector<o2::framework::DataProcessorSpec> tasks;

  tasks.emplace_back(
    adaptAnalysisTask<ChargedMCJetsOutlier>(cfgc,
                                                SetDefaultProcesses{}, TaskName{"jet-outlier-mcd-charged"}));

  // tasks.emplace_back(
  //   adaptAnalysisTask<NeutralMCJetsOutlier>(cfgc,
  //                                               SetDefaultProcesses{}, TaskName{"jet-outlier-mcd-neutral"}));

  // tasks.emplace_back(
  //   adaptAnalysisTask<FullMCJetsOutlier>(cfgc,
  //                                            SetDefaultProcesses{}, TaskName{"jet-outlier-mcd-full"}));

  // tasks.emplace_back(
  //   adaptAnalysisTask<D0ChargedMCJetsOutlier>(cfgc,
  //                                                 SetDefaultProcesses{}, TaskName{"jet-d0-outlier-mcd-charged"}));

  // tasks.emplace_back(
  //   adaptAnalysisTask<DplusChargedMCJetsOutlier>(cfgc,
  //                                                    SetDefaultProcesses{}, TaskName{"jet-dplus-outlier-mcd-charged"}));

  // tasks.emplace_back(
  //   adaptAnalysisTask<LcChargedMCJetsOutlier>(cfgc,
  //                                                 SetDefaultProcesses{}, TaskName{"jet-lc-outlier-mcd-charged"}));

  // tasks.emplace_back(
  //   adaptAnalysisTask<BplusChargedMCJetsOutlier>(cfgc,
  //                                                    SetDefaultProcesses{}, TaskName{"jet-bplus-outlier-mcd-charged"}));

  // tasks.emplace_back(
  //   adaptAnalysisTask<V0ChargedMCJetsOutlier>(cfgc,
  //                                                 SetDefaultProcesses{}, TaskName{"jet-v0-outlier-mcd-charged"}));

  return WorkflowSpec{tasks};
}
