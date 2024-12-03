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

// jet finder QA task
//
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>

#include <cmath>
#include <TRandom3.h>

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/O2DatabasePDGPlugin.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/runDataProcessing.h"

#include "Common/Core/TrackSelection.h"
#include "Common/Core/TrackSelectionDefaults.h"

#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"

#include "PWGJE/Core/FastJetUtilities.h"
#include "PWGJE/Core/JetFinder.h"
#include "PWGJE/Core/JetFindingUtilities.h"
#include "PWGJE/DataModel/Jet.h"

#include "PWGJE/Core/JetDerivedDataUtilities.h"

#include "EventFiltering/filterTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct JetFinderQATask {

  HistogramRegistry registry;

  Preslice<JetTracks> perCol = aod::jtrack::collisionId;
  Preslice<JetParticles> perColMC = aod::jmcparticle::mcCollisionId;
  Preslice<soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetEventWeights>> perColJets = aod::jet::collisionId;
  Preslice<soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetsMatchedToChargedMCParticleLevelJets, aod::ChargedMCDetectorLevelJetEventWeights>> perColJetsMatched = aod::jet::collisionId;

  Configurable<float> selectedJetsRadius{"selectedJetsRadius", 0.4, "resolution parameter for histograms without radius"};
  Configurable<std::string> eventSelections{"eventSelections", "sel8", "choose event selection"};
  Configurable<float> vertexZCut{"vertexZCut", 10.0f, "Accepted z-vertex range"};
  Configurable<float> centralityMin{"centralityMin", -999.0, "minimum centrality"};
  Configurable<float> centralityMax{"centralityMax", 999.0, "maximum centrality"};
  Configurable<std::vector<double>> jetRadii{"jetRadii", std::vector<double>{0.4}, "jet resolution parameters"};
  Configurable<float> trackEtaMin{"trackEtaMin", -0.9, "minimum eta acceptance for tracks"};
  Configurable<float> trackEtaMax{"trackEtaMax", 0.9, "maximum eta acceptance for tracks"};
  Configurable<float> trackPtMin{"trackPtMin", 0.15, "minimum pT acceptance for tracks"};
  Configurable<float> trackPtMax{"trackPtMax", 100.0, "maximum pT acceptance for tracks"};
  Configurable<std::string> trackSelections{"trackSelections", "globalTracks", "set track selections"};
  Configurable<float> pTHatMaxMCD{"pTHatMaxMCD", 999.0, "maximum fraction of hard scattering for jet acceptance in detector MC"};
  Configurable<float> pTHatMaxMCP{"pTHatMaxMCP", 999.0, "maximum fraction of hard scattering for jet acceptance in particle MC"};
  Configurable<float> pTHatExponent{"pTHatExponent", 6.0, "exponent of the event weight for the calculation of pTHat"};
  Configurable<double> jetPtMax{"jetPtMax", 200., "set jet pT bin max"};
  Configurable<float> jetEtaMin{"jetEtaMin", -99.0, "minimum jet pseudorapidity"};
  Configurable<float> jetEtaMax{"jetEtaMax", 99.0, "maximum jet pseudorapidity"};
  Configurable<int> nBinsEta{"nBinsEta", 200, "number of bins for eta axes"};
  Configurable<float> jetAreaFractionMin{"jetAreaFractionMin", -99.0, "used to make a cut on the jet areas"};
  Configurable<float> leadingConstituentPtMin{"leadingConstituentPtMin", -99.0, "minimum pT selection on jet constituent"};
  Configurable<float> leadingConstituentPtMax{"leadingConstituentPtMax", 9999.0, "maximum pT selection on jet constituent"};
  Configurable<float> randomConeR{"randomConeR", 0.4, "size of random Cone for estimating background fluctuations"};
  Configurable<float> randomConeLeadJetDeltaR{"randomConeLeadJetDeltaR", -99.0, "min distance between leading jet axis and random cone (RC) axis; if negative, min distance is set to automatic value of R_leadJet+R_RC "};
  Configurable<bool> checkMcCollisionIsMatched{"checkMcCollisionIsMatched", false, "0: count whole MCcollisions, 1: select MCcollisions which only have their correspond collisions"};
  Configurable<int> trackOccupancyInTimeRangeMax{"trackOccupancyInTimeRangeMax", 999999, "maximum occupancy of tracks in neighbouring collisions in a given time range; only applied to reconstructed collisions (data and mcd jets), not mc collisions (mcp jets)"};
  Configurable<int> trackOccupancyInTimeRangeMin{"trackOccupancyInTimeRangeMin", -999999, "minimum occupancy of tracks in neighbouring collisions in a given time range; only applied to reconstructed collisions (data and mcd jets), not mc collisions (mcp jets)"};
  Configurable<bool> isMB{"isMB", false, "flag to choose events based on weight (MB = 1, weighted = anything else"};
  Configurable<bool> isFillAmbiguous{"isFillAmbiguous", false, "flag to fill histograms checking ambiguous tracks"};
  Configurable<float> diffNeighbours{"diffNeighbours", 999.0, "maximum difference in pThat between current collision and neighbours"};
  Configurable<bool> isRejectHFNeighbours{"isRejectHFNeighbours", false, "reject neighbours with charm or beauty"};

  std::vector<bool> filledJetR_Both;
  std::vector<bool> filledJetR_Low;
  std::vector<bool> filledJetR_High;
  std::vector<double> jetRadiiValues;

  int eventSelection = -1;
  int trackSelection = -1;

  std::vector<double> jetPtBins;
  std::vector<double> jetPtBinsRhoAreaSub;

  void init(o2::framework::InitContext&)
  {
    eventSelection = jetderiveddatautilities::initialiseEventSelection(static_cast<std::string>(eventSelections));
    trackSelection = jetderiveddatautilities::initialiseTrackSelection(static_cast<std::string>(trackSelections));
    jetRadiiValues = (std::vector<double>)jetRadii;

    for (std::size_t iJetRadius = 0; iJetRadius < jetRadiiValues.size(); iJetRadius++) {
      filledJetR_Both.push_back(0.0);
      filledJetR_Low.push_back(0.0);
      filledJetR_High.push_back(0.0);
    }
    auto jetRadiiBins = (std::vector<double>)jetRadii;
    if (jetRadiiBins.size() > 1) {
      jetRadiiBins.push_back(jetRadiiBins[jetRadiiBins.size() - 1] + (TMath::Abs(jetRadiiBins[jetRadiiBins.size() - 1] - jetRadiiBins[jetRadiiBins.size() - 2])));
    } else {
      jetRadiiBins.push_back(jetRadiiBins[jetRadiiBins.size() - 1] + 0.1);
    }

    auto jetPtTemp = 0.0;
    jetPtBins.push_back(jetPtTemp);
    jetPtBinsRhoAreaSub.push_back(jetPtTemp);
    while (jetPtTemp < jetPtMax) {
      if (jetPtTemp < 100.0) {
        jetPtTemp += 1.0;
        jetPtBins.push_back(jetPtTemp);
        jetPtBinsRhoAreaSub.push_back(jetPtTemp);
        jetPtBinsRhoAreaSub.push_back(-jetPtTemp);
      } else if (jetPtTemp < 200.0) {
        jetPtTemp += 5.0;
        jetPtBins.push_back(jetPtTemp);
        jetPtBinsRhoAreaSub.push_back(jetPtTemp);
        jetPtBinsRhoAreaSub.push_back(-jetPtTemp);
      } else {
        jetPtTemp += 10.0;
        jetPtBins.push_back(jetPtTemp);
        jetPtBinsRhoAreaSub.push_back(jetPtTemp);
        jetPtBinsRhoAreaSub.push_back(-jetPtTemp);
      }
    }
    std::sort(jetPtBinsRhoAreaSub.begin(), jetPtBinsRhoAreaSub.end());

    AxisSpec jetPtAxis = {jetPtBins, "#it{p}_{T} (GeV/#it{c})"};
    AxisSpec jetPtAxisRhoAreaSub = {jetPtBinsRhoAreaSub, "#it{p}_{T} (GeV/#it{c})"};

    AxisSpec jetEtaAxis = {nBinsEta, jetEtaMin, jetEtaMax, "#eta"};
    AxisSpec trackEtaAxis = {nBinsEta, trackEtaMin, trackEtaMax, "#eta"};

    if (doprocessJetsData || doprocessJetsMCD || doprocessJetsMCDWeighted || doprocessTracksColl) {
      registry.add("h_jet_pt", "jet pT;#it{p}_{T,jet} (GeV/#it{c});entries", {HistType::kTH1F, {jetPtAxis}});
      registry.add("h_jet_eta", "jet #eta;#eta_{jet};entries", {HistType::kTH1F, {jetEtaAxis}});
      registry.add("h_jet_phi", "jet #varphi;#varphi_{jet};entries", {HistType::kTH1F, {{160, -1.0, 7.}}});
      registry.add("h_jet_ntracks", "jet N tracks;N_{jet tracks};entries", {HistType::kTH1F, {{200, -0.5, 199.5}}});
      registry.add("h2_centrality_jet_pt", "centrality vs #it{p}_{T,jet}; centrality; #it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH2F, {{1200, -10.0, 110.0}, jetPtAxis}});
      registry.add("h2_centrality_jet_eta", "centrality vs #eta_{jet}; centrality; #eta_{jet}", {HistType::kTH2F, {{1200, -10.0, 110.0}, jetEtaAxis}});
      registry.add("h2_centrality_jet_phi", "centrality vs #varphi_{jet}; centrality; #varphi_{jet}", {HistType::kTH2F, {{1200, -10.0, 110.0}, {160, -1.0, 7.}}});
      registry.add("h2_centrality_jet_ntracks", "centrality vs N_{jet tracks}; centrality; N_{jet tracks}", {HistType::kTH2F, {{1200, -10.0, 110.0}, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_centrality", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});centrality", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1200, -10.0, 110.0}}});
      registry.add("h3_jet_r_jet_pt_jet_eta", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, jetEtaAxis}});
      registry.add("h3_jet_r_jet_pt_jet_phi", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_eta_jet_phi", "#it{R}_{jet};#eta_{jet};#varphi_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetEtaAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_jet_ntracks", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});N_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_jet_area", "#it{R}_{jet}; #it{p}_{T,jet} (GeV/#it{c}); #it{area}_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {300, 0., 3.}}});
      registry.add("h3_jet_r_jet_pt_track_pt", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,jet tracks} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, 0., 200.}}});
      registry.add("h3_jet_r_jet_pt_track_eta", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, trackEtaAxis}});
      registry.add("h3_jet_r_jet_pt_track_phi", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_Dz", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});z = #it{p}_{T,track} / #it{p}_{T,jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {30, 0., 1.}}});
      registry.add("h3_jet_r_jet_pt_leadingtrack_pt", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c}); #it{p}_{T,leading track} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, 0.0, 200.0}}});
      registry.add("h_jet_phat", "jet #hat{p};#hat{p} (GeV/#it{c});entries", {HistType::kTH1F, {{1000, 0, 1000}}});
      registry.add("h_matched_phat", "jet #hat{p};#hat{p} (GeV/#it{c});entries", {HistType::kTH1F, {{1000, 0, 1000}}});
      registry.add("h_jet_ptcut", "p_{T} cut;p_{T,jet} (GeV/#it{c});N;entries", {HistType::kTH2F, {{300, 0, 300}, {20, 0, 5}}});
      registry.add("h_jet_phat_weighted", "jet #hat{p};#hat{p} (GeV/#it{c});entries", {HistType::kTH1F, {{1000, 0, 1000}}});
      registry.add("h3_centrality_occupancy_jet_pt", "centrality; occupancy; #it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH3F, {{120, -10.0, 110.0}, {60, 0, 30000}, jetPtAxis}});
      registry.add("h3_jet_pt_track_pt_pt_hat_ambiguous", "ambiguous;#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,track} (GeV/#it{c}); #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH3F, {{150,0,300}, {100,0,100}, {200,0,600}}});
      registry.add("h3_jet_pt_track_pt_pt_hat_unambiguous", "matched;#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,track} (GeV/#it{c}); #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH3F, {{150,0,300}, {100,0,100}, {200,0,600}}});
      registry.add("h3_jet_pt_frac_pt_ambiguous_pt_hat", "fraction pT;#it{p}_{T,jet} (GeV/#it{c});fraction of #it{p}_{T,track} unmatched; #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH3F, {{150,0,300}, {40,0,1.1}, {200,0,600}}});
      registry.add("h3_jet_pt_frac_constituents_ambiguous_pt_hat", "fraction const;#it{p}_{T,jet} (GeV/#it{c});fraction of constituents matched; #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH3F, {{150,0,300}, {40,0,1.1}, {200,0,600}}});
      registry.add("h3_jet_pt_track_pt_pt_hat_no_particle", "no matching particle;#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,track} (GeV/#it{c}); #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH3F, {{150,0,300}, {100,0,100}, {200,0,600}}});
      registry.add("h3_jet_pt_track_pt_pt_hat_with_particle", "with matching particle;#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,track} (GeV/#it{c}); #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH3F, {{150,0,300}, {100,0,100}, {200,0,600}}});
      registry.add("h3_jet_pt_frac_pt_unmatched_particle_pt_hat", "fraction pT;#it{p}_{T,jet} (GeV/#it{c});fraction of #it{p}_{T,track} unmatched; #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH3F, {{150,0,300}, {40,0,1.1}, {200,0,600}}});
      registry.add("h3_jet_pt_frac_constituents_unmatched_particle_pt_hat", "fraction const;#it{p}_{T,jet} (GeV/#it{c});fraction of constituents unmatched; #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH3F, {{150,0,300}, {40,0,1.1}, {200,0,600}}});
    }

    if (doprocessJetsRhoAreaSubData || doprocessJetsRhoAreaSubMCD) {

      registry.add("h_jet_pt_rhoareasubtracted", "jet pT;#it{p}_{T,jet} (GeV/#it{c});entries", {HistType::kTH1F, {jetPtAxisRhoAreaSub}});
      registry.add("h_jet_eta_rhoareasubtracted", "jet #eta;#eta_{jet};entries", {HistType::kTH1F, {jetEtaAxis}});
      registry.add("h_jet_phi_rhoareasubtracted", "jet #varphi;#varphi_{jet};entries", {HistType::kTH1F, {{160, -1.0, 7.}}});
      registry.add("h_jet_ntracks_rhoareasubtracted", "jet N tracks;N_{jet tracks};entries", {HistType::kTH1F, {{200, -0.5, 199.5}}});
      registry.add("h2_centrality_jet_pt_rhoareasubtracted", "centrality vs #it{p}_{T,jet}; centrality; #it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH2F, {{1200, -10.0, 110.0}, jetPtAxisRhoAreaSub}});
      registry.add("h2_centrality_jet_eta_rhoareasubtracted", "centrality vs #eta_{jet}; centrality; #eta_{jet}", {HistType::kTH2F, {{1200, -10.0, 110.0}, jetEtaAxis}});
      registry.add("h2_centrality_jet_phi_rhoareasubtracted", "centrality vs #varphi_{jet}; centrality; #varphi_{jet}", {HistType::kTH2F, {{1200, -10.0, 110.0}, {160, -1.0, 7.}}});
      registry.add("h2_centrality_jet_ntracks_rhoareasubtracted", "centrality vs N_{jet tracks}; centrality; N_{jet tracks}", {HistType::kTH2F, {{1200, -10.0, 110.0}, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_centrality_rhoareasubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});centrality", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxisRhoAreaSub, {1200, -10.0, 110.0}}});
      registry.add("h3_jet_r_jet_pt_jet_eta_rhoareasubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxisRhoAreaSub, jetEtaAxis}});
      registry.add("h3_jet_r_jet_pt_jet_phi_rhoareasubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxisRhoAreaSub, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_eta_jet_phi_rhoareasubtracted", "#it{R}_{jet};#eta_{jet};#varphi_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetEtaAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_jet_ntracks_rhoareasubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});N_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxisRhoAreaSub, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_jet_area_rhoareasubtracted", "#it{R}_{jet}; #it{p}_{T,jet} (GeV/#it{c}); #it{area}_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxisRhoAreaSub, {300, 0., 3.}}});
      registry.add("h3_jet_r_jet_pt_track_pt_rhoareasubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,jet tracks} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxisRhoAreaSub, {200, 0., 200.}}});
      registry.add("h3_jet_r_jet_pt_track_eta_rhoareasubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxisRhoAreaSub, trackEtaAxis}});
      registry.add("h3_jet_r_jet_pt_track_phi_rhoareasubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxisRhoAreaSub, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_jet_pt_rhoareasubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, jetPtAxisRhoAreaSub}});
      registry.add("h3_centrality_occupancy_jet_pt_rhoareasubtracted", "centrality; occupancy; #it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH3F, {{120, -10.0, 110.0}, {60, 0, 30000}, jetPtAxisRhoAreaSub}});
    }

    if (doprocessEvtWiseConstSubJetsData || doprocessEvtWiseConstSubJetsMCD) {
      registry.add("h_jet_pt_eventwiseconstituentsubtracted", "jet pT;#it{p}_{T,jet} (GeV/#it{c});entries", {HistType::kTH1F, {jetPtAxis}});
      registry.add("h_jet_eta_eventwiseconstituentsubtracted", "jet #eta;#eta_{jet};entries", {HistType::kTH1F, {jetEtaAxis}});
      registry.add("h_jet_phi_eventwiseconstituentsubtracted", "jet #varphi;#varphi_{jet};entries", {HistType::kTH1F, {{160, -1.0, 7.}}});
      registry.add("h_jet_ntracks_eventwiseconstituentsubtracted", "jet N tracks;N_{jet tracks};entries", {HistType::kTH1F, {{200, -0.5, 199.5}}});
      registry.add("h2_centrality_jet_pt_eventwiseconstituentsubtracted", "centrality vs #it{p}_{T,jet}; centrality; #it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH2F, {{1200, -10.0, 110.0}, jetPtAxis}});
      registry.add("h2_centrality_jet_eta_eventwiseconstituentsubtracted", "centrality vs #eta_{jet}; centrality; #eta_{jet}", {HistType::kTH2F, {{1200, -10.0, 110.0}, jetEtaAxis}});
      registry.add("h2_centrality_jet_phi_eventwiseconstituentsubtracted", "centrality vs #varphi_{jet}; centrality; #varphi_{jet}", {HistType::kTH2F, {{1200, -10.0, 110.0}, {160, -1.0, 7.}}});
      registry.add("h2_centrality_jet_ntracks_eventwiseconstituentsubtracted", "centrality vs N_{jet tracks}; centrality; N_{jet tracks}", {HistType::kTH2F, {{1200, -10.0, 110.0}, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_centrality_eventwiseconstituentsubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});centrality", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1200, -10.0, 110.0}}});
      registry.add("h3_jet_r_jet_pt_jet_eta_eventwiseconstituentsubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, jetEtaAxis}});
      registry.add("h3_jet_r_jet_pt_jet_phi_eventwiseconstituentsubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_eta_jet_phi_eventwiseconstituentsubtracted", "#it{R}_{jet};#eta_{jet};#varphi_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetEtaAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_jet_ntracks_eventwiseconstituentsubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});N_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_jet_area_eventwiseconstituentsubtracted", "#it{R}_{jet}; #it{p}_{T,jet} (GeV/#it{c}); #it{area}_{jet}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {300, 0., 3.}}});
      registry.add("h3_jet_r_jet_pt_track_pt_eventwiseconstituentsubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,jet tracks} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, jetPtAxis}});
      registry.add("h3_jet_r_jet_pt_track_eta_eventwiseconstituentsubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, trackEtaAxis}});
      registry.add("h3_jet_r_jet_pt_track_phi_eventwiseconstituentsubtracted", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
    }

    if (doprocessRho) {
      registry.add("h2_centrality_ntracks", "; centrality; N_{tracks};", {HistType::kTH2F, {{1100, 0., 110.0}, {10000, 0.0, 10000.0}}});
      registry.add("h2_ntracks_rho", "; N_{tracks}; #it{rho} (GeV/area);", {HistType::kTH2F, {{10000, 0.0, 10000.0}, {400, 0.0, 400.0}}});
      registry.add("h2_ntracks_rhom", "; N_{tracks}; #it{rho}_{m} (GeV/area);", {HistType::kTH2F, {{10000, 0.0, 10000.0}, {100, 0.0, 100.0}}});
      registry.add("h2_centrality_rho", "; centrality; #it{rho} (GeV/area);", {HistType::kTH2F, {{1100, 0., 110.}, {400, 0., 400.0}}});
      registry.add("h2_centrality_rhom", ";centrality; #it{rho}_{m} (GeV/area)", {HistType::kTH2F, {{1100, 0., 110.}, {100, 0., 100.0}}});
    }

    if (doprocessRandomConeData || doprocessRandomConeMCD) {
      registry.add("h2_centrality_rhorandomcone", "; centrality; #it{p}_{T,random cone} - #it{area, random cone} * #it{rho} (GeV/c);", {HistType::kTH2F, {{1100, 0., 110.}, {800, -400.0, 400.0}}});
      registry.add("h2_centrality_rhorandomconerandomtrackdirection", "; centrality; #it{p}_{T,random cone} - #it{area, random cone} * #it{rho} (GeV/c);", {HistType::kTH2F, {{1100, 0., 110.}, {800, -400.0, 400.0}}});
      registry.add("h2_centrality_rhorandomconewithoutleadingjet", "; centrality; #it{p}_{T,random cone} - #it{area, random cone} * #it{rho} (GeV/c);", {HistType::kTH2F, {{1100, 0., 110.}, {800, -400.0, 400.0}}});
      registry.add("h2_centrality_rhorandomconerandomtrackdirectionwithoutoneleadingjets", "; centrality; #it{p}_{T,random cone} - #it{area, random cone} * #it{rho} (GeV/c);", {HistType::kTH2F, {{1100, 0., 110.}, {800, -400.0, 400.0}}});
      registry.add("h2_centrality_rhorandomconerandomtrackdirectionwithouttwoleadingjets", "; centrality; #it{p}_{T,random cone} - #it{area, random cone} * #it{rho} (GeV/c);", {HistType::kTH2F, {{1100, 0., 110.}, {800, -400.0, 400.0}}});
    }

    if (doprocessJetsMCP || doprocessJetsMCPWeighted) {
      registry.add("h_jet_pt_part", "jet pT;#it{p}_{T,jet}^{part}(GeV/#it{c});entries", {HistType::kTH1F, {jetPtAxis}});
      registry.add("h_jet_eta_part", "jet #eta;#eta_{jet}^{part};entries", {HistType::kTH1F, {jetEtaAxis}});
      registry.add("h_jet_phi_part", "jet #varphi;#varphi_{jet}^{part};entries", {HistType::kTH1F, {{160, -1.0, 7.}}});
      registry.add("h_jet_ntracks_part", "jet N tracks;N_{jet tracks}^{part};entries", {HistType::kTH1F, {{200, -0.5, 199.5}}});
      registry.add("h3_jet_r_part_jet_pt_part_jet_eta_part", ";#it{R}_{jet}^{part};#it{p}_{T,jet}^{part} (GeV/#it{c});#eta_{jet}^{part}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, jetEtaAxis}});
      registry.add("h3_jet_r_part_jet_pt_part_jet_phi_part", ";#it{R}_{jet}^{part};#it{p}_{T,jet}^{part} (GeV/#it{c});#varphi_{jet}^{part}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_part_jet_eta_part_jet_phi_part", ";#it{R}_{jet}^{part};#eta_{jet}^{part};#varphi_{jet}^{part}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetEtaAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_part_jet_pt_part_jet_ntracks_part", "#it{R}_{jet}^{part};#it{p}_{T,jet}^{part} (GeV/#it{c});N_{jet tracks}^{part}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_part_jet_pt_part_track_pt_part", "#it{R}_{jet}^{part};#it{p}_{T,jet}^{part} (GeV/#it{c});#it{p}_{T,jet tracks}^{part} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, 0., 200.}}});
      registry.add("h3_jet_r_part_jet_pt_part_track_eta_part", "#it{R}_{jet}^{part};#it{p}_{T,jet}^{part} (GeV/#it{c});#eta_{jet tracks}^{part}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, trackEtaAxis}});
      registry.add("h3_jet_r_part_jet_pt_part_track_phi_part", "#it{R}_{jet}^{part};#it{p}_{T,jet}^{part} (GeV/#it{c});#varphi_{jet tracks}^{part}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_part_jet_pt_part_Dz_part", "#it{R}_{jet}^{part};#it{p}_{T,jet}^{part} (GeV/#it{c});z = #it{p}_{T,jet tracks}^{part} / #it{p}_{T,jet}^{part}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {30, 0., 1.}}});
      registry.add("h_jet_phat_part", "jet #hat{p};#hat{p} (GeV/#it{c});entries", {HistType::kTH1F, {{1000, 0, 1000}}});
      registry.add("h_jet_ptcut_part", "p_{T} cut;p_{T,jet}^{part} (GeV/#it{c});N;entries", {HistType::kTH2F, {{300, 0, 300}, {20, 0, 5}}});
      registry.add("h_jet_phat_part_weighted", "jet #hat{p};#hat{p} (GeV/#it{c});entries", {HistType::kTH1F, {{1000, 0, 1000}}});
    }

    if (doprocessJetsMCPMCDMatched || doprocessJetsMCPMCDMatchedWeighted || doprocessJetsSubMatched || doprocessTracksMatchedColl) {
      registry.add("h3_jet_r_jet_pt_tag_jet_pt_base_matchedgeo", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c});#it{p}_{T,jet}^{base} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, {300, 0., 300.}, {300, 0., 300.}}});
      registry.add("h3_jet_r_jet_eta_tag_jet_eta_base_matchedgeo", "#it{R}_{jet};#eta_{jet}^{tag};#eta_{jet}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetEtaAxis, jetEtaAxis}});
      registry.add("h3_jet_r_jet_phi_tag_jet_phi_base_matchedgeo", "#it{R}_{jet};#varphi_{jet}^{tag};#varphi_{jet}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, {160, -1.0, 7.}, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_ntracks_tag_jet_ntracks_base_matchedgeo", "#it{R}_{jet};N_{jet tracks}^{tag};N_{jet tracks}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, {200, -0.5, 199.5}, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_pt_base_diff_matchedgeo", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#it{p}_{T,jet}^{tag} (GeV/#it{c}) - #it{p}_{T,jet}^{base} (GeV/#it{c})) / #it{p}_{T,jet}^{tag} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 2.0}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_eta_base_diff_matchedgeo", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#eta_{jet}^{tag} - #eta_{jet}^{base}) / #eta_{jet}^{tag}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 5.0}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_phi_base_diff_matchedgeo", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#varphi_{jet}^{tag} - #varphi_{jet}^{base}) / #varphi_{jet}^{tag}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 5.0}}});
      registry.add("h3_jet_pt_tag_jet_eta_tag_jet_eta_base_matchedgeo", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); #eta_{jet}^{tag}; #eta_{jet}^{base}", {HistType::kTH3F, {jetPtAxis, jetEtaAxis, jetEtaAxis}});
      registry.add("h3_jet_pt_tag_jet_phi_tag_jet_phi_base_matchedgeo", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); #varphi_{jet}^{tag}; #varphi_{jet}^{base}", {HistType::kTH3F, {jetPtAxis, {160, -1.0, 7.}, {160, -1.0, 7.}}});
      registry.add("h3_jet_pt_tag_jet_ntracks_tag_jet_ntracks_base_matchedgeo", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); N_{jet tracks}^{tag}; N_{jet tracks}^{base}", {HistType::kTH3F, {jetPtAxis, {200, -0.5, 199.5}, {200, -0.5, 199.5}}});

      registry.add("h3_jet_r_jet_pt_tag_jet_pt_base_matchedpt", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c});#it{p}_{T,jet}^{base} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, jetPtAxis}});
      registry.add("h3_jet_r_jet_eta_tag_jet_eta_base_matchedpt", "#it{R}_{jet};#eta_{jet}^{tag};#eta_{jet}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetEtaAxis, jetEtaAxis}});
      registry.add("h3_jet_r_jet_phi_tag_jet_phi_base_matchedpt", "#it{R}_{jet};#varphi_{jet}^{tag};#varphi_{jet}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, {160, -1.0, 7.}, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_ntracks_tag_jet_ntracks_base_matchedpt", "#it{R}_{jet};N_{jet tracks}^{tag};N_{jet tracks}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, {200, -0.5, 199.5}, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_pt_base_diff_matchedpt", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#it{p}_{T,jet}^{tag} (GeV/#it{c}) - #it{p}_{T,jet}^{base} (GeV/#it{c})) / #it{p}_{T,jet}^{tag} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 5.0}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_eta_base_diff_matchedpt", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#eta_{jet}^{tag} - #eta_{jet}^{base}) / #eta_{jet}^{tag}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 5.0}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_phi_base_diff_matchedpt", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#varphi_{jet}^{tag} - #varphi_{jet}^{base}) / #varphi_{jet}^{tag}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 5.0}}});
      registry.add("h3_jet_pt_tag_jet_eta_tag_jet_eta_base_matchedpt", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); #eta_{jet}^{tag}; #eta_{jet}^{base}", {HistType::kTH3F, {jetPtAxis, jetEtaAxis, jetEtaAxis}});
      registry.add("h3_jet_pt_tag_jet_phi_tag_jet_phi_base_matchedpt", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); #varphi_{jet}^{tag}; #varphi_{jet}^{base}", {HistType::kTH3F, {jetPtAxis, {160, -1.0, 7.}, {160, -1.0, 7.}}});
      registry.add("h3_jet_pt_tag_jet_ntracks_tag_jet_ntracks_base_matchedpt", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); N_{jet tracks}^{tag}; N_{jet tracks}^{base}", {HistType::kTH3F, {jetPtAxis, {200, -0.5, 199.5}, {200, -0.5, 199.5}}});

      registry.add("h3_jet_r_jet_pt_tag_jet_pt_base_matchedgeopt", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c});#it{p}_{T,jet}^{base} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, jetPtAxis}});
      registry.add("h3_jet_r_jet_eta_tag_jet_eta_base_matchedgeopt", "#it{R}_{jet};#eta_{jet}^{tag};#eta_{jet}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetEtaAxis, jetEtaAxis}});
      registry.add("h3_jet_r_jet_phi_tag_jet_phi_base_matchedgeopt", "#it{R}_{jet};#varphi_{jet}^{tag};#varphi_{jet}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, {160, -1.0, 7.}, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_ntracks_tag_jet_ntracks_base_matchedgeopt", "#it{R}_{jet};N_{jet tracks}^{tag};N_{jet tracks}^{base}", {HistType::kTH3F, {{jetRadiiBins, ""}, {200, -0.5, 199.5}, {200, -0.5, 199.5}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_pt_base_diff_matchedgeopt", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#it{p}_{T,jet}^{tag} (GeV/#it{c}) - #it{p}_{T,jet}^{base} (GeV/#it{c})) / #it{p}_{T,jet}^{tag} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 5.0}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_eta_base_diff_matchedgeopt", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#eta_{jet}^{tag} - #eta_{jet}^{base}) / #eta_{jet}^{tag}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 5.0}}});
      registry.add("h3_jet_r_jet_pt_tag_jet_phi_base_diff_matchedgeopt", "#it{R}_{jet};#it{p}_{T,jet}^{tag} (GeV/#it{c}); (#varphi_{jet}^{tag} - #varphi_{jet}^{base}) / #varphi_{jet}^{tag}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {1000, -5.0, 5.0}}});
      registry.add("h3_jet_pt_tag_jet_eta_tag_jet_eta_base_matchedgeopt", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); #eta_{jet}^{tag}; #eta_{jet}^{base}", {HistType::kTH3F, {jetPtAxis, jetEtaAxis, jetEtaAxis}});
      registry.add("h3_jet_pt_tag_jet_phi_tag_jet_phi_base_matchedgeopt", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); #varphi_{jet}^{tag}; #varphi_{jet}^{base}", {HistType::kTH3F, {jetPtAxis, {160, -1.0, 7.}, {160, -1.0, 7.}}});
      registry.add("h3_jet_pt_tag_jet_ntracks_tag_jet_ntracks_base_matchedgeopt", ";#it{p}_{T,jet}^{tag} (GeV/#it{c}); N_{jet tracks}^{tag}; N_{jet tracks}^{base}", {HistType::kTH3F, {jetPtAxis, {200, -0.5, 199.5}, {200, -0.5, 199.5}}});
      registry.add("h3_ptcut_jet_pt_tag_jet_pt_base_matchedgeo", "N;#it{p}_{T,jet}^{tag} (GeV/#it{c});#it{p}_{T,jet}^{base} (GeV/#it{c})", {HistType::kTH3F, {{20, 0., 5.}, {300, 0., 300.}, {300, 0., 300.}}});
    }

    if (doprocessTriggeredData) {
      registry.add("h_collision_trigger_events", "event status;event status;entries", {HistType::kTH1F, {{6, 0.0, 6.0}}});
      registry.add("h_track_pt_MB", "track pT for MB events;#it{p}_{T,track} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0.0, 200.0}}});
      registry.add("h_track_eta_MB", "track #eta for MB events;#eta_{track};entries", {HistType::kTH1F, {trackEtaAxis}});
      registry.add("h_track_phi_MB", "track #varphi for MB events;#varphi_{track};entries", {HistType::kTH1F, {{160, -1.0, 7.}}});
      registry.add("h_track_pt_Triggered_Low", "track pT for low #it{p}_{T} Triggered events;#it{p}_{T,track} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0.0, 200.0}}});
      registry.add("h_track_eta_Triggered_Low", "track #eta for low #it{p}_{T} Triggered events;#eta_{track};entries", {HistType::kTH1F, {trackEtaAxis}});
      registry.add("h_track_phi_Triggered_Low", "track #varphi for low #it{p}_{T} Triggered events;#varphi_{track};entries", {HistType::kTH1F, {{160, -1.0, 7.}}});
      registry.add("h_track_pt_Triggered_High", "track pT for high #it{p}_{T} Triggered events;#it{p}_{T,track} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0.0, 200.0}}});
      registry.add("h_track_eta_Triggered_High", "track #eta for high #it{p}_{T} Triggered events;#eta_{track};entries", {HistType::kTH1F, {trackEtaAxis}});
      registry.add("h_track_phi_Triggered_High", "track #varphi for high #it{p}_{T} Triggered events;#varphi_{track};entries", {HistType::kTH1F, {{160, -1.0, 7.}}});
      registry.add("h_track_pt_Triggered_Both", "track pT for both #it{p}_{T} Triggered events;#it{p}_{T,track} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0.0, 200.0}}});
      registry.add("h_track_eta_Triggered_Both", "track #eta for both #it{p}_{T} Triggered events;#eta_{track};entries", {HistType::kTH1F, {trackEtaAxis}});
      registry.add("h_track_phi_Triggered_Both", "track #varphi for both #it{p}_{T} Triggered events;#varphi_{track};entries", {HistType::kTH1F, {{160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_collision", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});collision trigger status", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {4, -0.5, 3.5}}});
      registry.add("h3_jet_r_jet_eta_collision", "#it{R}_{jet};#eta_{jet};collision trigger status", {HistType::kTH3F, {{jetRadiiBins, ""}, jetEtaAxis, {4, -0.5, 3.5}}});
      registry.add("h3_jet_r_jet_phi_collision", "#it{R}_{jet};#varphi_{jet};collision trigger status", {HistType::kTH3F, {{jetRadiiBins, ""}, {160, -1.0, 7.}, {4, -0.5, 3.5}}});
      registry.add("h2_jet_r_jet_pT_triggered_Low", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH2F, {{jetRadiiBins, ""}, jetPtAxis}});
      registry.add("h2_jet_r_jet_pT_triggered_High", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH2F, {{jetRadiiBins, ""}, jetPtAxis}});
      registry.add("h2_jet_r_jet_pT_triggered_Both", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c})", {HistType::kTH2F, {{jetRadiiBins, ""}, jetPtAxis}});
      registry.add("h3_jet_r_jet_pt_track_pt_MB", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,jet tracks} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, 0., 200.}}});
      registry.add("h3_jet_r_jet_pt_track_eta_MB", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, trackEtaAxis}});
      registry.add("h3_jet_r_jet_pt_track_phi_MB", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_track_pt_Triggered_Low", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,jet tracks} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, 0., 200.}}});
      registry.add("h3_jet_r_jet_pt_track_eta_Triggered_Low", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, trackEtaAxis}});
      registry.add("h3_jet_r_jet_pt_track_phi_Triggered_Low", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_track_pt_Triggered_High", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,jet tracks} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, 0., 200.}}});
      registry.add("h3_jet_r_jet_pt_track_eta_Triggered_High", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, trackEtaAxis}});
      registry.add("h3_jet_r_jet_pt_track_phi_Triggered_High", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
      registry.add("h3_jet_r_jet_pt_track_pt_Triggered_Both", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#it{p}_{T,jet tracks} (GeV/#it{c})", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {200, 0., 200.}}});
      registry.add("h3_jet_r_jet_pt_track_eta_Triggered_Both", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#eta_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, trackEtaAxis}});
      registry.add("h3_jet_r_jet_pt_track_phi_Triggered_Both", "#it{R}_{jet};#it{p}_{T,jet} (GeV/#it{c});#varphi_{jet tracks}", {HistType::kTH3F, {{jetRadiiBins, ""}, jetPtAxis, {160, -1.0, 7.}}});
    }

    if (doprocessTracks || doprocessTracksWeighted) {
      registry.add("h_collisions", "event status;event status;entries", {HistType::kTH1F, {{4, 0.0, 4.0}}});
      registry.add("h2_centrality_collisions", "centrality vs collisions; centrality; collisions", {HistType::kTH2F, {{1200, -10.0, 110.0}, {4, 0.0, 4.0}}});
      registry.add("h3_centrality_track_pt_track_phi", "centrality vs track pT vs track #varphi; centrality; #it{p}_{T,track} (GeV/#it{c}); #varphi_{track}", {HistType::kTH3F, {{1200, -10.0, 110.0}, {200, 0., 200.}, {160, -1.0, 7.}}});
      registry.add("h3_centrality_track_pt_track_eta", "centrality vs track pT vs track #eta; centrality; #it{p}_{T,track} (GeV/#it{c}); #eta_{track}", {HistType::kTH3F, {{1200, -10.0, 110.0}, {200, 0., 200.}, trackEtaAxis}});
      registry.add("h3_centrality_track_pt_track_dcaxy", "centrality vs track pT vs track DCA_{xy}; centrality; #it{p}_{T,track} (GeV/#it{c}); track DCA_{xy}", {HistType::kTH3F, {{120, -10.0, 110.0}, {20, 0., 100.}, {200, -0.15, 0.15}}});
      registry.add("h3_track_pt_track_eta_track_phi", "track pT vs track #eta vs track #varphi; #it{p}_{T,track} (GeV/#it{c}); #eta_{track}; #varphi_{track}", {HistType::kTH3F, {{200, 0., 200.}, trackEtaAxis, {160, -1.0, 7.}}});
      if (doprocessTracksWeighted) {
        registry.add("h_collisions_weighted", "event status;event status;entries", {HistType::kTH1F, {{4, 0.0, 4.0}}});
      }
      registry.add("h_track_pt_no_collision", "no collision;p_{T,track} (GeV/#it{c});entries", {HistType::kTH1F, {{300, 0, 300}}});
      registry.add("h_track_pt_collision", "collision;p_{T,track} (GeV/#it{c});entries", {HistType::kTH1F, {{300, 0, 300}}});
    }
    if (doprocessTracksSub) {
      registry.add("h3_centrality_track_pt_track_phi_eventwiseconstituentsubtracted", "centrality vs track pT vs track #varphi; centrality; #it{p}_{T,track} (GeV/#it{c}); #varphi_{track}", {HistType::kTH3F, {{1200, -10.0, 110.0}, {200, 0., 200.}, {160, -1.0, 7.}}});
      registry.add("h3_centrality_track_pt_track_eta_eventwiseconstituentsubtracted", "centrality vs track pT vs track #eta; centrality; #it{p}_{T,track} (GeV/#it{c}); #eta_{track}", {HistType::kTH3F, {{1200, -10.0, 110.0}, {200, 0., 200.}, trackEtaAxis}});
      registry.add("h3_track_pt_track_eta_track_phi_eventwiseconstituentsubtracted", "track pT vs track #eta vs track #varphi; #it{p}_{T,track} (GeV/#it{c}); #eta_{track}; #varphi_{track}", {HistType::kTH3F, {{200, 0., 200.}, trackEtaAxis, {160, -1.0, 7.}}});
    }

    if (doprocessMCCollisionsWeighted) {
      AxisSpec weightAxis = {{VARIABLE_WIDTH, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0}, "weights"};
      registry.add("h_collision_eventweight_part", "event weight;event weight;entries", {HistType::kTH1F, {weightAxis}});
    }
    if (doprocessTracksColl) {
      registry.add("h_collisions_loop", "weight", {HistType::kTH1F, {{2, 0., 2.}}});
      registry.add("h_track_loop", "weight", {HistType::kTH1F, {{2, 0., 2.}}});
      registry.add("h_track_pt_loop", "weight track pt", {HistType::kTH1F, {{200, 0., 200.}}});
      registry.add("h_track_pt_loop_outlier", "weight track pt", {HistType::kTH1F, {{200, 0., 200.}}});
      registry.add("h2_pt_hat_track_pt_loop_outlier", "track; #hat{#it{p}_{T}} (GeV/#it{c});#it{p}_{T,track} (GeV/#it{c})", {HistType::kTH2F, {{600, 0, 600}, {150, 0, 300}}});
      registry.add("h2_neighbour_pt_hat_outlier", "neighbour; distance from collision; #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH2F, {{15, -7.5, 7.5}, {600, 0, 600}}});
      registry.add("h2_neighbour_track_pt_outlier", "neighbour; distance from collision; #it{p}_{T,track} (GeV/#it{c})", {HistType::kTH2F, {{15, -7.5, 7.5}, {200, 0, 100}}});
      registry.add("h2_neighbour_pt_hat_all", "neighbour; distance from collision; #hat{#it{p}_{T}} (GeV/#it{c})", {HistType::kTH2F, {{15, -7.5, 7.5}, {600, 0, 600}}});
      registry.add("h2_neighbour_track_pt_all", "neighbour; distance from collision; #it{p}_{T,track} (GeV/#it{c})", {HistType::kTH2F, {{15, -7.5, 7.5}, {200, 0, 100}}});
      registry.add("h_pt_hat_track_pt_loop", "loop; #hat{#it{p}_{T}} (GeV/#it{c});#it{p}_{T,track} (GeV/#it{c})", {HistType::kTH2F, {{200,0,600}, {100,0,100}}});
      registry.add("h_outlier_neighbours", "outlier", {HistType::kTH1F, {{2, 0., 2.}}});
      registry.add("h_charm_neighbours", "charm neighbour", {HistType::kTH1F, {{2, 0., 2.}}});
      registry.add("h_beauty_neighbours", "beauty neighbour", {HistType::kTH1F, {{2, 0., 2.}}});
    }
    if (doprocessTracksMatchedColl) {
      registry.add("h_outlier_neighbours_matched", "outlier", {HistType::kTH1F, {{2, 0., 2.}}});
      registry.add("h_charm_neighbours_matched", "charm neighbour", {HistType::kTH1F, {{2, 0., 2.}}});
      registry.add("h_beauty_neighbours_matched", "beauty neighbour", {HistType::kTH1F, {{2, 0., 2.}}});
    }
  }

  Filter trackCuts = (aod::jtrack::pt >= trackPtMin && aod::jtrack::pt < trackPtMax && aod::jtrack::eta > trackEtaMin && aod::jtrack::eta < trackEtaMax);
  Filter trackSubCuts = (aod::jtracksub::pt >= trackPtMin && aod::jtracksub::pt < trackPtMax && aod::jtracksub::eta > trackEtaMin && aod::jtracksub::eta < trackEtaMax);
  Filter eventCuts = (nabs(aod::jcollision::posZ) < vertexZCut && aod::jcollision::centrality >= centralityMin && aod::jcollision::centrality < centralityMax);
  PresliceUnsorted<soa::Filtered<JetCollisionsMCD>> CollisionsPerMCPCollision = aod::jmccollisionlb::mcCollisionId;

  template <typename T, typename U>
  bool isAcceptedJet(U const& jet)
  {

    if (jetAreaFractionMin > -98.0) {
      if (jet.area() < jetAreaFractionMin * M_PI * (jet.r() / 100.0) * (jet.r() / 100.0)) {
        return false;
      }
    }
    bool checkConstituentPt = true;
    bool checkConstituentMinPt = (leadingConstituentPtMin > -98.0);
    bool checkConstituentMaxPt = (leadingConstituentPtMax < 9998.0);
    if (!checkConstituentMinPt && !checkConstituentMaxPt) {
      checkConstituentPt = false;
    }

    if (checkConstituentPt) {
      bool isMinLeadingConstituent = !checkConstituentMinPt;
      bool isMaxLeadingConstituent = true;

      for (const auto& constituent : jet.template tracks_as<T>()) {
        double pt = constituent.pt();

        if (checkConstituentMinPt && pt >= leadingConstituentPtMin) {
          isMinLeadingConstituent = true;
        }
        if (checkConstituentMaxPt && pt > leadingConstituentPtMax) {
          isMaxLeadingConstituent = false;
        }
      }
      return isMinLeadingConstituent && isMaxLeadingConstituent;
    }

    return true;
  }

  template <typename T, typename U>
  bool trackIsInJet(T const& track, U const& jet)
  {
    for (auto const& constituentId : jet.tracksIds()) {
      if (constituentId == track.globalIndex()) {
        return true;
      }
    }
    return false;
  }

  template <typename T>
  void fillHistograms(T const& jet, float centrality, float occupancy, float weight = 1.0)
  {

    float pTHat = 10. / (std::pow(weight, 1.0 / pTHatExponent));
    if (jet.pt() > pTHatMaxMCD * pTHat) {
      return;
    }
    registry.fill(HIST("h_jet_phat"), pTHat);
    registry.fill(HIST("h_jet_phat_weighted"), pTHat, weight);

    if (jet.r() == round(selectedJetsRadius * 100.0f)) {
      registry.fill(HIST("h_jet_pt"), jet.pt(), weight);
      registry.fill(HIST("h_jet_eta"), jet.eta(), weight);
      registry.fill(HIST("h_jet_phi"), jet.phi(), weight);
      registry.fill(HIST("h_jet_ntracks"), jet.tracksIds().size(), weight);
      registry.fill(HIST("h2_centrality_jet_pt"), centrality, jet.pt(), weight);
      registry.fill(HIST("h2_centrality_jet_eta"), centrality, jet.eta(), weight);
      registry.fill(HIST("h2_centrality_jet_phi"), centrality, jet.phi(), weight);
      registry.fill(HIST("h2_centrality_jet_ntracks"), centrality, jet.tracksIds().size(), weight);
      registry.fill(HIST("h3_centrality_occupancy_jet_pt"), centrality, occupancy, jet.pt(), weight);
    }

    registry.fill(HIST("h3_jet_r_jet_pt_centrality"), jet.r() / 100.0, jet.pt(), centrality, weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_eta"), jet.r() / 100.0, jet.pt(), jet.eta(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_phi"), jet.r() / 100.0, jet.pt(), jet.phi(), weight);
    registry.fill(HIST("h3_jet_r_jet_eta_jet_phi"), jet.r() / 100.0, jet.eta(), jet.phi(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_ntracks"), jet.r() / 100.0, jet.pt(), jet.tracksIds().size(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_area"), jet.r() / 100.0, jet.pt(), jet.area(), weight);

    for (auto& constituent : jet.template tracks_as<JetTracksMCD>()) {

      registry.fill(HIST("h3_jet_r_jet_pt_track_pt"), jet.r() / 100.0, jet.pt(), constituent.pt(), weight);
      registry.fill(HIST("h3_jet_r_jet_pt_track_eta"), jet.r() / 100.0, jet.pt(), constituent.eta(), weight);
      registry.fill(HIST("h3_jet_r_jet_pt_track_phi"), jet.r() / 100.0, jet.pt(), constituent.phi(), weight);
      registry.fill(HIST("h3_jet_r_jet_pt_Dz"), jet.r() / 100.0, jet.pt(), constituent.pt() / jet.pt(), weight);
    }
  }
  template <typename T, typename U>
  void fillHistogramsAmbiguous(T const& jet, float weight = 1.0, U const& tracksAmbiguous = 0x0)
  {
    // take method from qaEventTrack.cxx

    float pTHat = 10. / (std::pow(weight, 1.0 / pTHatExponent));
    if (jet.pt() > pTHatMaxMCD * pTHat) {
      return;
    }
    if (jet.r() == round(selectedJetsRadius * 100.0f)) {
     // registry.fill(HIST("h_jet_pt"), jet.pt(), weight);
    }


    // test
    //LOG(info) << "================================";
    //LOG(info) << "=== soa::Filtered<TrackTableData> const& tracks, size=" << jet.template tracks_as<JetTracks>().size();
    //int iTrack = 0;
    //for (auto& t : jet.template tracks_as<JetTracks>()) {
    //  LOG(info) << "[" << iTrack << "] .index()=" << t.index() << ", .globalIndex()=" << t.globalIndex();
    //  iTrack++;
    //}
    //LOG(info) << "=== aod::AmbiguousTracks const& ambitracks, size=" << tracksAmbiguous.size();
    //int iTrackAmbi = 0;
    //for (auto& tamb : tracksAmbiguous) {
    //  LOG(info) << "[" << iTrackAmbi << "] .index()=" << tamb.index() << ", .globalIndex()=" << tamb.globalIndex() << ", .trackId()=" << tamb.trackId();
    //  iTrackAmbi++;
    //}
    //LOG(info) << "================================";


    auto iterAmbiguous = tracksAmbiguous.begin();
    bool searchAmbiguous = true;
    int nAmbTracks = 0;
    int nUnmatchedTracks = 0;
    double pt_total = 0;
    double pt_amb = 0;
    double pt_unmatched = 0;
    //LOG(info) << "===== constituent.size()=" << jet.template tracks_as<JetTracks>().size() << ", tracksAmbiguous.size()=" << tracksAmbiguous.size();
    for (auto& constituent : jet.template tracks_as<JetTracksMCD>()) {
      pt_total += constituent.pt();
      if(!constituent.has_collision()) {
        LOG(info) << "NO COLLISION : track.index()=" << constituent.index() << " track.globalIndex()=" << constituent.globalIndex();
      }
      bool has_MCparticle = constituent.has_mcParticle();
      if(!has_MCparticle) {
        //LOG(info) << "constituent NO MC PARTICLE: track.index()=" << constituent.index() << " track.globalIndex()=" << constituent.globalIndex();
        registry.fill(HIST("h3_jet_pt_track_pt_pt_hat_no_particle"), jet.pt(), constituent.pt(), pTHat, weight);
        pt_unmatched += constituent.pt();
        nUnmatchedTracks++;
      }
      else {
        //LOG(info) << "constituent HAS MC PARTICLE: track.index()=" << constituent.index() << " track.globalIndex()=" << constituent.globalIndex();
        registry.fill(HIST("h3_jet_pt_track_pt_pt_hat_with_particle"), jet.pt(), constituent.pt(), pTHat, weight);
      }


      bool goFillHisto = (iterAmbiguous != tracksAmbiguous.end());
      if (goFillHisto) {
        while (constituent.globalIndex() > iterAmbiguous.trackId()) {
          //LOG(info) << "track.index()=" << constituent.index() << " track.globalIndex()=" << constituent.globalIndex() << " ||| iterAmbiguous.trackId()=" << iterAmbiguous.trackId() << ", iterAmbiguous.globalIndex()=" << iterAmbiguous.globalIndex();
          iterAmbiguous++;
          if (iterAmbiguous == tracksAmbiguous.end()) { /// all ambiguous tracks found
            searchAmbiguous = false;
            goFillHisto = false;
            break;
          }
        }
      }
      if (goFillHisto) {
        if (constituent.globalIndex() == iterAmbiguous.trackId()) {
          nAmbTracks++;
          //LOG(info) << "   >>> FOUND AMBIGUOUS TRACK! track.index()=" << constituent.index() << " track.globalIndex()=" << constituent.globalIndex() << " ||| iterAmbiguous.trackId()=" << iterAmbiguous.trackId() << ", iterAmbiguous.globalIndex()=" << iterAmbiguous.globalIndex();
          registry.fill(HIST("h3_jet_pt_track_pt_pt_hat_ambiguous"), jet.pt(), constituent.pt(), pTHat, weight);
          pt_amb += constituent.pt();
          nAmbTracks++;
        }
        else {
          registry.fill(HIST("h3_jet_pt_track_pt_pt_hat_unambiguous"), jet.pt(), constituent.pt(), pTHat, weight);
        }
      }
    }
    registry.fill(HIST("h3_jet_pt_frac_pt_ambiguous_pt_hat"), jet.pt(), pt_amb/pt_total, pTHat, weight);
    registry.fill(HIST("h3_jet_pt_frac_constituents_ambiguous_pt_hat"), jet.pt(), double(nAmbTracks)/double(jet.template tracks_as<JetTracksMCD>().size()), pTHat, weight);

    registry.fill(HIST("h3_jet_pt_frac_pt_unmatched_particle_pt_hat"), jet.pt(), pt_unmatched/pt_total, pTHat, weight);
    registry.fill(HIST("h3_jet_pt_frac_constituents_unmatched_particle_pt_hat"), jet.pt(), double(nUnmatchedTracks)/double(jet.template tracks_as<JetTracksMCD>().size()), pTHat, weight);
  }

  template <typename T>
  void fillRhoAreaSubtractedHistograms(T const& jet, float centrality, float occupancy, float rho, float weight = 1.0)
  {
    if (jet.r() == round(selectedJetsRadius * 100.0f)) {
      registry.fill(HIST("h_jet_pt_rhoareasubtracted"), jet.pt() - (rho * jet.area()), weight);
      registry.fill(HIST("h_jet_eta_rhoareasubtracted"), jet.eta(), weight);
      registry.fill(HIST("h_jet_phi_rhoareasubtracted"), jet.phi(), weight);
      registry.fill(HIST("h_jet_ntracks_rhoareasubtracted"), jet.tracksIds().size(), weight);
      registry.fill(HIST("h2_centrality_jet_pt_rhoareasubtracted"), centrality, jet.pt() - (rho * jet.area()), weight);
      registry.fill(HIST("h3_centrality_occupancy_jet_pt_rhoareasubtracted"), centrality, occupancy, jet.pt() - (rho * jet.area()), weight);
      if (jet.pt() - (rho * jet.area()) > 0) {
        registry.fill(HIST("h2_centrality_jet_eta_rhoareasubtracted"), centrality, jet.eta(), weight);
        registry.fill(HIST("h2_centrality_jet_phi_rhoareasubtracted"), centrality, jet.phi(), weight);
        registry.fill(HIST("h2_centrality_jet_ntracks_rhoareasubtracted"), centrality, jet.tracksIds().size(), weight);
      }
    }

    registry.fill(HIST("h3_jet_r_jet_pt_centrality_rhoareasubtracted"), jet.r() / 100.0, jet.pt() - (rho * jet.area()), centrality, weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_eta_rhoareasubtracted"), jet.r() / 100.0, jet.pt() - (rho * jet.area()), jet.eta(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_phi_rhoareasubtracted"), jet.r() / 100.0, jet.pt() - (rho * jet.area()), jet.phi(), weight);
    registry.fill(HIST("h3_jet_r_jet_eta_jet_phi_rhoareasubtracted"), jet.r() / 100.0, jet.eta(), jet.phi(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_ntracks_rhoareasubtracted"), jet.r() / 100.0, jet.pt() - (rho * jet.area()), jet.tracksIds().size(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_area_rhoareasubtracted"), jet.r() / 100.0, jet.pt() - (rho * jet.area()), jet.area(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_pt_rhoareasubtracted"), jet.r() / 100.0, jet.pt(), jet.pt() - (rho * jet.area()), weight);

    for (auto& constituent : jet.template tracks_as<JetTracks>()) {

      registry.fill(HIST("h3_jet_r_jet_pt_track_pt_rhoareasubtracted"), jet.r() / 100.0, jet.pt() - (rho * jet.area()), constituent.pt(), weight);
      registry.fill(HIST("h3_jet_r_jet_pt_track_eta_rhoareasubtracted"), jet.r() / 100.0, jet.pt() - (rho * jet.area()), constituent.eta(), weight);
      registry.fill(HIST("h3_jet_r_jet_pt_track_phi_rhoareasubtracted"), jet.r() / 100.0, jet.pt() - (rho * jet.area()), constituent.phi(), weight);
    }
  }

  template <typename T>
  void fillEventWiseConstituentSubtractedHistograms(T const& jet, float centrality, float weight = 1.0)
  {
    if (jet.r() == round(selectedJetsRadius * 100.0f)) {
      registry.fill(HIST("h_jet_pt_eventwiseconstituentsubtracted"), jet.pt(), weight);
      registry.fill(HIST("h_jet_eta_eventwiseconstituentsubtracted"), jet.eta(), weight);
      registry.fill(HIST("h_jet_phi_eventwiseconstituentsubtracted"), jet.phi(), weight);
      registry.fill(HIST("h_jet_ntracks_eventwiseconstituentsubtracted"), jet.tracksIds().size(), weight);
      registry.fill(HIST("h2_centrality_jet_pt_eventwiseconstituentsubtracted"), centrality, jet.pt(), weight);
      registry.fill(HIST("h2_centrality_jet_eta_eventwiseconstituentsubtracted"), centrality, jet.eta(), weight);
      registry.fill(HIST("h2_centrality_jet_phi_eventwiseconstituentsubtracted"), centrality, jet.phi(), weight);
      registry.fill(HIST("h2_centrality_jet_ntracks_eventwiseconstituentsubtracted"), centrality, jet.tracksIds().size(), weight);
    }

    registry.fill(HIST("h3_jet_r_jet_pt_centrality_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.pt(), centrality, weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_eta_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.pt(), jet.eta(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_phi_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.pt(), jet.phi(), weight);
    registry.fill(HIST("h3_jet_r_jet_eta_jet_phi_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.eta(), jet.phi(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_ntracks_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.pt(), jet.tracksIds().size(), weight);
    registry.fill(HIST("h3_jet_r_jet_pt_jet_area_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.pt(), jet.area(), weight);

    for (auto& constituent : jet.template tracks_as<JetTracksSub>()) {

      registry.fill(HIST("h3_jet_r_jet_pt_track_pt_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.pt(), constituent.pt(), weight);
      registry.fill(HIST("h3_jet_r_jet_pt_track_eta_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.pt(), constituent.eta(), weight);
      registry.fill(HIST("h3_jet_r_jet_pt_track_phi_eventwiseconstituentsubtracted"), jet.r() / 100.0, jet.pt(), constituent.phi(), weight);
    }
  }

  template <typename T>
  void fillMCPHistograms(T const& jet, float weight = 1.0)
  {

    float pTHat = 10. / (std::pow(weight, 1.0 / pTHatExponent));
    if (jet.pt() > pTHatMaxMCP * pTHat) {
      return;
    }
    registry.fill(HIST("h_jet_phat_part"), pTHat);
    registry.fill(HIST("h_jet_phat_part_weighted"), pTHat, weight);

    if (jet.r() == round(selectedJetsRadius * 100.0f)) {
      registry.fill(HIST("h_jet_pt_part"), jet.pt(), weight);
      registry.fill(HIST("h_jet_eta_part"), jet.eta(), weight);
      registry.fill(HIST("h_jet_phi_part"), jet.phi(), weight);
      registry.fill(HIST("h_jet_ntracks_part"), jet.tracksIds().size(), weight);
    }

    registry.fill(HIST("h3_jet_r_part_jet_pt_part_jet_eta_part"), jet.r() / 100.0, jet.pt(), jet.eta(), weight);
    registry.fill(HIST("h3_jet_r_part_jet_pt_part_jet_phi_part"), jet.r() / 100.0, jet.pt(), jet.phi(), weight);
    registry.fill(HIST("h3_jet_r_part_jet_eta_part_jet_phi_part"), jet.r() / 100.0, jet.eta(), jet.phi(), weight);
    registry.fill(HIST("h3_jet_r_part_jet_pt_part_jet_ntracks_part"), jet.r() / 100.0, jet.pt(), jet.tracksIds().size(), weight);

    for (auto& constituent : jet.template tracks_as<JetParticles>()) {

      registry.fill(HIST("h3_jet_r_part_jet_pt_part_track_pt_part"), jet.r() / 100.0, jet.pt(), constituent.pt(), weight);
      registry.fill(HIST("h3_jet_r_part_jet_pt_part_track_eta_part"), jet.r() / 100.0, jet.pt(), constituent.eta(), weight);
      registry.fill(HIST("h3_jet_r_part_jet_pt_part_track_phi_part"), jet.r() / 100.0, jet.pt(), constituent.phi(), weight);
      registry.fill(HIST("h3_jet_r_part_jet_pt_part_Dz_part"), jet.r() / 100.0, jet.pt(), constituent.pt() / jet.pt(), weight);
    }
  }

  template <typename T, typename U>
  void fillMatchedHistograms(T const& jetBase, float weight = 1.0)
  {
    float pTHat = 10. / (std::pow(weight, 1.0 / pTHatExponent));
    if (jetBase.pt() > pTHatMaxMCD * pTHat) {
      return;
    }

    if (jetBase.has_matchedJetGeo()) {
      for (auto& jetTag : jetBase.template matchedJetGeo_as<std::decay_t<U>>()) {
        if (jetTag.pt() > pTHatMaxMCP * pTHat) {
          continue;
        }
        registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_pt_base_matchedgeo"), jetBase.r() / 100.0, jetTag.pt(), jetBase.pt(), weight);
        registry.fill(HIST("h3_jet_r_jet_eta_tag_jet_eta_base_matchedgeo"), jetBase.r() / 100.0, jetTag.eta(), jetBase.eta(), weight);
        registry.fill(HIST("h3_jet_r_jet_phi_tag_jet_phi_base_matchedgeo"), jetBase.r() / 100.0, jetTag.phi(), jetBase.phi(), weight);
        registry.fill(HIST("h3_jet_r_jet_ntracks_tag_jet_ntracks_base_matchedgeo"), jetBase.r() / 100.0, jetTag.tracksIds().size(), jetBase.tracksIds().size(), weight);
        registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_pt_base_diff_matchedgeo"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.pt() - jetBase.pt()) / jetTag.pt(), weight);
        registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_eta_base_diff_matchedgeo"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.eta() - jetBase.eta()) / jetTag.eta(), weight);
        registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_phi_base_diff_matchedgeo"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.phi() - jetBase.phi()) / jetTag.phi(), weight);

        for (int N = 1; N < 21; N++) {
          if (jetBase.pt() < N * 0.25 * pTHat && jetTag.pt() < N * 0.25 * pTHat) {
            registry.fill(HIST("h3_ptcut_jet_pt_tag_jet_pt_base_matchedgeo"), N * 0.25, jetTag.pt(), jetBase.pt(), weight);
          }
        }

        if (jetBase.r() == round(selectedJetsRadius * 100.0f)) {
          registry.fill(HIST("h3_jet_pt_tag_jet_eta_tag_jet_eta_base_matchedgeo"), jetTag.pt(), jetTag.eta(), jetBase.eta(), weight);
          registry.fill(HIST("h3_jet_pt_tag_jet_phi_tag_jet_phi_base_matchedgeo"), jetTag.pt(), jetTag.phi(), jetBase.phi(), weight);
          registry.fill(HIST("h3_jet_pt_tag_jet_ntracks_tag_jet_ntracks_base_matchedgeo"), jetTag.pt(), jetTag.tracksIds().size(), jetBase.tracksIds().size(), weight);
        }
      }
      registry.fill(HIST("h_matched_phat"),pTHat);
    }
    if (jetBase.has_matchedJetPt()) {
      for (auto& jetTag : jetBase.template matchedJetPt_as<std::decay_t<U>>()) {
        if (jetTag.pt() > pTHatMaxMCP * pTHat) {
          continue;
        }
        registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_pt_base_matchedpt"), jetBase.r() / 100.0, jetTag.pt(), jetBase.pt(), weight);
        registry.fill(HIST("h3_jet_r_jet_eta_tag_jet_eta_base_matchedpt"), jetBase.r() / 100.0, jetTag.eta(), jetBase.eta(), weight);
        registry.fill(HIST("h3_jet_r_jet_phi_tag_jet_phi_base_matchedpt"), jetBase.r() / 100.0, jetTag.phi(), jetBase.phi(), weight);
        registry.fill(HIST("h3_jet_r_jet_ntracks_tag_jet_ntracks_base_matchedpt"), jetBase.r() / 100.0, jetTag.tracksIds().size(), jetBase.tracksIds().size(), weight);
        registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_pt_base_diff_matchedpt"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.pt() - jetBase.pt()) / jetTag.pt(), weight);
        registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_eta_base_diff_matchedpt"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.eta() - jetBase.eta()) / jetTag.eta(), weight);
        registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_phi_base_diff_matchedpt"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.phi() - jetBase.phi()) / jetTag.phi(), weight);

        if (jetBase.r() == round(selectedJetsRadius * 100.0f)) {
          registry.fill(HIST("h3_jet_pt_tag_jet_eta_tag_jet_eta_base_matchedpt"), jetTag.pt(), jetTag.eta(), jetBase.eta(), weight);
          registry.fill(HIST("h3_jet_pt_tag_jet_phi_tag_jet_phi_base_matchedpt"), jetTag.pt(), jetTag.phi(), jetBase.phi(), weight);
          registry.fill(HIST("h3_jet_pt_tag_jet_ntracks_tag_jet_ntracks_base_matchedpt"), jetTag.pt(), jetTag.tracksIds().size(), jetBase.tracksIds().size(), weight);
        }
      }
    }

    if (jetBase.has_matchedJetGeo() && jetBase.has_matchedJetPt()) {

      for (auto& jetTag : jetBase.template matchedJetGeo_as<std::decay_t<U>>()) {
        if (jetTag.pt() > pTHatMaxMCP * pTHat) {
          continue;
        }

        if (jetBase.template matchedJetGeo_first_as<std::decay_t<U>>().globalIndex() == jetBase.template matchedJetPt_first_as<std::decay_t<U>>().globalIndex()) { // not a good way to do this
          registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_pt_base_matchedgeopt"), jetBase.r() / 100.0, jetTag.pt(), jetBase.pt(), weight);
          registry.fill(HIST("h3_jet_r_jet_eta_tag_jet_eta_base_matchedgeopt"), jetBase.r() / 100.0, jetTag.eta(), jetBase.eta(), weight);
          registry.fill(HIST("h3_jet_r_jet_phi_tag_jet_phi_base_matchedgeopt"), jetBase.r() / 100.0, jetTag.phi(), jetBase.phi(), weight);
          registry.fill(HIST("h3_jet_r_jet_ntracks_tag_jet_ntracks_base_matchedgeopt"), jetBase.r() / 100.0, jetTag.tracksIds().size(), jetBase.tracksIds().size(), weight);
          registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_pt_base_diff_matchedgeopt"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.pt() - jetBase.pt()) / jetTag.pt(), weight);
          registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_eta_base_diff_matchedgeopt"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.eta() - jetBase.eta()) / jetTag.eta(), weight);
          registry.fill(HIST("h3_jet_r_jet_pt_tag_jet_phi_base_diff_matchedgeopt"), jetBase.r() / 100.0, jetTag.pt(), (jetTag.phi() - jetBase.phi()) / jetTag.phi(), weight);

          if (jetBase.r() == round(selectedJetsRadius * 100.0f)) {
            registry.fill(HIST("h3_jet_pt_tag_jet_eta_tag_jet_eta_base_matchedgeopt"), jetTag.pt(), jetTag.eta(), jetBase.eta(), weight);
            registry.fill(HIST("h3_jet_pt_tag_jet_phi_tag_jet_phi_base_matchedgeopt"), jetTag.pt(), jetTag.phi(), jetBase.phi(), weight);
            registry.fill(HIST("h3_jet_pt_tag_jet_ntracks_tag_jet_ntracks_base_matchedgeopt"), jetTag.pt(), jetTag.tracksIds().size(), jetBase.tracksIds().size(), weight);
          }
        }
      }
    }
  }

  template <typename T, typename U>
  void fillTrackHistograms(T const& collision, U const& tracks, float weight = 1.0)
  {
    for (auto const& track : tracks) {
      //LOG(info) << " track.has_collision() = " << track.has_collision() << " track.collisionId() = "<< track.collisionId() << " collision.globalIndex() = " << collision.globalIndex(); 
      if(!track.has_collision() || track.collisionId() != collision.globalIndex()) {
        LOG(info) << "NO COLLISION : track.index()=" << track.index() << " track.globalIndex()=" << track.globalIndex();
        registry.fill(HIST("h_track_pt_no_collision"),  track.pt(), weight);
      }
      else {
        registry.fill(HIST("h_track_pt_collision"),  track.pt(), weight);
      }
      bool has_MCparticle = track.has_mcParticle();
      if(!has_MCparticle) {
        // not implemented ATM, could be added to also check tracks though if using same track selection, should be consistent with jet constituents
        //LOG(info) << "NO MC PARTICLE: track.index()=" << track.index() << " track.globalIndex()=" << track.globalIndex();
      }
      else {
        //LOG(info) << "HAS MC PARTICLE: track.index()=" << track.index() << " track.globalIndex()=" << track.globalIndex();
      }

      if (!jetderiveddatautilities::selectTrack(track, trackSelection)) {
        continue;
      }
      registry.fill(HIST("h3_centrality_track_pt_track_phi"), collision.centrality(), track.pt(), track.phi(), weight);
      registry.fill(HIST("h3_centrality_track_pt_track_eta"), collision.centrality(), track.pt(), track.eta(), weight);
      registry.fill(HIST("h3_centrality_track_pt_track_dcaxy"), collision.centrality(), track.pt(), track.dcaXY(), weight);
      registry.fill(HIST("h3_track_pt_track_eta_track_phi"), track.pt(), track.eta(), track.phi(), weight);
    }
  }

  template <typename T, typename U, typename V>
  void randomCone(T const& collision, U const& jets, V const& tracks)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelection)) {
      return;
    }
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    TRandom3 randomNumber(0);
    float randomConeEta = randomNumber.Uniform(trackEtaMin + randomConeR, trackEtaMax - randomConeR);
    float randomConePhi = randomNumber.Uniform(0.0, 2 * M_PI);
    float randomConePt = 0;
    for (auto const& track : tracks) {
      if (jetderiveddatautilities::selectTrack(track, trackSelection)) {
        float dPhi = RecoDecay::constrainAngle(track.phi() - randomConePhi, static_cast<float>(-M_PI));
        float dEta = track.eta() - randomConeEta;
        if (TMath::Sqrt(dEta * dEta + dPhi * dPhi) < randomConeR) {
          randomConePt += track.pt();
        }
      }
    }
    registry.fill(HIST("h2_centrality_rhorandomcone"), collision.centrality(), randomConePt - M_PI * randomConeR * randomConeR * collision.rho());

    // randomised eta,phi for tracks, to assess part of fluctuations coming from statistically independently emitted particles
    randomConePt = 0;
    for (auto const& track : tracks) {
      if (jetderiveddatautilities::selectTrack(track, trackSelection)) {
        float dPhi = RecoDecay::constrainAngle(randomNumber.Uniform(0.0, 2 * M_PI) - randomConePhi, static_cast<float>(-M_PI)); // ignores actual phi of track
        float dEta = randomNumber.Uniform(trackEtaMin, trackEtaMax) - randomConeEta;                                            // ignores actual eta of track
        if (TMath::Sqrt(dEta * dEta + dPhi * dPhi) < randomConeR) {
          randomConePt += track.pt();
        }
      }
    }
    registry.fill(HIST("h2_centrality_rhorandomconerandomtrackdirection"), collision.centrality(), randomConePt - M_PI * randomConeR * randomConeR * collision.rho());

    // removing the leading jet from the random cone
    if (jets.size() > 0) { // if there are no jets in the acceptance (from the jetfinder cuts) then there can be no leading jet
      float dPhiLeadingJet = RecoDecay::constrainAngle(jets.iteratorAt(0).phi() - randomConePhi, static_cast<float>(-M_PI));
      float dEtaLeadingJet = jets.iteratorAt(0).eta() - randomConeEta;

      bool jetWasInCone = false;
      while ((randomConeLeadJetDeltaR <= 0 && (TMath::Sqrt(dEtaLeadingJet * dEtaLeadingJet + dPhiLeadingJet * dPhiLeadingJet) < jets.iteratorAt(0).r() / 100.0 + randomConeR)) || (randomConeLeadJetDeltaR > 0 && (TMath::Sqrt(dEtaLeadingJet * dEtaLeadingJet + dPhiLeadingJet * dPhiLeadingJet) < randomConeLeadJetDeltaR))) {
        jetWasInCone = true;
        randomConeEta = randomNumber.Uniform(trackEtaMin + randomConeR, trackEtaMax - randomConeR);
        randomConePhi = randomNumber.Uniform(0.0, 2 * M_PI);
        dPhiLeadingJet = RecoDecay::constrainAngle(jets.iteratorAt(0).phi() - randomConePhi, static_cast<float>(-M_PI));
        dEtaLeadingJet = jets.iteratorAt(0).eta() - randomConeEta;
      }
      if (jetWasInCone) {
        randomConePt = 0.0;
        for (auto const& track : tracks) {
          if (jetderiveddatautilities::selectTrack(track, trackSelection)) { // if track selection is uniformTrack, dcaXY and dcaZ cuts need to be added as they aren't in the selection so that they can be studied here
            float dPhi = RecoDecay::constrainAngle(track.phi() - randomConePhi, static_cast<float>(-M_PI));
            float dEta = track.eta() - randomConeEta;
            if (TMath::Sqrt(dEta * dEta + dPhi * dPhi) < randomConeR) {
              randomConePt += track.pt();
            }
          }
        }
      }
    }

    registry.fill(HIST("h2_centrality_rhorandomconewithoutleadingjet"), collision.centrality(), randomConePt - M_PI * randomConeR * randomConeR * collision.rho());

    // randomised eta,phi for tracks, to assess part of fluctuations coming from statistically independently emitted particles, removing tracks from 2 leading jets
    double randomConePtWithoutOneLeadJet = 0;
    double randomConePtWithoutTwoLeadJet = 0;
    for (auto const& track : tracks) {
      if (jetderiveddatautilities::selectTrack(track, trackSelection)) {
        float dPhi = RecoDecay::constrainAngle(randomNumber.Uniform(0.0, 2 * M_PI) - randomConePhi, static_cast<float>(-M_PI)); // ignores actual phi of track
        float dEta = randomNumber.Uniform(trackEtaMin, trackEtaMax) - randomConeEta;                                            // ignores actual eta of track
        if (TMath::Sqrt(dEta * dEta + dPhi * dPhi) < randomConeR) {
          if (!trackIsInJet(track, jets.iteratorAt(0))) {
            randomConePtWithoutOneLeadJet += track.pt();
            if (!trackIsInJet(track, jets.iteratorAt(1))) {
              randomConePtWithoutTwoLeadJet += track.pt();
            }
          }
        }
      }
    }
    registry.fill(HIST("h2_centrality_rhorandomconerandomtrackdirectionwithoutoneleadingjets"), collision.centrality(), randomConePtWithoutOneLeadJet - M_PI * randomConeR * randomConeR * collision.rho());
    registry.fill(HIST("h2_centrality_rhorandomconerandomtrackdirectionwithouttwoleadingjets"), collision.centrality(), randomConePtWithoutTwoLeadJet - M_PI * randomConeR * randomConeR * collision.rho());
  }

  void processJetsData(soa::Filtered<JetCollisions>::iterator const& collision, soa::Join<aod::ChargedJets, aod::ChargedJetConstituents> const& jets, JetTracks const&)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (auto const& jet : jets) {
      if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracks>(jet)) {
        continue;
      }
      fillHistograms(jet, collision.centrality(), collision.trackOccupancyInTimeRange());
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsData, "jet finder QA data", false);

  void processJetsRhoAreaSubData(soa::Filtered<soa::Join<JetCollisions, aod::BkgChargedRhos>>::iterator const& collision,
                                 soa::Join<aod::ChargedJets, aod::ChargedJetConstituents> const& jets,
                                 JetTracks const&)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (auto jet : jets) {
      if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracks>(jet)) {
        continue;
      }
      fillRhoAreaSubtractedHistograms(jet, collision.centrality(), collision.trackOccupancyInTimeRange(), collision.rho());
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsRhoAreaSubData, "jet finder QA for rho-area subtracted jets", false);

  void processJetsRhoAreaSubMCD(soa::Filtered<soa::Join<JetCollisions, aod::BkgChargedRhos>>::iterator const& collision,
                                soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents> const& jets,
                                JetTracks const&)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (auto jet : jets) {
      if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracks>(jet)) {
        continue;
      }
      fillRhoAreaSubtractedHistograms(jet, collision.centrality(), collision.trackOccupancyInTimeRange(), collision.rho());
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsRhoAreaSubMCD, "jet finder QA for rho-area subtracted mcd jets", false);

  void processEvtWiseConstSubJetsData(soa::Filtered<JetCollisions>::iterator const& collision, soa::Join<aod::ChargedEventWiseSubtractedJets, aod::ChargedEventWiseSubtractedJetConstituents> const& jets, JetTracksSub const&)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (auto const& jet : jets) {
      if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracksSub>(jet)) {
        continue;
      }
      fillEventWiseConstituentSubtractedHistograms(jet, collision.centrality());
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processEvtWiseConstSubJetsData, "jet finder QA for eventwise constituent-subtracted jets data", false);

  void processEvtWiseConstSubJetsMCD(soa::Filtered<JetCollisions>::iterator const& collision, soa::Join<aod::ChargedMCDetectorLevelEventWiseSubtractedJets, aod::ChargedMCDetectorLevelEventWiseSubtractedJetConstituents> const& jets, JetTracksSub const&)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (auto const& jet : jets) {
      if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracksSub>(jet)) {
        continue;
      }
      fillEventWiseConstituentSubtractedHistograms(jet, collision.centrality());
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processEvtWiseConstSubJetsMCD, "jet finder QA for eventwise constituent-subtracted mcd jets", false);

  void processJetsSubMatched(soa::Filtered<JetCollisions>::iterator const& collision,
                             soa::Join<aod::ChargedJets, aod::ChargedJetConstituents, aod::ChargedJetsMatchedToChargedEventWiseSubtractedJets> const& jets,
                             soa::Join<aod::ChargedEventWiseSubtractedJets, aod::ChargedEventWiseSubtractedJetConstituents, aod::ChargedEventWiseSubtractedJetsMatchedToChargedJets> const&,
                             JetTracks const&, JetTracksSub const&)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (const auto& jet : jets) {
      if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracks>(jet)) {
        continue;
      }
      fillMatchedHistograms<soa::Join<aod::ChargedJets, aod::ChargedJetConstituents, aod::ChargedJetsMatchedToChargedEventWiseSubtractedJets>::iterator, soa::Join<aod::ChargedEventWiseSubtractedJets, aod::ChargedEventWiseSubtractedJetConstituents, aod::ChargedEventWiseSubtractedJetsMatchedToChargedJets>>(jet);
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsSubMatched, "jet finder QA matched unsubtracted and constituent subtracted jets", false);

  void processJetsMCD(soa::Filtered<JetCollisions>::iterator const& collision, soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents> const& jets, JetTracksMCD const&, const aod::AmbiguousTracks& tracksAmbiguous)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (auto const& jet : jets) {
      if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracks>(jet)) {
        continue;
      }
      fillHistograms(jet, collision.centrality(), collision.trackOccupancyInTimeRange());
      if(isFillAmbiguous) fillHistogramsAmbiguous(jet, 1., tracksAmbiguous);
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsMCD, "jet finder QA mcd", false);

  void processTracksColl(soa::Join<JetCollisions, aod::JMcCollisionLbs> const& collisions, 
      JetMcCollisions const&,
      JetParticles const& particles,
      JetTracks const& tracks,
      soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetEventWeights> const& jets, 
      JetTracksMCD const&
      )
  {
    for (auto const& collision : collisions) {

      registry.fill(HIST("h_collisions_loop"), 0.5);
      if (!jetderiveddatautilities::selectCollision(collision, eventSelection)) {
        continue;
      }
      registry.fill(HIST("h_collisions_loop"), 1.5);
      if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
        continue;
      }
      float eventWeight = collision.mcCollision().weight();
      double pTHat = 10. / (std::pow(eventWeight, 1.0 / pTHatExponent));
      const auto tracksColl = tracks.sliceBy(perCol, collision.globalIndex());
      //LOG(info) << "coll index = " << collision.globalIndex();
      //LOG(info) << "n tracks = " << tracksColl.size();
      //
      // find / reject outlier collisions
      //
      bool jetCollisionOK = true;
      bool charmCollisionOK = true;
      bool beautyCollisionOK = true;
      for (auto const& collisionOutlier : collisions) { // find collisions closeby
        float eventWeightOutlier = collisionOutlier.mcCollision().weight();
        double pTHatOutlier = 10. / (std::pow(eventWeightOutlier, 1.0 / pTHatExponent));
        int diffColl = collision.globalIndex() - collisionOutlier.globalIndex();

        //LOG(info) << "collision global index = " << collision.globalIndex() << "outlier global index = "<< collisionOutlier.globalIndex();
        if( diffColl > -5 && diffColl < 0) { // check 5 collisions prior
          //LOG(info) << "pThat = " << pTHat << "pThat neighbour = "<<pTHatOutlier;
          //if(pTHatOutlier - pTHat > diffNeighbours ) { 
          //  // choose events that are close to the current event
          //  // doesn't work as the rejection factor varies with the current event pThat
          //  jetCollisionOK = false;
          //  //break;
          //}
          if(pTHatOutlier > diffNeighbours ) { 
            jetCollisionOK = false;
          }

          const auto particlesColl = particles.sliceBy(perColMC, collisionOutlier.globalIndex());
          //LOG(info) << "Particles coll size = "<<particlesColl.size();
          //LOG(info) << "Particles size = "<<particles.size();
          for(auto const &particle : particlesColl) {
            int pdgCode = particle.pdgCode();
            //LOG(info) << "pdg code = "<<pdgCode;
            if(std::abs(pdgCode) == 4) {
              charmCollisionOK = false;
            }
            if(std::abs(pdgCode) == 5) {
              beautyCollisionOK = false;
            }
          }
        }
      }
      registry.fill(HIST("h_outlier_neighbours"), jetCollisionOK?1.5:0.5);
      registry.fill(HIST("h_charm_neighbours"), charmCollisionOK?1.5:0.5);
      registry.fill(HIST("h_beauty_neighbours"), beautyCollisionOK?1.5:0.5);
      if(!jetCollisionOK) {
        //LOG(info) << "outlier close by";
        continue;
      }
      if(isRejectHFNeighbours && (!charmCollisionOK || !beautyCollisionOK)) {
        continue;
      }

      // 
      // Check Tracks
      //

      for (auto const& track : tracksColl) {

        registry.fill(HIST("h_track_loop"), 0.5);
        if (!jetderiveddatautilities::selectTrack(track, trackSelection)) {
          continue;
        }
        registry.fill(HIST("h_track_loop"), 1.5);

        registry.fill(HIST("h_track_pt_loop"), track.pt());
        registry.fill(HIST("h_pt_hat_track_pt_loop"), pTHat, track.pt());

        if(track.pt() < 100 && track.pt() > 1.5 * pTHat) { // high weight outlier track
          registry.fill(HIST("h_track_pt_loop_outlier"), track.pt());
          registry.fill(HIST("h2_pt_hat_track_pt_loop_outlier"),pTHat, track.pt());
          for (auto const& collisionOutlier : collisions) { // find collisions closeby
            float eventWeightOutlier = collisionOutlier.mcCollision().weight();
            double pTHatOutlier = 10. / (std::pow(eventWeightOutlier, 1.0 / pTHatExponent));
            int diffColl = collision.globalIndex() - collisionOutlier.globalIndex();

            if(abs(diffColl) < 6) {

              registry.fill(HIST("h2_neighbour_pt_hat_outlier"), float(diffColl+0.1), pTHatOutlier);
              registry.fill(HIST("h2_neighbour_track_pt_outlier"), float(diffColl+0.1), track.pt());

            }
          }
        }
        // all
        for (auto const& collisionOutlier : collisions) { // find collisions closeby
          float eventWeightOutlier = collisionOutlier.mcCollision().weight();
          double pTHatOutlier = 10. / (std::pow(eventWeightOutlier, 1.0 / pTHatExponent));
          int diffColl = collision.globalIndex() - collisionOutlier.globalIndex();

          if(abs(diffColl) < 6) {
            //LOG(info) << "pThat = " << pTHat << "pThat neighbour = "<<pTHatOutlier;
            registry.fill(HIST("h2_neighbour_pt_hat_all"), float(diffColl+0.1), pTHatOutlier);
            registry.fill(HIST("h2_neighbour_track_pt_all"), float(diffColl+0.1), track.pt());
          }
        }
      }
      // 
      // Check MC particles (search for c and b)
      //
      bool foundCharm = false;
      bool foundBeauty= false;
      const auto particlesColl = particles.sliceBy(perColMC, collision.globalIndex());
      //LOG(info) << "Particles coll size = "<<particlesColl.size();
      //LOG(info) << "Particles size = "<<particles.size();
      for(auto const &particle : particlesColl) {
        int pdgCode = particle.pdgCode();
        //LOG(info) << "pdg code = "<<pdgCode;
        if(std::abs(pdgCode) == 4) {
          foundCharm = true;
        }
        if(std::abs(pdgCode) == 5) {
          foundBeauty = true;
        }
      }
      //LOG(info) << "Charm  = "<<foundCharm;
      //LOG(info) << "Beauty = "<<foundBeauty;


      //
      // Jets
      //
      const auto jetsColl = jets.sliceBy(perColJets, collision.globalIndex());
      for (auto const& jet : jetsColl) {
        if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
          continue;
        }
        if (!isAcceptedJet<JetTracks>(jet)) {
          continue;
        }
        if(!isMB && jet.eventWeight()==1) {
          continue;
        }

        fillHistograms(jet, collision.centrality(), collision.trackOccupancyInTimeRange(), jet.eventWeight());
      }
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processTracksColl, "jet finder QA mcd", false);

//    /// work with collision grouping
//  for (auto const& collision : collisions) {
//    const auto& tracksColl = tracks.sliceBy(perRecoCollision, collision.globalIndex());
//    const auto& tracksUnfilteredColl = tracksUnfiltered.sliceBy(perRecoCollision, collision.globalIndex());
//    fillRecoHistogramsGroupedTracks<false>(collision, tracksColl, tracksUnfilteredColl);
//  }

  void processJetsMCDWeighted(soa::Filtered<JetCollisions>::iterator const& collision, soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetEventWeights> const& jets, JetTracksMCD const&, const aod::AmbiguousTracks& tracksAmbiguous)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (auto const& jet : jets) {
      if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracks>(jet)) {
        continue;
      }
      //LOG(info) << "isMB =  " << isMB << " event weight = " << jet.eventWeight();
      if(!isMB && jet.eventWeight()==1) {
        continue;
      }
      double pTHat = 10. / (std::pow(jet.eventWeight(), 1.0 / pTHatExponent));
      for (int N = 1; N < 21; N++) {
        if (jet.pt() < N * 0.25 * pTHat && jet.r() == round(selectedJetsRadius * 100.0f)) {
          registry.fill(HIST("h_jet_ptcut"), jet.pt(), N * 0.25, jet.eventWeight());
        }
      }
      fillHistograms(jet, collision.centrality(), collision.trackOccupancyInTimeRange(), jet.eventWeight());
      if(isFillAmbiguous) fillHistogramsAmbiguous(jet, jet.eventWeight(), tracksAmbiguous);
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsMCDWeighted, "jet finder QA mcd with weighted events", false);

  void processJetsMCP(soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents>::iterator const& jet, JetParticles const&, JetMcCollisions const&, soa::Filtered<JetCollisionsMCD> const& collisions)
  {
    if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
      return;
    }
    if (!isAcceptedJet<JetParticles>(jet)) {
      return;
    }
    if (checkMcCollisionIsMatched) {
      auto collisionspermcpjet = collisions.sliceBy(CollisionsPerMCPCollision, jet.mcCollisionId());
      if (collisionspermcpjet.size() >= 1 && jetderiveddatautilities::selectCollision(collisionspermcpjet.begin(), eventSelection)) {
        fillMCPHistograms(jet);
      }
    } else {
      fillMCPHistograms(jet);
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsMCP, "jet finder QA mcp", false);

  void processJetsMCPWeighted(soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents, aod::ChargedMCParticleLevelJetEventWeights>::iterator const& jet, JetParticles const&, JetMcCollisions const&, soa::Filtered<JetCollisionsMCD> const& collisions)
  {
    if (!jetfindingutilities::isInEtaAcceptance(jet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
      return;
    }
    if (!isAcceptedJet<JetParticles>(jet)) {
      return;
    }
    if(!isMB && jet.eventWeight()==1) {
      return;
    }
    double pTHat = 10. / (std::pow(jet.eventWeight(), 1.0 / pTHatExponent));
    for (int N = 1; N < 21; N++) {
      if (jet.pt() < N * 0.25 * pTHat && jet.r() == round(selectedJetsRadius * 100.0f)) {
        registry.fill(HIST("h_jet_ptcut_part"), jet.pt(), N * 0.25, jet.eventWeight());
      }
    }
    if (checkMcCollisionIsMatched) {
      auto collisionspermcpjet = collisions.sliceBy(CollisionsPerMCPCollision, jet.mcCollisionId());
      if (collisionspermcpjet.size() >= 1) {
        fillMCPHistograms(jet, jet.eventWeight());
      }
    } else {
      fillMCPHistograms(jet, jet.eventWeight());
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsMCPWeighted, "jet finder QA mcp with weighted events", false);

  void processJetsMCPMCDMatched(soa::Filtered<JetCollisions>::iterator const& collision,
                                soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetsMatchedToChargedMCParticleLevelJets> const& mcdjets,
                                soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents, aod::ChargedMCParticleLevelJetsMatchedToChargedMCDetectorLevelJets> const&,
                                JetTracks const&, JetParticles const&)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (const auto& mcdjet : mcdjets) {
      if (!jetfindingutilities::isInEtaAcceptance(mcdjet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracks>(mcdjet)) {
        continue;
      }
      fillMatchedHistograms<soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetsMatchedToChargedMCParticleLevelJets>::iterator, soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents, aod::ChargedMCParticleLevelJetsMatchedToChargedMCDetectorLevelJets>>(mcdjet);
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsMCPMCDMatched, "jet finder QA matched mcp and mcd", false);


  void processTracksMatchedColl(soa::Join<JetCollisions, aod::JMcCollisionLbs> const& collisions, 
      JetMcCollisions const&,
      soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetsMatchedToChargedMCParticleLevelJets, aod::ChargedMCDetectorLevelJetEventWeights> const& mcdjets,
      soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents, aod::ChargedMCParticleLevelJetsMatchedToChargedMCDetectorLevelJets, aod::ChargedMCParticleLevelJetEventWeights> const&,
      JetParticles const& particles,
      JetTracks const& tracks
      )
  {
    for (auto const& collision : collisions) {

      if (!jetderiveddatautilities::selectCollision(collision, eventSelection)) {
        continue;
      }
      if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
        continue;
      }
      float eventWeight = collision.mcCollision().weight();
      double pTHat = 10. / (std::pow(eventWeight, 1.0 / pTHatExponent));
      const auto tracksColl = tracks.sliceBy(perCol, collision.globalIndex());
      //
      // find / reject outlier collisions
      //
      bool jetCollisionOK = true;
      bool charmCollisionOK = true;
      bool beautyCollisionOK = true;
      //LOG(info) << "n collisions = "<<collisions.size();
      for (auto const& collisionOutlier : collisions) { // find collisions closeby
        float eventWeightOutlier = collisionOutlier.mcCollision().weight();
        double pTHatOutlier = 10. / (std::pow(eventWeightOutlier, 1.0 / pTHatExponent));
        int diffColl = collision.globalIndex() - collisionOutlier.globalIndex();
        if( diffColl > -5 && diffColl < 0) { // check 5 collisions prior
          if(pTHatOutlier > diffNeighbours ) { 
            jetCollisionOK = false;
          }
          //LOG(info) << "particles size= "<<particles.size();
          const auto particlesColl = particles.sliceBy(perColMC, collisionOutlier.globalIndex());
          // LOG(info) << "particles coll size= "<<particlesColl.size();
          for(auto const &particle : particlesColl) {
            int pdgCode = particle.pdgCode();
            if(std::abs(pdgCode) == 4) {
              charmCollisionOK = false;
            }
            if(std::abs(pdgCode) == 5) {
              beautyCollisionOK = false;
            }
          }
        }
      }
      registry.fill(HIST("h_outlier_neighbours_matched"), jetCollisionOK?1.5:0.5);
      registry.fill(HIST("h_charm_neighbours_matched"), charmCollisionOK?1.5:0.5);
      registry.fill(HIST("h_beauty_neighbours_matched"), beautyCollisionOK?1.5:0.5);
      if(!jetCollisionOK) {
        continue;
      }
      if(isRejectHFNeighbours && (!charmCollisionOK || !beautyCollisionOK)) {
        continue;
      }

      const auto mcdjetsColl = mcdjets.sliceBy(perColJetsMatched, collision.globalIndex());
      for (auto const& mcdjet : mcdjetsColl) {
        //  for (const auto& mcdjet : mcdjets) {
        if (!jetfindingutilities::isInEtaAcceptance(mcdjet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
          continue;
        }
        if (!isAcceptedJet<JetTracks>(mcdjet)) {
          continue;
        }
        if(!isMB && mcdjet.eventWeight()==1) {
          continue;
        }
        fillMatchedHistograms<soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetsMatchedToChargedMCParticleLevelJets, aod::ChargedMCDetectorLevelJetEventWeights>::iterator, soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents, aod::ChargedMCParticleLevelJetsMatchedToChargedMCDetectorLevelJets, aod::ChargedMCParticleLevelJetEventWeights>>(mcdjet, mcdjet.eventWeight());
      }
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processTracksMatchedColl, "jet finder QA matched mcp and mcd with weighted events, neighbour rejection", false);


  void processJetsMCPMCDMatchedWeighted(soa::Filtered<JetCollisions>::iterator const& collision,
                                        soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetsMatchedToChargedMCParticleLevelJets, aod::ChargedMCDetectorLevelJetEventWeights> const& mcdjets,
                                        soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents, aod::ChargedMCParticleLevelJetsMatchedToChargedMCDetectorLevelJets, aod::ChargedMCParticleLevelJetEventWeights> const&,
                                        JetTracks const&, JetParticles const&)
  {
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (const auto& mcdjet : mcdjets) {
      if (!jetfindingutilities::isInEtaAcceptance(mcdjet, jetEtaMin, jetEtaMax, trackEtaMin, trackEtaMax)) {
        continue;
      }
      if (!isAcceptedJet<JetTracks>(mcdjet)) {
        continue;
      }
      if(!isMB && mcdjet.eventWeight()==1) {
        continue;
      }
      fillMatchedHistograms<soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetsMatchedToChargedMCParticleLevelJets, aod::ChargedMCDetectorLevelJetEventWeights>::iterator, soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents, aod::ChargedMCParticleLevelJetsMatchedToChargedMCDetectorLevelJets, aod::ChargedMCParticleLevelJetEventWeights>>(mcdjet, mcdjet.eventWeight());
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processJetsMCPMCDMatchedWeighted, "jet finder QA matched mcp and mcd with weighted events", false);

  void processMCCollisionsWeighted(JetMcCollision const& collision)
  {
    if(!isMB && collision.weight()==1) {
      return;
    }
    registry.fill(HIST("h_collision_eventweight_part"), collision.weight());
  }
  PROCESS_SWITCH(JetFinderQATask, processMCCollisionsWeighted, "collision QA for weighted events", false);

  void processTriggeredData(soa::Join<JetCollisions, aod::JChTrigSels>::iterator const& collision,
                            soa::Join<aod::ChargedJets, aod::ChargedJetConstituents> const& jets,
                            soa::Filtered<JetTracks> const& tracks)
  {
    registry.fill(HIST("h_collision_trigger_events"), 0.5); // all events
    if (collision.posZ() > vertexZCut) {
      return;
    }
    registry.fill(HIST("h_collision_trigger_events"), 1.5); // all events with z vertex cut
    if (!jetderiveddatautilities::selectCollision(collision, eventSelection)) {
      return;
    }
    registry.fill(HIST("h_collision_trigger_events"), 2.5); // events with sel8()
    if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt)) {
      registry.fill(HIST("h_collision_trigger_events"), 3.5); // events with high pT triggered jets
    }
    if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt)) {
      registry.fill(HIST("h_collision_trigger_events"), 4.5); // events with high pT triggered jets
    }
    if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt) && jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt)) {
      registry.fill(HIST("h_collision_trigger_events"), 5.5); // events with high pT triggered jets
    }

    for (std::size_t iJetRadius = 0; iJetRadius < jetRadiiValues.size(); iJetRadius++) {
      filledJetR_Both[iJetRadius] = false;
      filledJetR_Low[iJetRadius] = false;
      filledJetR_High[iJetRadius] = false;
    }
    for (auto& jet : jets) {
      for (std::size_t iJetRadius = 0; iJetRadius < jetRadiiValues.size(); iJetRadius++) {
        if (jet.r() == round(jetRadiiValues[iJetRadius] * 100.0f)) {
          if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt) && !filledJetR_Low[iJetRadius]) {
            filledJetR_Low[iJetRadius] = true;
            for (double pt = 0.0; pt <= jet.pt(); pt += 1.0) {
              registry.fill(HIST("h2_jet_r_jet_pT_triggered_Low"), jet.r() / 100.0, pt);
            }
          }
          if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt) && !filledJetR_High[iJetRadius]) {
            filledJetR_High[iJetRadius] = true;
            for (double pt = 0.0; pt <= jet.pt(); pt += 1.0) {
              registry.fill(HIST("h2_jet_r_jet_pT_triggered_High"), jet.r() / 100.0, pt);
            }
          }
          if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt) && jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt) && !filledJetR_Both[iJetRadius]) {
            filledJetR_Both[iJetRadius] = true;
            for (double pt = 0.0; pt <= jet.pt(); pt += 1.0) {
              registry.fill(HIST("h2_jet_r_jet_pT_triggered_Both"), jet.r() / 100.0, pt);
            }
          }
        }
      }

      if ((jet.eta() < (trackEtaMin + jet.r() / 100.0)) || (jet.eta() > (trackEtaMax - jet.r() / 100.0))) {
        continue;
      }

      registry.fill(HIST("h3_jet_r_jet_pt_collision"), jet.r() / 100.0, jet.pt(), 0.0);
      registry.fill(HIST("h3_jet_r_jet_eta_collision"), jet.r() / 100.0, jet.eta(), 0.0);
      registry.fill(HIST("h3_jet_r_jet_phi_collision"), jet.r() / 100.0, jet.phi(), 0.0);

      if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt)) {
        registry.fill(HIST("h3_jet_r_jet_pt_collision"), jet.r() / 100.0, jet.pt(), 1.0);
        registry.fill(HIST("h3_jet_r_jet_eta_collision"), jet.r() / 100.0, jet.eta(), 1.0);
        registry.fill(HIST("h3_jet_r_jet_phi_collision"), jet.r() / 100.0, jet.phi(), 1.0);
      }
      if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt)) {
        registry.fill(HIST("h3_jet_r_jet_pt_collision"), jet.r() / 100.0, jet.pt(), 2.0);
        registry.fill(HIST("h3_jet_r_jet_eta_collision"), jet.r() / 100.0, jet.eta(), 2.0);
        registry.fill(HIST("h3_jet_r_jet_phi_collision"), jet.r() / 100.0, jet.phi(), 2.0);
      }
      if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt) && jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt)) {
        registry.fill(HIST("h3_jet_r_jet_pt_collision"), jet.r() / 100.0, jet.pt(), 3.0);
        registry.fill(HIST("h3_jet_r_jet_eta_collision"), jet.r() / 100.0, jet.eta(), 3.0);
        registry.fill(HIST("h3_jet_r_jet_phi_collision"), jet.r() / 100.0, jet.phi(), 3.0);
      }

      for (auto& constituent : jet.template tracks_as<soa::Filtered<JetTracks>>()) {
        registry.fill(HIST("h3_jet_r_jet_pt_track_pt_MB"), jet.r() / 100.0, jet.pt(), constituent.pt());
        registry.fill(HIST("h3_jet_r_jet_pt_track_eta_MB"), jet.r() / 100.0, jet.pt(), constituent.eta());
        registry.fill(HIST("h3_jet_r_jet_pt_track_phi_MB"), jet.r() / 100.0, jet.pt(), constituent.phi());

        if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt)) {
          registry.fill(HIST("h3_jet_r_jet_pt_track_pt_Triggered_Low"), jet.r() / 100.0, jet.pt(), constituent.pt());
          registry.fill(HIST("h3_jet_r_jet_pt_track_eta_Triggered_Low"), jet.r() / 100.0, jet.pt(), constituent.eta());
          registry.fill(HIST("h3_jet_r_jet_pt_track_phi_Triggered_Low"), jet.r() / 100.0, jet.pt(), constituent.phi());
        }
        if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt)) {
          registry.fill(HIST("h3_jet_r_jet_pt_track_pt_Triggered_High"), jet.r() / 100.0, jet.pt(), constituent.pt());
          registry.fill(HIST("h3_jet_r_jet_pt_track_eta_Triggered_High"), jet.r() / 100.0, jet.pt(), constituent.eta());
          registry.fill(HIST("h3_jet_r_jet_pt_track_phi_Triggered_High"), jet.r() / 100.0, jet.pt(), constituent.phi());
        }
        if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt) && jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt)) {
          registry.fill(HIST("h3_jet_r_jet_pt_track_pt_Triggered_Both"), jet.r() / 100.0, jet.pt(), constituent.pt());
          registry.fill(HIST("h3_jet_r_jet_pt_track_eta_Triggered_Both"), jet.r() / 100.0, jet.pt(), constituent.eta());
          registry.fill(HIST("h3_jet_r_jet_pt_track_phi_Triggered_Both"), jet.r() / 100.0, jet.pt(), constituent.phi());
        }
      }
    }

    for (auto& track : tracks) {
      if (!jetderiveddatautilities::selectTrack(track, trackSelection)) {
        continue;
      }
      registry.fill(HIST("h_track_pt_MB"), track.pt());
      registry.fill(HIST("h_track_eta_MB"), track.eta());
      registry.fill(HIST("h_track_phi_MB"), track.phi());

      if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt)) {
        registry.fill(HIST("h_track_pt_Triggered_Low"), track.pt());
        registry.fill(HIST("h_track_eta_Triggered_Low"), track.eta());
        registry.fill(HIST("h_track_phi_Triggered_Low"), track.phi());
      }
      if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt)) {
        registry.fill(HIST("h_track_pt_Triggered_High"), track.pt());
        registry.fill(HIST("h_track_eta_Triggered_High"), track.eta());
        registry.fill(HIST("h_track_phi_Triggered_High"), track.phi());
      }
      if (jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChLowPt) && jetderiveddatautilities::selectTrigger(collision, jetderiveddatautilities::JTrigSel::JetChHighPt)) {
        registry.fill(HIST("h_track_pt_Triggered_Both"), track.pt());
        registry.fill(HIST("h_track_eta_Triggered_Both"), track.eta());
        registry.fill(HIST("h_track_phi_Triggered_Both"), track.phi());
      }
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processTriggeredData, "QA for charged jet trigger", false);

  void processTracks(soa::Join<JetCollisions, aod::JMcCollisionLbs>::iterator const& collision,
                             JetMcCollisions const&,
                             soa::Filtered<soa::Join<JetTracks, aod::JTrackExtras, aod::JMcTrackLbs>> const& tracks)
  {
    float eventWeight = collision.mcCollision().weight();
    if(eventWeight!=1) {
      return;
    }
    registry.fill(HIST("h_collisions"), 0.5);
    registry.fill(HIST("h2_centrality_collisions"), collision.centrality(), 0.5);
    if (!jetderiveddatautilities::selectCollision(collision, eventSelection)) {
      return;
    }
    registry.fill(HIST("h_collisions"), 1.5);
    registry.fill(HIST("h2_centrality_collisions"), collision.centrality(), 1.5);
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    registry.fill(HIST("h_collisions"), 2.5);
    registry.fill(HIST("h2_centrality_collisions"), collision.centrality(), 2.5);
    fillTrackHistograms(collision, tracks);
  }
  PROCESS_SWITCH(JetFinderQATask, processTracks, "QA for charged tracks", false);

  void processTracksWeighted(soa::Join<JetCollisions, aod::JMcCollisionLbs>::iterator const& collision,
                             JetMcCollisions const&,
                             soa::Filtered<soa::Join<JetTracks, aod::JTrackExtras, aod::JMcTrackLbs>> const& tracks)
  {
    float eventWeight = collision.mcCollision().weight();
    if(!isMB && eventWeight==1) {
      return;
    }
    registry.fill(HIST("h_collisions"), 0.5);
    registry.fill(HIST("h_collisions_weighted"), 0.5, eventWeight);
    if (!jetderiveddatautilities::selectCollision(collision, eventSelection)) {
      return;
    }
    registry.fill(HIST("h_collisions"), 1.5);
    registry.fill(HIST("h_collisions_weighted"), 1.5, eventWeight);
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    registry.fill(HIST("h_collisions"), 2.5);
    registry.fill(HIST("h_collisions_weighted"), 2.5, eventWeight);
    fillTrackHistograms(collision, tracks, eventWeight);
  }
  PROCESS_SWITCH(JetFinderQATask, processTracksWeighted, "QA for charged tracks weighted", false);

  void processTracksSub(soa::Filtered<JetCollisions>::iterator const& collision,
                        soa::Filtered<JetTracksSub> const& tracks)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelection)) {
      return;
    }
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    for (auto const& track : tracks) {
      registry.fill(HIST("h3_centrality_track_pt_track_phi_eventwiseconstituentsubtracted"), collision.centrality(), track.pt(), track.phi());
      registry.fill(HIST("h3_centrality_track_pt_track_eta_eventwiseconstituentsubtracted"), collision.centrality(), track.pt(), track.eta());
      registry.fill(HIST("h3_track_pt_track_eta_track_phi_eventwiseconstituentsubtracted"), track.pt(), track.eta(), track.phi());
    }
  }
  PROCESS_SWITCH(JetFinderQATask, processTracksSub, "QA for charged event-wise embedded subtracted tracks", false);

  void processRho(soa::Filtered<soa::Join<JetCollisions, aod::BkgChargedRhos>>::iterator const& collision, soa::Filtered<JetTracks> const& tracks)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelection)) {
      return;
    }
    if (collision.trackOccupancyInTimeRange() < trackOccupancyInTimeRangeMin || trackOccupancyInTimeRangeMax < collision.trackOccupancyInTimeRange()) {
      return;
    }
    int nTracks = 0;
    for (auto const& track : tracks) {
      if (jetderiveddatautilities::selectTrack(track, trackSelection)) {
        nTracks++;
      }
    }
    registry.fill(HIST("h2_centrality_ntracks"), collision.centrality(), nTracks);
    registry.fill(HIST("h2_ntracks_rho"), nTracks, collision.rho());
    registry.fill(HIST("h2_ntracks_rhom"), nTracks, collision.rhoM());
    registry.fill(HIST("h2_centrality_rho"), collision.centrality(), collision.rho());
    registry.fill(HIST("h2_centrality_rhom"), collision.centrality(), collision.rhoM());
  }
  PROCESS_SWITCH(JetFinderQATask, processRho, "QA for rho-area subtracted jets", false);

  void processRandomConeData(soa::Filtered<soa::Join<JetCollisions, aod::BkgChargedRhos>>::iterator const& collision, soa::Join<aod::ChargedJets, aod::ChargedJetConstituents> const& jets, soa::Filtered<JetTracks> const& tracks)
  {
    randomCone(collision, jets, tracks);
  }
  PROCESS_SWITCH(JetFinderQATask, processRandomConeData, "QA for random cone estimation of background fluctuations in data", false);

  void processRandomConeMCD(soa::Filtered<soa::Join<JetCollisions, aod::BkgChargedRhos>>::iterator const& collision, soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents> const& jets, soa::Filtered<JetTracks> const& tracks)
  {
    randomCone(collision, jets, tracks);
  }
  PROCESS_SWITCH(JetFinderQATask, processRandomConeMCD, "QA for random cone estimation of background fluctuations in mcd", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc) { return WorkflowSpec{adaptAnalysisTask<JetFinderQATask>(cfgc, TaskName{"jet-finder-charged-qa"})}; }
