// Pythia.cc - Combined Pion Analysis (Pythia8 + FastJet + ROOT)
// e+ e- -> Z -> qqbar
// Two histograms:
//  1) choose charged pion in each jet closest to original Z-decay quark (fewest ancestry steps)
//  2) choose highest-pT charged pion in each jet
// Fill pT of the combined pion system

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <utility>

#include "TFile.h"
#include "TH1D.h"

// Pythia
#include "Pythia8/Pythia.h"

// FastJet
#include "fastjet/ClusterSequence.hh"

using namespace Pythia8;
using namespace fastjet;

// ---------- Utilities ----------
static inline double wrapToPi(double x) {
  while (x <= -M_PI) x += 2.0 * M_PI;
  while (x >   M_PI) x -= 2.0 * M_PI;
  return x;
}

// ---------- Find the two primary quarks from Z decay ----------
std::pair<int, int> findZdecayQuarks(const Event& event) {
  int quark1 = -1, quark2 = -1;

  for (int i = 0; i < event.size(); ++i) {
    if (event[i].id() != 23) continue; // Z
    int d1 = event[i].daughter1();
    int d2 = event[i].daughter2();

    if (d1 <= 0 || d2 <= 0 || d1 >= event.size() || d2 >= event.size()) continue;

    int pdg1 = std::abs(event[d1].id());
    int pdg2 = std::abs(event[d2].id());

    if (pdg1 >= 1 && pdg1 <= 5 && pdg2 >= 1 && pdg2 <= 5) {
      quark1 = d1;
      quark2 = d2;
      break;
    }
  }
  return {quark1, quark2};
}

struct AncestryResult {
  int steps = 0;
  bool foundQuark = false;
  int quarkIndex = -1;
};

// Walk up mother1 chain until we hit one of the Z quarks (or bail)
AncestryResult countStepsToQuark(const Event& event, int pion_idx, int targetQuark1, int targetQuark2) {
  AncestryResult result;
  int current = pion_idx;
  std::set<int> visited;

  while (current > 0 && visited.find(current) == visited.end()) {
    visited.insert(current);

    int mother = event[current].mother1();
    if (!(mother > 0 && mother < event.size())) break;

    result.steps++;
    current = mother;

    if (current == targetQuark1 || current == targetQuark2) {
      result.foundQuark = true;
      result.quarkIndex = current;
      break;
    }

    if (result.steps > 200) break; // safety
  }
  return result;
}

// ---------- Thrust (grid scan) ----------
double calculateThrust(const std::vector<fastjet::PseudoJet>& particles) {
  if (particles.empty()) return 0.0;

  double totalP = 0.0;
  for (const auto& p : particles) {
    totalP += std::sqrt(p.px()*p.px() + p.py()*p.py() + p.pz()*p.pz());
  }
  if (totalP <= 0.0) return 0.0;

  double maxThrust = 0.0;
  const int nSamples = 100;

  for (int i = 0; i < nSamples; ++i) {
    double theta = M_PI * i / nSamples;
    for (int j = 0; j < nSamples; ++j) {
      double phi = 2.0 * M_PI * j / nSamples;

      double nx = std::sin(theta) * std::cos(phi);
      double ny = std::sin(theta) * std::sin(phi);
      double nz = std::cos(theta);

      double sum = 0.0;
      for (const auto& p : particles) {
        sum += std::abs(p.px()*nx + p.py()*ny + p.pz()*nz);
      }
      double thrust = sum / totalP;
      if (thrust > maxThrust) maxThrust = thrust;
    }
  }
  return maxThrust;
}

struct PionInfo {
  int idx = -1;
  double pT = 0.0;
  int steps = 999999;
};

int main() {
  // Output ROOT file
  TFile* fout = new TFile("week10.root", "RECREATE");

  TH1D* h_closest = new TH1D(
    "h_combined_pT_closestToQuark",
    "p_{T} of combined pion system (closest to quark);p_{T} [GeV];Events",
    100, 0, 50
  );

  TH1D* h_highest = new TH1D(
    "h_combined_pT_highestMomentum",
    "p_{T} of combined pion system (highest momentum);p_{T} [GeV];Events",
    100, 0, 50
  );

  h_closest->SetDirectory(nullptr);
  h_highest->SetDirectory(nullptr);

  // ========== PYTHIA ==========
  Pythia pythia;
  pythia.readString("Beams:idA = -11");
  pythia.readString("Beams:idB = 11");
  pythia.readString("Beams:eCM = 91.2");
  pythia.readString("PDF:lepton = off");
  pythia.readString("HadronLevel:all = on");
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 123456788");

  if (!pythia.init()) {
    std::cerr << "Pythia initialization failed\n";
    return 1;
  }

  // Settings (match Herwig logic)
  const int nEvents = 15000;
  const double R = 0.4;
  const double jetPtMin = 5.0;
  const double thrustCut = 0.8;
  const double backToBackCut = 2.8; // ~160 deg

  int nProcessed = 0, n2Jets = 0, nBackToBack = 0, nWithPions = 0;

  for (int ievt = 0; ievt < nEvents; ++ievt) {
    if (!pythia.next()) continue;
    nProcessed++;

    if ((ievt + 1) % 1000 == 0)
      std::cout << "Processed " << (ievt + 1) << " events...\n";

    // Z->qq
    auto [zQuark1, zQuark2] = findZdecayQuarks(pythia.event);
    if (zQuark1 < 0 || zQuark2 < 0) continue;

    // Visible final-state particles for jets/thrust
    std::vector<int> finals;
    finals.reserve(256);
    for (int i = 0; i < pythia.event.size(); ++i) {
      if (!pythia.event[i].isFinal()) continue;
      if (!pythia.event[i].isVisible()) continue;
      finals.push_back(i);
    }
    if (finals.empty()) continue;

    std::vector<PseudoJet> fjInputs;
    fjInputs.reserve(finals.size());
    for (int idx : finals) {
      PseudoJet pj(pythia.event[idx].px(),
                   pythia.event[idx].py(),
                   pythia.event[idx].pz(),
                   pythia.event[idx].e());
      pj.set_user_index(idx); // store Pythia event index
      fjInputs.push_back(pj);
    }

    // Thrust cut
    double thrust = calculateThrust(fjInputs);
    if (thrust < thrustCut) continue;

    // Jets
    JetDefinition jetDef(antikt_algorithm, R);
    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));
    if (jets.size() != 2) continue;
    n2Jets++;

    PseudoJet jet0 = jets[0], jet1 = jets[1];
    double dphi = wrapToPi(jet0.phi() - jet1.phi());
    if (std::fabs(dphi) < backToBackCut) continue;
    nBackToBack++;

    // Collect charged pions in each jet with ancestry steps
    auto collectPions = [&](const PseudoJet& j) {
      std::vector<PionInfo> out;
      for (const auto& c : j.constituents()) {
        int idx = c.user_index();
        if (idx < 0 || idx >= pythia.event.size()) continue;

        int pdg = pythia.event[idx].id();
        if (std::abs(pdg) != 211) continue; // pi±

        AncestryResult anc = countStepsToQuark(pythia.event, idx, zQuark1, zQuark2);
        if (!anc.foundQuark) continue;

        PionInfo info;
        info.idx = idx;
        info.pT = pythia.event[idx].pT();
        info.steps = anc.steps;
        out.push_back(info);
      }
      return out;
    };

    auto pions0 = collectPions(jet0);
    auto pions1 = collectPions(jet1);
    if (pions0.empty() || pions1.empty()) continue;
    nWithPions++;

    // Closest-to-quark: min steps (tie-breaker: larger pT)
    auto pickClosest = [](std::vector<PionInfo>& v) {
      std::sort(v.begin(), v.end(), [](const PionInfo& a, const PionInfo& b) {
        if (a.steps != b.steps) return a.steps < b.steps;
        return a.pT > b.pT;
      });
      return v.front();
    };

    // Highest momentum: max pT
    auto pickHighest = [](std::vector<PionInfo>& v) {
      std::sort(v.begin(), v.end(), [](const PionInfo& a, const PionInfo& b) {
        return a.pT > b.pT;
      });
      return v.front();
    };

    PionInfo c0 = pickClosest(pions0);
    PionInfo c1 = pickClosest(pions1);

    PionInfo h0 = pickHighest(pions0);
    PionInfo h1 = pickHighest(pions1);

    // Fill combined pT (closest)
    {
      const Particle& p0 = pythia.event[c0.idx];
      const Particle& p1 = pythia.event[c1.idx];
      double px = p0.px() + p1.px();
      double py = p0.py() + p1.py();
      double pT = std::sqrt(px*px + py*py);
      h_closest->Fill(pT);
    }

    // Fill combined pT (highest)
    {
      const Particle& p0 = pythia.event[h0.idx];
      const Particle& p1 = pythia.event[h1.idx];
      double px = p0.px() + p1.px();
      double py = p0.py() + p1.py();
      double pT = std::sqrt(px*px + py*py);
      h_highest->Fill(pT);
    }
  }

  std::cout << "\n========================================\n";
  std::cout << "         EVENT SUMMARY (PYTHIA)\n";
  std::cout << "========================================\n";
  std::cout << "Total events processed:        " << nProcessed << "\n";
  std::cout << "Events with exactly 2 jets:    " << n2Jets << "\n";
  std::cout << "Events back-to-back (>160°):   " << nBackToBack << "\n";
  std::cout << "Events with pions in both jets:" << nWithPions << "\n";
  std::cout << "========================================\n\n";

  fout->cd();
  h_closest->Write();
  h_highest->Write();
  fout->Close();
  delete fout;

  std::cout << "Output written to: week10.root\n";
  pythia.stat();
  return 0;
}
