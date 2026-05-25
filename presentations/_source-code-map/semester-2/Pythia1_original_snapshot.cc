#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <utility>
#include <optional>

#include "TFile.h"
#include "TH1D.h"
// src(ROOT I/O + histograms): https://root.cern/doc/master/classTFile.html  ;  https://root.cern/doc/master/classTH1.html
// src(ROOT ownership / SetDirectory): https://root.cern/manual/object_ownership/

#include "Pythia8/Pythia.h"
// src(Pythia basic skeleton: readString/init/next/stat): https://github.com/mortenpi/pythia8/blob/master/examples/main01.cc
// src(Pythia e+e- @ LEP1 / Z-pole style settings): https://pythia.org/latest-manual/examples/main103.html
// src(Pythia event record API: mother/daughter, particle accessors): https://pythia.org/latest-manual/EventRecord.html

#include "fastjet/ClusterSequence.hh"
// src(FastJet clustering + constituents): https://fastjet.fr/repo/fastjet-doc-3.4.0.pdf
// src(FastJet user_index mapping): https://fastjet.fr/repo/doxygen-3.4.0/classfastjet_1_1PseudoJet.html

using namespace Pythia8;
using namespace fastjet;

static inline double wrapToPi(double x) {
  while (x <= -M_PI) x += 2.0 * M_PI;
  while (x >   M_PI) x -= 2.0 * M_PI;
  return x;
}

static inline double combinedPt(const Particle& a, const Particle& b) {
  double px = a.px() + b.px();
  double py = a.py() + b.py();
  return std::sqrt(px * px + py * py);
}

std::pair<int, int> findZdecayQuarks(const Event& event) {
  int quark1 = -1, quark2 = -1;

  for (int i = 0; i < event.size(); ++i) {
    if (event[i].id() != 23) continue;
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
// src(Pythia genealogy access: id/daughter1/daughter2): https://pythia.org/latest-manual/EventRecord.html

struct AncestryResult {
  int steps = 0;
  bool foundQuark = false;
  int quarkIndex = -1;
};

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

    if (result.steps > 200) break;
  }
  return result;
}
// src(Pythia ancestry walk via mother1): https://pythia.org/latest-manual/EventRecord.html

double calculateThrust(const std::vector<fastjet::PseudoJet>& particles) {
  if (particles.empty()) return 0.0;

  double totalP = 0.0;
  for (const auto& p : particles) {
    totalP += std::sqrt(p.px() * p.px() + p.py() * p.py() + p.pz() * p.pz());
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
        sum += std::abs(p.px() * nx + p.py() * ny + p.pz() * nz);
      }
      double thrust = sum / totalP;
      if (thrust > maxThrust) maxThrust = thrust;
    }
  }
  return maxThrust;
}
// src(Thrust definition / event-shape context): https://pythia.org/latest-manual/EventAnalysis.html

struct PionInfo {
  int idx = -1;
  double pT = 0.0;
  int steps = 999999;
  int charge = 0;
};

static inline bool betterClosest(const PionInfo& a, const PionInfo& b) {
  if (a.steps != b.steps) return a.steps < b.steps;
  return a.pT > b.pT;
}

static inline bool betterHighest(const PionInfo& a, const PionInfo& b) {
  return a.pT > b.pT;
}

struct PairChoice {
  PionInfo a;
  PionInfo b;
  bool valid = false;
  int sumSteps = 0;
  double sumPt = 0.0;
};

static inline bool pairBetterClosest(const PairChoice& x, const PairChoice& y) {
  if (!y.valid) return x.valid;
  if (!x.valid) return false;
  if (x.sumSteps != y.sumSteps) return x.sumSteps < y.sumSteps;
  return x.sumPt > y.sumPt;
}

static inline bool pairBetterHighest(const PairChoice& x, const PairChoice& y) {
  if (!y.valid) return x.valid;
  if (!x.valid) return false;
  return x.sumPt > y.sumPt;
}

int main() {
  TFile* fout = new TFile("pythia1.root", "RECREATE");
  // src AIM: https://root.cern/doc/master/classTFile.html  ;  src ownership note: https://root.cern/manual/object_ownership/

  TH1D* h_closest_OS = new TH1D(
    "h_combined_pT_closestToQuark_OS",
    "q_{T} of opposite charge pion pair (closest to quark);q_{T} [GeV];Events",
    100, 0, 50
  );

  TH1D* h_closest_SS = new TH1D(
    "h_combined_pT_closestToQuark_SS",
    "q_{t} of same sign charge pion pair (closest to quark);q_{t} [GeV];Events",
    100, 0, 50
  );

  TH1D* h_highest_OS = new TH1D(
    "h_combined_pT_highestPt_OS",
    "q_{T} of opposite charge pion pair (highest momentum);q_{T} [GeV];Events",
    100, 0, 50
  );

  TH1D* h_highest_SS = new TH1D(
    "h_combined_pT_highestPt_SS",
    "q_{t} of same sign charge pion pair (highest momentum);q_{t} [GeV];Events",
    100, 0, 50
  );

  h_closest_OS->SetDirectory(nullptr);
  h_closest_SS->SetDirectory(nullptr);
  h_highest_OS->SetDirectory(nullptr);
  h_highest_SS->SetDirectory(nullptr);
  // src(SetDirectory / ownership): https://root.cern/manual/object_ownership/

  Pythia pythia;
  pythia.readString("Beams:idA = -11");
  pythia.readString("Beams:idB = 11");
  pythia.readString("Beams:eCM = 91.2");
  pythia.readString("PDF:lepton = off");
  pythia.readString("HadronLevel:all = on");
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 123456788");
  // src(Pythia ee @ LEP1 config pattern): https://pythia.org/latest-manual/examples/main103.html

  if (!pythia.init()) {
    std::cerr << "Pythia initialization failed\n";
    return 1;
  }

  const int nEvents = 15000;
  const double R = 0.4;
  const double jetPtMin = 5.0;
  const double thrustCut = 0.8;
  const double backToBackCut = 2.8;

  int nProcessed = 0, n2Jets = 0, nBackToBack = 0;
  int nWithAnyPions = 0;

  int nFillClosestOS = 0, nFillClosestSS = 0;
  int nFillHighestOS = 0, nFillHighestSS = 0;

  for (int ievt = 0; ievt < nEvents; ++ievt) {
    if (!pythia.next()) continue;
    nProcessed++;
    // src(event loop idiom next()/continue + stat() later): https://github.com/mortenpi/pythia8/blob/master/examples/main01.cc

    if ((ievt + 1) % 1000 == 0)
      std::cout << "Processed " << (ievt + 1) << " events...\n";

    auto [zQuark1, zQuark2] = findZdecayQuarks(pythia.event);
    if (zQuark1 < 0 || zQuark2 < 0) continue;

    std::vector<int> finals;
    finals.reserve(256);
    for (int i = 0; i < pythia.event.size(); ++i) {
      if (!pythia.event[i].isFinal()) continue;
      if (!pythia.event[i].isVisible()) continue;
      finals.push_back(i);
    }
    if (finals.empty()) continue;
    // src(isFinal/isVisible + accessors): https://pythia.org/latest-manual/ParticleProperties.html

    std::vector<PseudoJet> fjInputs;
    fjInputs.reserve(finals.size());
    for (int idx : finals) {
      PseudoJet pj(pythia.event[idx].px(),
                   pythia.event[idx].py(),
                   pythia.event[idx].pz(),
                   pythia.event[idx].e());
      pj.set_user_index(idx);
      fjInputs.push_back(pj);
    }
    // src(map constituents back to inputs): https://fastjet.fr/repo/doxygen-3.4.0/classfastjet_1_1PseudoJet.html

    double thrust = calculateThrust(fjInputs);
    if (thrust < thrustCut) continue;

    JetDefinition jetDef(antikt_algorithm, R);
    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));
    // src(ClusterSequence / inclusive_jets / sorted_by_pt): https://fastjet.fr/repo/fastjet-doc-3.4.0.pdf

    if (jets.size() != 2) continue;
    n2Jets++;

    PseudoJet jet0 = jets[0], jet1 = jets[1];
    double dphi = wrapToPi(jet0.phi() - jet1.phi());
    if (std::fabs(dphi) < backToBackCut) continue;
    nBackToBack++;

    auto collectPions = [&](const PseudoJet& j) {
      std::vector<PionInfo> out;
      for (const auto& c : j.constituents()) {
        int idx = c.user_index();
        if (idx < 0 || idx >= pythia.event.size()) continue;

        int pdg = pythia.event[idx].id();
        if (std::abs(pdg) != 211) continue;

        AncestryResult anc = countStepsToQuark(pythia.event, idx, zQuark1, zQuark2);
        if (!anc.foundQuark) continue;

        PionInfo info;
        info.idx = idx;
        info.pT = pythia.event[idx].pT();
        info.steps = anc.steps;
        info.charge = (pdg > 0 ? +1 : -1);
        out.push_back(info);
      }
      return out;
    };
    // src(constituents + user_index; EventRecord id/mother1; Particle pT): https://pythia.org/latest-manual/EventRecord.html

    auto pions0 = collectPions(jet0);
    auto pions1 = collectPions(jet1);
    if (pions0.empty() || pions1.empty()) continue;
    nWithAnyPions++;

    auto bestByCharge = [&](const std::vector<PionInfo>& v, int charge, bool useClosest) -> std::optional<PionInfo> {
      std::optional<PionInfo> best;
      for (const auto& pi : v) {
        if (pi.charge != charge) continue;
        if (!best.has_value()) {
          best = pi;
        } else {
          if (useClosest) {
            if (betterClosest(pi, *best)) best = pi;
          } else {
            if (betterHighest(pi, *best)) best = pi;
          }
        }
      }
      return best;
    };

    auto c0_pos = bestByCharge(pions0, +1, true);
    auto c0_neg = bestByCharge(pions0, -1, true);
    auto c1_pos = bestByCharge(pions1, +1, true);
    auto c1_neg = bestByCharge(pions1, -1, true);

    auto h0_pos = bestByCharge(pions0, +1, false);
    auto h0_neg = bestByCharge(pions0, -1, false);
    auto h1_pos = bestByCharge(pions1, +1, false);
    auto h1_neg = bestByCharge(pions1, -1, false);

    PairChoice closestOS, closestSS;

    {
      PairChoice opt1, opt2;
      if (c0_pos && c1_neg) {
        opt1.valid = true; opt1.a = *c0_pos; opt1.b = *c1_neg;
        opt1.sumSteps = opt1.a.steps + opt1.b.steps;
        opt1.sumPt = opt1.a.pT + opt1.b.pT;
      }
      if (c0_neg && c1_pos) {
        opt2.valid = true; opt2.a = *c0_neg; opt2.b = *c1_pos;
        opt2.sumSteps = opt2.a.steps + opt2.b.steps;
        opt2.sumPt = opt2.a.pT + opt2.b.pT;
      }
      closestOS = pairBetterClosest(opt1, opt2) ? opt1 : opt2;
    }

    {
      PairChoice opt1, opt2;
      if (c0_pos && c1_pos) {
        opt1.valid = true; opt1.a = *c0_pos; opt1.b = *c1_pos;
        opt1.sumSteps = opt1.a.steps + opt1.b.steps;
        opt1.sumPt = opt1.a.pT + opt1.b.pT;
      }
      if (c0_neg && c1_neg) {
        opt2.valid = true; opt2.a = *c0_neg; opt2.b = *c1_neg;
        opt2.sumSteps = opt2.a.steps + opt2.b.steps;
        opt2.sumPt = opt2.a.pT + opt2.b.pT;
      }
      closestSS = pairBetterClosest(opt1, opt2) ? opt1 : opt2;
    }

    if (closestOS.valid) {
      const Particle& p0 = pythia.event[closestOS.a.idx];
      const Particle& p1 = pythia.event[closestOS.b.idx];
      h_closest_OS->Fill(combinedPt(p0, p1));
      nFillClosestOS++;
    }
    if (closestSS.valid) {
      const Particle& p0 = pythia.event[closestSS.a.idx];
      const Particle& p1 = pythia.event[closestSS.b.idx];
      h_closest_SS->Fill(combinedPt(p0, p1));
      nFillClosestSS++;
    }

    PairChoice highestOS, highestSS;

    {
      PairChoice opt1, opt2;
      if (h0_pos && h1_neg) {
        opt1.valid = true; opt1.a = *h0_pos; opt1.b = *h1_neg;
        opt1.sumPt = opt1.a.pT + opt1.b.pT;
      }
      if (h0_neg && h1_pos) {
        opt2.valid = true; opt2.a = *h0_neg; opt2.b = *h1_pos;
        opt2.sumPt = opt2.a.pT + opt2.b.pT;
      }
      highestOS = pairBetterHighest(opt1, opt2) ? opt1 : opt2;
    }

    {
      PairChoice opt1, opt2;
      if (h0_pos && h1_pos) {
        opt1.valid = true; opt1.a = *h0_pos; opt1.b = *h1_pos;
        opt1.sumPt = opt1.a.pT + opt1.b.pT;
      }
      if (h0_neg && h1_neg) {
        opt2.valid = true; opt2.a = *h0_neg; opt2.b = *h1_neg;
        opt2.sumPt = opt2.a.pT + opt2.b.pT;
      }
      highestSS = pairBetterHighest(opt1, opt2) ? opt1 : opt2;
    }

    if (highestOS.valid) {
      const Particle& p0 = pythia.event[highestOS.a.idx];
      const Particle& p1 = pythia.event[highestOS.b.idx];
      h_highest_OS->Fill(combinedPt(p0, p1));
      nFillHighestOS++;
    }
    if (highestSS.valid) {
      const Particle& p0 = pythia.event[highestSS.a.idx];
      const Particle& p1 = pythia.event[highestSS.b.idx];
      h_highest_SS->Fill(combinedPt(p0, p1));
      nFillHighestSS++;
    }
  }

  std::cout << "\n========================================\n";
  std::cout << "         EVENT SUMMARY (PYTHIA)\n";
  std::cout << "========================================\n";
  std::cout << "Total events processed:            " << nProcessed << "\n";
  std::cout << "Events with exactly 2 jets:        " << n2Jets << "\n";
  std::cout << "Events back-to-back (>160 deg):    " << nBackToBack << "\n";
  std::cout << "Events with any pions in both jets:" << nWithAnyPions << "\n";
  std::cout << "---- Filled histograms ----\n";
  std::cout << "Closest OS filled:                 " << nFillClosestOS << "\n";
  std::cout << "Closest SS filled:                 " << nFillClosestSS << "\n";
  std::cout << "Highest-pT OS filled:              " << nFillHighestOS << "\n";
  std::cout << "Highest-pT SS filled:              " << nFillHighestSS << "\n";
  std::cout << "========================================\n\n";

  fout->cd();
  h_closest_OS->Write();
  h_closest_SS->Write();
  h_highest_OS->Write();
  h_highest_SS->Write();
  fout->Close();
  delete fout;

  std::cout << "Output written to: pythia1.root\n";
  pythia.stat();
  return 0;
}
