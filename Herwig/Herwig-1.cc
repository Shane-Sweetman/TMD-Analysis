#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <utility>
#include <optional>
#include <memory>
#include <fstream>
#include <string>

#include "TFile.h"
#include "TH1D.h"
// src(ROOT I/O + histograms): https://root.cern/doc/master/classTFile.html  ;  https://root.cern/doc/master/classTH1.html
// src(ROOT ownership / SetDirectory): https://root.cern/manual/object_ownership/

#include "fastjet/ClusterSequence.hh"
// src(FastJet clustering + constituents): https://fastjet.fr/repo/fastjet-doc-3.4.0.pdf
// src(FastJet user_index mapping): https://fastjet.fr/repo/doxygen-3.4.0/classfastjet_1_1PseudoJet.html

#include "HepMC3/Reader.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/ReaderAsciiHepMC2.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"
#include "HepMC3/FourVector.h"
// src(HepMC3 overview + event record graph concepts): https://arxiv.org/pdf/1912.08005
// src(HepMC3 repo + examples incl. HepMC2 reader example): https://gitlab.cern.ch/hepmc/HepMC3

using namespace fastjet;

static inline double wrapToPi(double x) {
  while (x <= -M_PI) x += 2.0 * M_PI;
  while (x >   M_PI) x -= 2.0 * M_PI;
  return x;
}

static inline double ptFromPxPy(double px, double py) {
  return std::sqrt(px * px + py * py);
}

static inline double combinedPt(const HepMC3::FourVector& a, const HepMC3::FourVector& b) {
  return ptFromPxPy(a.px() + b.px(), a.py() + b.py());
}

static bool isNeutrino(int pidAbs) {
  return (pidAbs == 12 || pidAbs == 14 || pidAbs == 16);
}

static bool isVisibleFinal(const HepMC3::ConstGenParticlePtr& p) {
  if (!p) return false;
  if (p->status() != 1) return false;
  int pidAbs = std::abs(p->pid());
  if (isNeutrino(pidAbs)) return false;
  return true;
}

double calculateThrust(const std::vector<PseudoJet>& particles) {
  if (particles.empty()) return 0.0;

  double totalP = 0.0;
  for (const auto& p : particles)
    totalP += std::sqrt(p.px()*p.px() + p.py()*p.py() + p.pz()*p.pz());
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
      for (const auto& p : particles)
        sum += std::abs(p.px()*nx + p.py()*ny + p.pz()*nz);

      double thrust = sum / totalP;
      if (thrust > maxThrust) maxThrust = thrust;
    }
  }
  return maxThrust;
}
// src(Thrust definition / event-shape context): https://pythia.org/latest-manual/EventAnalysis.html

std::unique_ptr<HepMC3::Reader> makeReader(const std::string& fname) {
  std::ifstream fin(fname);
  std::string line;
  bool looksAsciiV3 = false;
  for (int i = 0; i < 20 && std::getline(fin, line); ++i) {
    if (line.find("Asciiv3") != std::string::npos) { looksAsciiV3 = true; break; }
  }
  if (looksAsciiV3) return std::make_unique<HepMC3::ReaderAscii>(fname);
  return std::make_unique<HepMC3::ReaderAsciiHepMC2>(fname);
}
// src(HepMC3 I/O readers): https://gitlab.cern.ch/hepmc/HepMC3

std::pair<int, int> findZdecayQuarks(const HepMC3::GenEvent& evt) {
  int q1 = -1, q2 = -1;

  for (const auto& p : evt.particles()) {
    if (!p) continue;
    if (p->pid() != 23) continue;

    auto v = p->end_vertex();
    if (!v) continue;

    for (const auto& d : v->particles_out()) {
      if (!d) continue;
      int apdg = std::abs(d->pid());
      if (apdg >= 1 && apdg <= 5) {
        if (q1 < 0) q1 = d->id();
        else if (q2 < 0) { q2 = d->id(); break; }
      }
    }
    if (q1 >= 0 && q2 >= 0) break;
  }
  return {q1, q2};
}
// src(HepMC3 parent/child via end_vertex + particles_out): https://arxiv.org/pdf/1912.08005

struct AncestryResult {
  int steps = 0;
  bool foundQuark = false;
  int quarkId = -1;
};

AncestryResult countStepsToQuark(const HepMC3::ConstGenParticlePtr& pion,
                                int targetQuark1Id, int targetQuark2Id) {
  AncestryResult result;
  auto current = pion;
  std::set<int> visited;

  while (current) {
    int cid = current->id();
    if (visited.find(cid) != visited.end()) break;
    visited.insert(cid);

    auto v = current->production_vertex();
    if (!v) break;
    if (v->particles_in().empty()) break;

    auto mother = v->particles_in().front();
    if (!mother) break;

    result.steps++;
    current = mother;

    int mid = current->id();
    if (mid == targetQuark1Id || mid == targetQuark2Id) {
      result.foundQuark = true;
      result.quarkId = mid;
      break;
    }

    if (result.steps > 200) break;
  }
  return result;
}
// src(HepMC3 ancestry via production_vertex + particles_in): https://gitlab.cern.ch/hepmc/HepMC3

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

int main(int argc, char* argv[]) {
  std::string infile = "lep91_hepmc.hepmc";
  if (argc >= 2) infile = argv[1];

  auto reader = makeReader(infile);

  TFile* fout = new TFile("herwig1.root", "RECREATE");

  TH1D* h_closest_OS = new TH1D(
    "h_combined_pT_closestToQuark_OS",
    "q_{T} of opposite charge pion pair (closest to quark);q_{T} [GeV];Events",
    100, 0, 50
  );

  TH1D* h_closest_SS = new TH1D(
    "h_combined_pT_closestToQuark_SS",
    "q_{T} of same sign charge pion pair (closest to quark);q_{T} [GeV];Events",
    100, 0, 50
  );

  TH1D* h_highest_OS = new TH1D(
    "h_combined_pT_highestPt_OS",
    "q_{T} of opposite charge pion pair (highest momentum);q_{T} [GeV];Events",
    100, 0, 50
  );

  TH1D* h_highest_SS = new TH1D(
    "h_combined_pT_highestPt_SS",
    "q_{T} of same sign charge pion pair (highest momentum);q_{T} [GeV];Events",
    100, 0, 50
  );

  h_closest_OS->SetDirectory(nullptr);
  h_closest_SS->SetDirectory(nullptr);
  h_highest_OS->SetDirectory(nullptr);
  h_highest_SS->SetDirectory(nullptr);

  const int nEventsTarget = 15000;
  const double R = 0.4;
  const double jetPtMin = 5.0;
  const double thrustCut = 0.8;
  const double backToBackCut = 2.8;

  int nRead = 0, n2Jets = 0, nBackToBack = 0, nWithAnyPions = 0;
  int nFillClosestOS = 0, nFillClosestSS = 0, nFillHighestOS = 0, nFillHighestSS = 0;

  while (true) {
    HepMC3::GenEvent evt;
    reader->read_event(evt);
    if (reader->failed()) break;

    nRead++;
    if (nRead % 1000 == 0) std::cout << "Read " << nRead << " events...\n";
    if (nRead > nEventsTarget) break;

    auto [zQuark1Id, zQuark2Id] = findZdecayQuarks(evt);
    if (zQuark1Id < 0 || zQuark2Id < 0) continue;

    std::vector<PseudoJet> fjInputs;
    std::vector<HepMC3::ConstGenParticlePtr> parts;
    std::vector<int> pids;
    fjInputs.reserve(256);
    parts.reserve(256);
    pids.reserve(256);

    for (const auto& p : evt.particles()) {
      if (!isVisibleFinal(p)) continue;

      const auto& m = p->momentum();
      PseudoJet pj(m.px(), m.py(), m.pz(), m.e());
      int idx = (int)parts.size();
      pj.set_user_index(idx);

      fjInputs.push_back(pj);
      parts.push_back(p);
      pids.push_back(p->pid());
    }

    if (fjInputs.empty()) continue;

    double thrust = calculateThrust(fjInputs);
    if (thrust < thrustCut) continue;

    JetDefinition jetDef(antikt_algorithm, R);
    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));
    if (jets.size() != 2) continue;
    n2Jets++;

    double dphi = wrapToPi(jets[0].phi() - jets[1].phi());
    if (std::fabs(dphi) < backToBackCut) continue;
    nBackToBack++;

    auto collectPions = [&](const PseudoJet& j) {
      std::vector<PionInfo> out;
      for (const auto& c : j.constituents()) {
        int idx = c.user_index();
        if (idx < 0 || idx >= (int)parts.size()) continue;

        int pid = pids[idx];
        if (std::abs(pid) != 211) continue;

        auto pptr = parts[idx];
        AncestryResult anc = countStepsToQuark(pptr, zQuark1Id, zQuark2Id);
        if (!anc.foundQuark) continue;

        const auto& m = pptr->momentum();

        PionInfo info;
        info.idx = idx;
        info.pT = ptFromPxPy(m.px(), m.py());
        info.steps = anc.steps;
        info.charge = (pid > 0 ? +1 : -1);
        out.push_back(info);
      }
      return out;
    };

    auto pions0 = collectPions(jets[0]);
    auto pions1 = collectPions(jets[1]);
    if (pions0.empty() || pions1.empty()) continue;
    nWithAnyPions++;

    auto bestByCharge = [&](const std::vector<PionInfo>& v, int charge, bool useClosest) -> std::optional<PionInfo> {
      std::optional<PionInfo> best;
      for (const auto& pi : v) {
        if (pi.charge != charge) continue;
        if (!best.has_value()) best = pi;
        else {
          if (useClosest) { if (betterClosest(pi, *best)) best = pi; }
          else           { if (betterHighest(pi, *best)) best = pi; }
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
      const auto& p0 = parts[closestOS.a.idx]->momentum();
      const auto& p1 = parts[closestOS.b.idx]->momentum();
      h_closest_OS->Fill(combinedPt(p0, p1));
      nFillClosestOS++;
    }
    if (closestSS.valid) {
      const auto& p0 = parts[closestSS.a.idx]->momentum();
      const auto& p1 = parts[closestSS.b.idx]->momentum();
      h_closest_SS->Fill(combinedPt(p0, p1));
      nFillClosestSS++;
    }

    PairChoice highestOS, highestSS;

    {
      PairChoice opt1, opt2;
      if (h0_pos && h1_neg) { opt1.valid = true; opt1.a = *h0_pos; opt1.b = *h1_neg; opt1.sumPt = opt1.a.pT + opt1.b.pT; }
      if (h0_neg && h1_pos) { opt2.valid = true; opt2.a = *h0_neg; opt2.b = *h1_pos; opt2.sumPt = opt2.a.pT + opt2.b.pT; }
      highestOS = pairBetterHighest(opt1, opt2) ? opt1 : opt2;
    }

    {
      PairChoice opt1, opt2;
      if (h0_pos && h1_pos) { opt1.valid = true; opt1.a = *h0_pos; opt1.b = *h1_pos; opt1.sumPt = opt1.a.pT + opt1.b.pT; }
      if (h0_neg && h1_neg) { opt2.valid = true; opt2.a = *h0_neg; opt2.b = *h1_neg; opt2.sumPt = opt2.a.pT + opt2.b.pT; }
      highestSS = pairBetterHighest(opt1, opt2) ? opt1 : opt2;
    }

    if (highestOS.valid) {
      const auto& p0 = parts[highestOS.a.idx]->momentum();
      const auto& p1 = parts[highestOS.b.idx]->momentum();
      h_highest_OS->Fill(combinedPt(p0, p1));
      nFillHighestOS++;
    }
    if (highestSS.valid) {
      const auto& p0 = parts[highestSS.a.idx]->momentum();
      const auto& p1 = parts[highestSS.b.idx]->momentum();
      h_highest_SS->Fill(combinedPt(p0, p1));
      nFillHighestSS++;
    }
  }

  std::cout << "\n========================================\n";
  std::cout << "         EVENT SUMMARY (HERWIG/HepMC)\n";
  std::cout << "========================================\n";
  std::cout << "Total events read:                " << nRead << "\n";
  std::cout << "Events with exactly 2 jets:       " << n2Jets << "\n";
  std::cout << "Events back-to-back (>160 deg):   " << nBackToBack << "\n";
  std::cout << "Events with any pions in both jets:" << nWithAnyPions << "\n";
  std::cout << "---- Filled histograms ----\n";
  std::cout << "Closest OS filled:                " << nFillClosestOS << "\n";
  std::cout << "Closest SS filled:                " << nFillClosestSS << "\n";
  std::cout << "Highest-pT OS filled:             " << nFillHighestOS << "\n";
  std::cout << "Highest-pT SS filled:             " << nFillHighestSS << "\n";
  std::cout << "========================================\n\n";

  fout->cd();
  h_closest_OS->Write();
  h_closest_SS->Write();
  h_highest_OS->Write();
  h_highest_SS->Write();
  fout->Close();
  delete fout;

  std::cout << "Output written to: herwig1.root\n";
  return 0;
}
