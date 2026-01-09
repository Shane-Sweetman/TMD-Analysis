// HerwigPtHighest.cc
// Read HepMC (from Herwig), cluster jets with FastJet, pick highest-pT charged pion in each jet,
// fill combined pT histogram into a ROOT file.
//
// Compile (example — we’ll run this next step):
// g++ -std=c++17 \
//   -I$HOME/hep/Herwig7/include \
//   $(root-config --cflags) \
//   /Users/shanesweetman/Downloads/fastjet/bin/fastjet-config --cxxflags \
//   HerwigPtHighest.cc -o herwig_pt_highest \
//   -L$HOME/hep/Herwig7/lib -lHepMC3 \
//   $(root-config --libs) \
//   /Users/shanesweetman/Downloads/fastjet/bin/fastjet-config --libs \
//   -Wl,-rpath,$HOME/hep/Herwig7/lib -Wl,-rpath,/Users/shanesweetman/Downloads/fastjet/lib

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>

#include "TFile.h"
#include "TH1D.h"

#include "fastjet/ClusterSequence.hh"

// HepMC3 (can read HepMC2 ASCII too)
#include "HepMC3/Reader.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/ReaderAsciiHepMC2.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/FourVector.h"

using namespace fastjet;

static inline double wrapToPi(double x) {
  while (x <= -M_PI) x += 2.0 * M_PI;
  while (x >   M_PI) x -= 2.0 * M_PI;
  return x;
}

static bool isNeutrino(int pidAbs) {
  return (pidAbs == 12 || pidAbs == 14 || pidAbs == 16);
}

// very simple “visible” proxy: drop neutrinos only
static bool isVisibleFinal(const HepMC3::ConstGenParticlePtr& p) {
  if (!p) return false;
  if (p->status() != 1) return false; // final state
  int pidAbs = std::abs(p->pid());
  if (isNeutrino(pidAbs)) return false;
  return true;
}

// crude thrust (same sampling approach you used)
double calculateThrust(const std::vector<PseudoJet> &particles) {
  if (particles.empty()) return 0.0;

  double totalP = 0.0;
  for (const auto &p : particles)
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
      for (const auto &p : particles) {
        double dot = std::abs(p.px()*nx + p.py()*ny + p.pz()*nz);
        sum += dot;
      }
      double thrust = sum / totalP;
      if (thrust > maxThrust) maxThrust = thrust;
    }
  }
  return maxThrust;
}

// Decide reader based on whether file looks like Asciiv3
std::unique_ptr<HepMC3::Reader> makeReader(const std::string& fname) {
  std::ifstream fin(fname);
  std::string line;
  bool looksAsciiV3 = false;
  for (int i = 0; i < 20 && std::getline(fin, line); ++i) {
    if (line.find("Asciiv3") != std::string::npos) {
      looksAsciiV3 = true;
      break;
    }
  }
  if (looksAsciiV3) return std::make_unique<HepMC3::ReaderAscii>(fname);
  return std::make_unique<HepMC3::ReaderAsciiHepMC2>(fname);
}

int main(int argc, char* argv[]) {
  std::string infile = "lep91_hepmc.hepmc"; // change if your big file has a different name
  if (argc >= 2) infile = argv[1];

  auto reader = makeReader(infile);

  TFile *fout = new TFile("herwig_pt_highest.root", "RECREATE");
  TH1D *h_combined_pT_highestMomentum = new TH1D(
    "h_combined_pT_highestMomentum",
    "Herwig: p_{T} of combined pion system (highest momentum);p_{T} [GeV];Events",
    100, 0, 50
  );
  h_combined_pT_highestMomentum->SetDirectory(nullptr);

  const int nEventsTarget = 20000;
  const double R = 0.4;
  const double jetPtMin = 5.0;

  int nRead = 0, n2Jets = 0, nBackToBack = 0, nWithPions = 0;

  while (true) {
    HepMC3::GenEvent evt;
    reader->read_event(evt);
    if (reader->failed()) break;

    nRead++;
    if (nRead % 1000 == 0) std::cout << "Read " << nRead << " events...\n";
    if (nRead > nEventsTarget) break;

    // build FastJet inputs
    std::vector<PseudoJet> fjInputs;
    fjInputs.reserve(256);

    int uidx = 0;
    for (const auto& p : evt.particles()) {
      if (!isVisibleFinal(p)) continue;

      const auto& m = p->momentum();
      PseudoJet pj(m.px(), m.py(), m.pz(), m.e());
      pj.set_user_index(uidx);

      // stash pid in a parallel array-style trick: store in user_info-like way via a map later
      // For now, we’ll just store pointers in a vector aligned with user_index.
      fjInputs.push_back(pj);
      uidx++;
    }

    if (fjInputs.empty()) continue;

    double thrust = calculateThrust(fjInputs);
    if (thrust < 0.8) continue;

    JetDefinition jetDef(antikt_algorithm, R);
    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));

    if (jets.size() != 2) continue;
    n2Jets++;

    double dphi = wrapToPi(jets[0].phi() - jets[1].phi());
    if (std::fabs(dphi) < 2.8) continue;
    nBackToBack++;

    // We need to find highest-pT charged pion in each jet.
    // Since we didn’t keep a pid array above, redo a simple particle list aligned with fjInputs order.
    // (Kept simple on purpose for first pass.)
    std::vector<int> pids;
    pids.reserve(fjInputs.size());
    for (const auto& p : evt.particles()) {
      if (!isVisibleFinal(p)) continue;
      pids.push_back(p->pid());
    }

    auto highestPionInJet = [&](const PseudoJet& j, PseudoJet& outPion) -> bool {
      bool found = false;
      double bestPt = -1.0;

      for (const auto& c : j.constituents()) {
        int idx = c.user_index();
        if (idx < 0 || idx >= (int)pids.size()) continue;

        int pid = pids[idx];
        if (std::abs(pid) != 211) continue; // pi+/- only

        double pt = c.perp();
        if (pt > bestPt) {
          bestPt = pt;
          outPion = c;
          found = true;
        }
      }
      return found;
    };

    PseudoJet pion0, pion1;
    if (!highestPionInJet(jets[0], pion0)) continue;
    if (!highestPionInJet(jets[1], pion1)) continue;
    nWithPions++;

    double px_sum = pion0.px() + pion1.px();
    double py_sum = pion0.py() + pion1.py();
    double pT_combined = std::sqrt(px_sum*px_sum + py_sum*py_sum);

    h_combined_pT_highestMomentum->Fill(pT_combined);
  }

  std::cout << "\n=== SUMMARY (Herwig HepMC read) ===\n";
  std::cout << "Events read:                 " << nRead << "\n";
  std::cout << "Events with exactly 2 jets:  " << n2Jets << "\n";
  std::cout << "Back-to-back (>160 deg):     " << nBackToBack << "\n";
  std::cout << "With pi in both jets:        " << nWithPions << "\n";

  fout->cd();
  h_combined_pT_highestMomentum->Write();
  fout->Close();
  delete fout;

  std::cout << "Wrote: herwig_pt_highest.root\n";
  return 0;
}
