#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <utility>
#include <optional>
#include <iomanip>
#include <sstream>
#include <string>
// src(iomanip formatting): https://en.cppreference.com/w/cpp/header/iomanip
// src(ostringstream): https://en.cppreference.com/w/cpp/io/basic_ostringstream
// src(std::string): https://en.cppreference.com/w/cpp/header/string

#include "TFile.h"
#include "TH1D.h"
#include "TAxis.h"
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

static inline double phiFromPxPy(double px, double py) {
  return std::atan2(py, px);
}

static inline double etaFromP(double px, double py, double pz) {
  double p = std::sqrt(px*px + py*py + pz*pz);
  if (p <= 0.0) return 0.0;
  double c = pz / p;
  if (c >  1.0) c =  1.0;
  if (c < -1.0) c = -1.0;
  double theta = std::acos(c);
  double t = std::tan(theta / 2.0);
  if (t <= 0.0) return (pz >= 0.0 ? 1e9 : -1e9);
  return -std::log(t);
}
// src(eta definition via polar angle): standard; avoids relying on Particle::eta()

// --- count histogram entries in a window around a target (e.g. 20 GeV) ---
// ROOT note: TH1::FindBin is not const in ROOT 6.36, so use the axis FindBin instead.
static inline long long countInWindow(const TH1D* h, double center, double halfWidth) {
  if (!h) return 0;

  const double x1 = center - halfWidth;
  const double x2 = center + halfWidth;

  const double eps = 1e-9; // avoid bin-edge ambiguity

  const TAxis* ax = h->GetXaxis();
  if (!ax) return 0;

  int b1 = ax->FindBin(x1 + eps);
  int b2 = ax->FindBin(x2 - eps);

  // clamp to valid bins (ignore under/overflow)
  if (b1 < 1) b1 = 1;
  if (b2 > h->GetNbinsX()) b2 = h->GetNbinsX();
  if (b2 < b1) return 0;

  double sum = h->Integral(b1, b2);
  return (long long) std::llround(sum); // for unweighted fills, integer-valued
}
// --------------------------------------------------------------------------------

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

// Simple pager (terminal interactive)
char promptPager(const char* msg = "More") {
  std::cout << msg << " [Enter=continue, n=next event, q=quit]: " << std::flush;
  std::string line;
  if (!std::getline(std::cin, line)) return 'q';
  if (line == "q" || line == "Q") return 'q';
  if (line == "n" || line == "N") return 'n';
  return 'c';
}

// Simple focus prompt (terminal interactive)
char promptFocus(const char* msg = "Focus") {
  std::cout << msg << " [Enter=continue, a=ancestry, t=table, q=quit]: " << std::flush;
  std::string line;
  if (!std::getline(std::cin, line)) return 'q';
  if (line == "q" || line == "Q") return 'q';
  if (line == "a" || line == "A") return 'a';
  if (line == "t" || line == "T") return 't';
  return 'c';
}

static void printEventTableWide(const Pythia8::Event& ev, int ievt,
                                int startRow, int rowsPerPage) {
  using std::cout; using std::left; using std::right; using std::setw; using std::setprecision; using std::fixed;

  const int N = ev.size();
  const int endRow = std::min(N - 1, startRow + rowsPerPage - 1);

  const int W_ROW   = 6;
  const int W_EVT   = 7;
  const int W_SIZE  = 6;
  const int W_NO    = 6;
  const int W_ID    = 8;
  const int W_NAME  = 14;
  const int W_ST    = 6;
  const int W_M1    = 6;
  const int W_M2    = 6;
  const int W_D1    = 6;
  const int W_D2    = 6;
  const int W_P     = 13;

  auto sep = [&](){
    cout
      << std::string(W_ROW,   '-') << "+"
      << std::string(W_EVT,   '-') << "+"
      << std::string(W_SIZE,  '-') << "+"
      << std::string(W_NO,    '-') << "+"
      << std::string(W_ID,    '-') << "+"
      << std::string(W_NAME,  '-') << "+"
      << std::string(W_ST,    '-') << "+"
      << std::string(W_M1+1+W_M2, '-') << "+"
      << std::string(W_D1+1+W_D2, '-') << "+"
      << std::string(W_P,     '-') << "+"
      << std::string(W_P,     '-') << "+"
      << std::string(W_P,     '-') << "+"
      << std::string(W_P,     '-') << "\n";
  };

  cout << "\n=== EVENT " << ievt << " (table) ===\n\n";
  cout << left
       << setw(W_ROW)  << "row"  << "|"
       << setw(W_EVT)  << "event"<< "|"
       << setw(W_SIZE) << "size" << "|"
       << setw(W_NO)   << "no"   << "|"
       << setw(W_ID)   << "id"   << "|"
       << setw(W_NAME) << "name" << "|"
       << setw(W_ST)   << "st"   << "|"
       << setw(W_M1)   << "m1"   << " " << setw(W_M2) << "m2" << "|"
       << setw(W_D1)   << "d1"   << " " << setw(W_D2) << "d2" << "|"
       << setw(W_P)    << "px"   << "|"
       << setw(W_P)    << "py"   << "|"
       << setw(W_P)    << "pz"   << "|"
       << setw(W_P)    << "E"    << "\n";
  sep();

  cout << fixed << setprecision(3);

  for (int i = startRow; i <= endRow; ++i) {
    const auto& p = ev[i];
    const int pid = p.id();
    std::string pname;
    switch (pid) {
      case 90: pname = "system"; break;
      case 22: pname = "gamma"; break;
      case 23: pname = "Z0"; break;
      case 11: pname = "e-"; break;
      case -11: pname = "e+"; break;
      case 13: pname = "mu-"; break;
      case -13: pname = "mu+"; break;
      case 14: pname = "nu_mu"; break;
      case -14: pname = "nu_mubar"; break;
      case 21: pname = "g"; break;
      case 1: pname = "d"; break;
      case -1: pname = "dbar"; break;
      case 111: pname = "pi0"; break;
      case 211: pname = "pi+"; break;
      case -211: pname = "pi-"; break;
      case 321: pname = "K+"; break;
      case -321: pname = "K-"; break;
      case 113: pname = "rho0"; break;
      case -213: pname = "rho-"; break;
      case 213: pname = "rho+"; break;
      case 2212: pname = "p+"; break;
      case -2212: pname = "pbar"; break;
      default: {
        std::ostringstream os;
        os << "id" << pid;
        pname = os.str();
      }
    }

    int d1 = p.daughter1();
    int d2 = p.daughter2();

    cout << left
         << setw(W_ROW)  << i         << "|"
         << setw(W_EVT)  << ievt      << "|"
         << setw(W_SIZE) << N         << "|"
         << setw(W_NO)   << p.index() << "|"
         << setw(W_ID)   << pid       << "|"
         << setw(W_NAME) << pname     << "|"
         << setw(W_ST)   << p.status()<< "|"
         << setw(W_M1)   << p.mother1()<< " " << setw(W_M2) << p.mother2() << "|"
         << setw(W_D1)   << d1        << " " << setw(W_D2) << d2           << "|"
         << right
         << setw(W_P)    << p.px()    << "|"
         << setw(W_P)    << p.py()    << "|"
         << setw(W_P)    << p.pz()    << "|"
         << setw(W_P)    << p.e()     << "\n";
  }

  std::cout << "(showing rows " << startRow << " to " << endRow << " of " << N << " total)\n";
}

// print only one event-record row (useful for selected pions)
static void printOneRow(const Pythia8::Event& ev, int i) {
  if (i < 0 || i >= ev.size()) return;
  const auto& p = ev[i];
  std::cout
    << "idx=" << i
    << " id=" << p.id()
    << " st=" << p.status()
    << " m1=" << p.mother1() << " m2=" << p.mother2()
    << " d1=" << p.daughter1() << " d2=" << p.daughter2()
    << " px=" << p.px()
    << " py=" << p.py()
    << " pz=" << p.pz()
    << " E="  << p.e()
    << "\n";
}

// Print a short mother-chain (optional), starting from idx, stopping at Z-decay quark if found.
static void printMotherChain(const Pythia8::Event& ev, int startIdx,
                             int zQuark1, int zQuark2,
                             int maxSteps = 25) {
  std::set<int> visited;
  int cur = startIdx;

  std::cout << "mother-chain (start idx=" << startIdx << "):\n";

  for (int s = 0; s < maxSteps; ++s) {
    if (cur <= 0 || cur >= ev.size()) {
      std::cout << "  (stop: idx out of range)\n";
      break;
    }
    if (visited.find(cur) != visited.end()) {
      std::cout << "  (stop: loop detected at idx=" << cur << ")\n";
      break;
    }
    visited.insert(cur);

    std::cout << "  step " << s << ": ";
    printOneRow(ev, cur);

    if (cur == zQuark1 || cur == zQuark2) {
      std::cout << "  (hit Z-decay quark idx=" << cur << ")\n";
      break;
    }

    int m = ev[cur].mother1();
    if (m <= 0 || m >= ev.size()) {
      std::cout << "  (stop: mother1 invalid m1=" << m << ")\n";
      break;
    }
    cur = m;
  }
}

int main(int argc, char* argv[]) {
  bool INTERACTIVE_MODE = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--interactive" || arg == "-i") {
      INTERACTIVE_MODE = true;
      std::cout << "\n*** INTERACTIVE MODE ENABLED ***\n";
      std::cout << "qT~20 probe events will pause in terminal.\n\n";
    }
  }

  TFile* fout = new TFile("pythia1.root", "RECREATE");

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

  const int nEvents = 15000;
  const double R = 0.4;
  const double jetPtMin = 5.0;
  const double thrustCut = 0.8;
  const double backToBackCut = 2.8;

  const double probeQt = 20.0;
  const double probeWindow = 1.0;
  const int maxProbe = 10;

  int nProbeClosestOS = 0, nProbeClosestSS = 0;
  int nProbeHighestOS = 0, nProbeHighestSS = 0;

  int nProcessed = 0, n2Jets = 0, nBackToBack = 0;
  int nWithAnyPions = 0;

  int nFillClosestOS = 0, nFillClosestSS = 0;
  int nFillHighestOS = 0, nFillHighestSS = 0;

  bool keepInteractive = true;

  for (int ievt = 0; ievt < nEvents; ++ievt) {
    if (!pythia.next()) continue;
    nProcessed++;

    if ((ievt + 1) % 1000 == 0)
      std::cout << "Processed " << (ievt + 1) << " events...\n";

    std::pair<int,int> zq = findZdecayQuarks(pythia.event);
    int zQuark1 = zq.first;
    int zQuark2 = zq.second;
    if (zQuark1 < 0 || zQuark2 < 0) continue;

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
      pj.set_user_index(idx);
      fjInputs.push_back(pj);
    }

    double thrust = calculateThrust(fjInputs);
    if (thrust < thrustCut) continue;

    JetDefinition jetDef(antikt_algorithm, R);
    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));

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

    auto pickLead = [&](const std::optional<PionInfo>& pos,
                        const std::optional<PionInfo>& neg,
                        bool useClosest) -> std::optional<PionInfo> {
      if (pos && neg) {
        if (useClosest) return betterClosest(*pos, *neg) ? pos : neg;
        return betterHighest(*pos, *neg) ? pos : neg;
      }
      return pos ? pos : neg;
    };

    auto maybeLogQt20Interactive = [&](const char* tag,
                                      bool isOS,
                                      double qT,
                                      const PionInfo& a,
                                      const PionInfo& b,
                                      const Particle& p0,
                                      const Particle& p1,
                                      const PseudoJet& j0,
                                      const PseudoJet& j1,
                                      int& counter) {
      if (std::fabs(qT - probeQt) > probeWindow) return;
      if (counter >= maxProbe) return;

      counter++;

      double phi0 = phiFromPxPy(p0.px(), p0.py());
      double phi1 = phiFromPxPy(p1.px(), p1.py());
      double dphiPi = wrapToPi(phi0 - phi1);

      double eta0 = etaFromP(p0.px(), p0.py(), p0.pz());
      double eta1 = etaFromP(p1.px(), p1.py(), p1.pz());

      double dphi_p0_j0 = wrapToPi(phi0 - j0.phi());
      double dphi_p1_j1 = wrapToPi(phi1 - j1.phi());

      std::cout << "\n=== QT PROBE (" << tag << ", " << (isOS ? "OS" : "SS") << ") ===\n";
      std::cout << "ievt=" << ievt
                << "  qT=" << std::fixed << std::setprecision(3) << qT
                << "  thrust=" << thrust
                << "  dphi(j0,j1)=" << dphi
                << "  count=" << counter << "/" << maxProbe << "\n";

      std::cout << "jet0: pt=" << j0.pt() << " eta=" << j0.eta() << " phi=" << j0.phi() << "\n";
      std::cout << "jet1: pt=" << j1.pt() << " eta=" << j1.eta() << " phi=" << j1.phi() << "\n";

      std::cout << "pion0: idx=" << a.idx
                << "  q=" << a.charge
                << "  steps=" << a.steps
                << "  pT=" << p0.pT()
                << "  eta=" << eta0
                << "  phi=" << phi0
                << "  (dphi to jet0=" << dphi_p0_j0 << ")\n";
      std::cout << "       px=" << p0.px() << " py=" << p0.py() << " pz=" << p0.pz() << " E=" << p0.e() << "\n";

      std::cout << "pion1: idx=" << b.idx
                << "  q=" << b.charge
                << "  steps=" << b.steps
                << "  pT=" << p1.pT()
                << "  eta=" << eta1
                << "  phi=" << phi1
                << "  (dphi to jet1=" << dphi_p1_j1 << ")\n";
      std::cout << "       px=" << p1.px() << " py=" << p1.py() << " pz=" << p1.pz() << " E=" << p1.e() << "\n";

      std::cout << "pair: dphi(pi0,pi1)=" << dphiPi << "  qT=" << qT << "\n";
      std::cout << "src(note: indices refer to Pythia event record)\n";

      if (!INTERACTIVE_MODE || !keepInteractive) return;

      std::cout << "\n--- FOCUSED VIEW: selected pion pair only ---\n";
      std::cout << "pair pion0 row:\n";
      printOneRow(pythia.event, a.idx);
      std::cout << "pair pion1 row:\n";
      printOneRow(pythia.event, b.idx);

      while (true) {
        char f = promptFocus("Focus");
        if (f == 'q') { keepInteractive = false; return; }
        if (f == 'c') { return; }
        if (f == 'a') {
          std::cout << "\n--- ANCESTRY (mother1 chain) ---\n";
          std::cout << "pion0 ancestry:\n";
          printMotherChain(pythia.event, a.idx, zQuark1, zQuark2, 25);
          std::cout << "pion1 ancestry:\n";
          printMotherChain(pythia.event, b.idx, zQuark1, zQuark2, 25);
          continue;
        }
        if (f == 't') {
          int startRow = 0;
          const int rowsPerPage = 25;

          while (true) {
            printEventTableWide(pythia.event, ievt, startRow, rowsPerPage);
            char c = promptPager("More");
            if (c == 'q') { keepInteractive = false; break; }
            if (c == 'n') break;
            startRow += rowsPerPage;
            if (startRow >= pythia.event.size()) break;
          }
          return;
        }
      }
    };

    auto c0_pos = bestByCharge(pions0, +1, true);
    auto c0_neg = bestByCharge(pions0, -1, true);
    auto c1_pos = bestByCharge(pions1, +1, true);
    auto c1_neg = bestByCharge(pions1, -1, true);

    auto h0_pos = bestByCharge(pions0, +1, false);
    auto h0_neg = bestByCharge(pions0, -1, false);
    auto h1_pos = bestByCharge(pions1, +1, false);
    auto h1_neg = bestByCharge(pions1, -1, false);

    auto c0_lead = pickLead(c0_pos, c0_neg, true);
    auto c1_lead = pickLead(c1_pos, c1_neg, true);

    if (c0_lead && c1_lead) {
      const Particle& p0 = pythia.event[c0_lead->idx];
      const Particle& p1 = pythia.event[c1_lead->idx];

      double qT = combinedPt(p0, p1);
      bool isOS = (c0_lead->charge != c1_lead->charge);

      if (isOS) {
        h_closest_OS->Fill(qT);
        nFillClosestOS++;
        maybeLogQt20Interactive("closest", true, qT, *c0_lead, *c1_lead, p0, p1, jet0, jet1, nProbeClosestOS);
      } else {
        h_closest_SS->Fill(qT);
        nFillClosestSS++;
        maybeLogQt20Interactive("closest", false, qT, *c0_lead, *c1_lead, p0, p1, jet0, jet1, nProbeClosestSS);
      }
    }

    auto h0_lead = pickLead(h0_pos, h0_neg, false);
    auto h1_lead = pickLead(h1_pos, h1_neg, false);

    if (h0_lead && h1_lead) {
      const Particle& p0 = pythia.event[h0_lead->idx];
      const Particle& p1 = pythia.event[h1_lead->idx];

      double qT = combinedPt(p0, p1);
      bool isOS = (h0_lead->charge != h1_lead->charge);

      if (isOS) {
        h_highest_OS->Fill(qT);
        nFillHighestOS++;
        maybeLogQt20Interactive("highest", true, qT, *h0_lead, *h1_lead, p0, p1, jet0, jet1, nProbeHighestOS);
      } else {
        h_highest_SS->Fill(qT);
        nFillHighestSS++;
        maybeLogQt20Interactive("highest", false, qT, *h0_lead, *h1_lead, p0, p1, jet0, jet1, nProbeHighestSS);
      }
    }
  }

  std::cout << "\n========================================\n";
  std::cout << "         EVENT SUMMARY (PYTHIA)\n";
  std::cout << "========================================\n";
  std::cout << "Total events processed:             " << nProcessed << "\n";
  std::cout << "Events with exactly 2 jets:         " << n2Jets << "\n";
  std::cout << "Events back-to-back (>160 deg):     " << nBackToBack << "\n";
  std::cout << "Events with any pions in both jets: " << nWithAnyPions << "\n";
  std::cout << "---- Filled histograms ----\n";
  std::cout << "Closest OS filled:                  " << nFillClosestOS << "\n";
  std::cout << "Closest SS filled:                  " << nFillClosestSS << "\n";
  std::cout << "Highest-pT OS filled:               " << nFillHighestOS << "\n";
  std::cout << "Highest-pT SS filled:               " << nFillHighestSS << "\n";
  std::cout << "---- qT~20 probe logs (printed, capped by maxProbe) ----\n";
  std::cout << "Probe closest OS printed:           " << nProbeClosestOS << "\n";
  std::cout << "Probe closest SS printed:           " << nProbeClosestSS << "\n";
  std::cout << "Probe highest OS printed:           " << nProbeHighestOS << "\n";
  std::cout << "Probe highest SS printed:           " << nProbeHighestSS << "\n";

  std::cout << "---- Counts in qT window around 20 GeV (from histograms) ----\n";
  std::cout << "Window: [" << (probeQt - probeWindow) << ", " << (probeQt + probeWindow) << "] GeV\n";

  long long n20_closest_OS = countInWindow(h_closest_OS, probeQt, probeWindow);
  long long n20_closest_SS = countInWindow(h_closest_SS, probeQt, probeWindow);
  long long n20_highest_OS = countInWindow(h_highest_OS, probeQt, probeWindow);
  long long n20_highest_SS = countInWindow(h_highest_SS, probeQt, probeWindow);

  std::cout << "Closest OS in window:               " << n20_closest_OS << "\n";
  std::cout << "Closest SS in window:               " << n20_closest_SS << "\n";
  std::cout << "Highest-pT OS in window:            " << n20_highest_OS << "\n";
  std::cout << "Highest-pT SS in window:            " << n20_highest_SS << "\n";

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
