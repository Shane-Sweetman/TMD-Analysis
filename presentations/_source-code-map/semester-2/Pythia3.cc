// Semester-2/Pythia1.cc

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

#include "TFile.h"
#include "TH1D.h"
#include "TAxis.h"

#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"

using namespace Pythia8;
using namespace fastjet;

// ------------------------- small math helpers -------------------------

struct Vec3 {
  double x=0, y=0, z=0;
  Vec3() = default;
  Vec3(double X,double Y,double Z): x(X),y(Y),z(Z) {}
};

static inline Vec3 operator+(const Vec3& a, const Vec3& b){ return Vec3(a.x+b.x,a.y+b.y,a.z+b.z); }
static inline Vec3 operator-(const Vec3& a, const Vec3& b){ return Vec3(a.x-b.x,a.y-b.y,a.z-b.z); }
static inline Vec3 operator*(double s, const Vec3& a){ return Vec3(s*a.x,s*a.y,s*a.z); }

static inline double dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b){
  return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline double norm(const Vec3& a){ return std::sqrt(dot(a,a)); }
static inline Vec3 unit(const Vec3& a){
  double n = norm(a);
  if (n <= 0.0) return Vec3(0,0,1);
  return (1.0/n) * a;
}

static inline double wrapToPi(double x) {
  while (x <= -M_PI) x += 2.0 * M_PI;
  while (x >   M_PI) x -= 2.0 * M_PI;
  return x;
}

// component transverse to axis n (n must be unit)
static inline Vec3 transverseTo(const Vec3& p, const Vec3& n) {
  return p - dot(p,n)*n;
}

// qT(thrust) = | (p0+p1)_T | wrt thrust axis
static inline double qT_thrust(const Particle& a, const Particle& b, const Vec3& nHat) {
  Vec3 P(a.px()+b.px(), a.py()+b.py(), a.pz()+b.pz());
  Vec3 PT = transverseTo(P, nHat);
  return norm(PT);
}

// phi around thrust axis, defined via a fixed perpendicular basis (e1,e2)
static inline double phiAroundAxis(const Particle& p, const Vec3& nHat, const Vec3& e1, const Vec3& e2) {
  Vec3 pp(p.px(), p.py(), p.pz());
  Vec3 pt = transverseTo(pp, nHat);
  double x = dot(pt, e1);
  double y = dot(pt, e2);
  return std::atan2(y, x);
}

struct ThrustResult {
  double T = 0.0;
  Vec3 nHat{0,0,1};
};

// brute-force scan (kept simple + robust)
static ThrustResult calculateThrustAxis(const std::vector<fastjet::PseudoJet>& particles) {
  ThrustResult out;
  if (particles.empty()) return out;

  double totalP = 0.0;
  for (const auto& p : particles) {
    totalP += std::sqrt(p.px()*p.px() + p.py()*p.py() + p.pz()*p.pz());
  }
  if (totalP <= 0.0) return out;

  // NOTE: increase for accuracy, decrease for speed
  const int nTheta = 50;
  const int nPhi   = 50;

  double bestT = -1.0;
  Vec3 bestN(0,0,1);

  for (int it = 0; it < nTheta; ++it) {
    double theta = M_PI * (it + 0.5) / nTheta;
    double st = std::sin(theta);
    double ct = std::cos(theta);

    for (int ip = 0; ip < nPhi; ++ip) {
      double phi = 2.0 * M_PI * (ip + 0.5) / nPhi;
      double nx = st * std::cos(phi);
      double ny = st * std::sin(phi);
      double nz = ct;
      Vec3 n(nx,ny,nz);

      double sum = 0.0;
      for (const auto& p : particles) {
        sum += std::abs(p.px()*nx + p.py()*ny + p.pz()*nz);
      }
      double T = sum / totalP;
      if (T > bestT) { bestT = T; bestN = n; }
    }
  }

  out.T = bestT;
  out.nHat = unit(bestN);
  return out;
}

// ------------------------- physics helpers -------------------------

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

// ------------------------- interactive table UI -------------------------

static char promptPager() {
  std::cout << "More [Enter=continue, n=next event, q=quit]: " << std::flush;
  std::string line;
  if (!std::getline(std::cin, line)) return 'q';
  if (line == "q" || line == "Q") return 'q';
  if (line == "n" || line == "N") return 'n';
  return 'c'; // continue (next page)
}

static void printEventTableWide(const Pythia8::Event& ev, int ievt, int startRow, int rowsPerPage) {
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

    // keep names minimal like your table screenshots
    std::string pname;
    switch (pid) {
      case 90: pname = "system"; break;
      case 22: pname = "gamma"; break;
      case 23: pname = "Z0"; break;
      case 11: pname = "e-"; break;
      case -11: pname = "e+"; break;
      case 21: pname = "g"; break;
      case 1: pname = "d"; break;
      case -1: pname = "dbar"; break;
      case 2: pname = "u"; break;
      case -2: pname = "ubar"; break;
      case 3: pname = "s"; break;
      case -3: pname = "sbar"; break;
      case 4: pname = "c"; break;
      case -4: pname = "cbar"; break;
      case 5: pname = "b"; break;
      case -5: pname = "bbar"; break;
      case 111: pname = "pi0"; break;
      case 211: pname = "pi+"; break;
      case -211: pname = "pi-"; break;
      case 321: pname = "K+"; break;
      case -321: pname = "K-"; break;
      case 113: pname = "rho0"; break;
      case -213: pname = "rho-"; break;
      case 213: pname = "rho+"; break;
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

// ------------------------- version stamp -------------------------
static const char* BUILD_STAMP =
#if defined(__DATE__) && defined(__TIME__)
  __DATE__ " " __TIME__;
#else
  "unknown";
#endif

// ------------------------- main -------------------------

int main(int argc, char* argv[]) {
  bool INTERACTIVE_MODE = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--interactive" || arg == "-i") INTERACTIVE_MODE = true;
    if (arg == "--version") {
      std::cout << "BUILD_STAMP: " << BUILD_STAMP << "\n";
      return 0;
    }
  }

  // ROOT output
  TFile* fout = new TFile("pythia1.root", "RECREATE");

  // Zoom settings for qT(thrust)
  const int    qTBins = 200;
  const double qTMax  = 10.0;

  TH1D* h_qT_thrust_closest_OS = new TH1D(
    "h_qT_thrust_closest_OS",
    "q_{T}^{thrust} (closest), OS;q_{T}^{thrust} [GeV];Events",
    qTBins, 0, qTMax
  );

  TH1D* h_qT_thrust_closest_SS = new TH1D(
    "h_qT_thrust_closest_SS",
    "q_{T}^{thrust} (closest), SS;q_{T}^{thrust} [GeV];Events",
    qTBins, 0, qTMax
  );

  TH1D* h_qT_thrust_highest_OS = new TH1D(
    "h_qT_thrust_highest_OS",
    "q_{T}^{thrust} (highest), OS;q_{T}^{thrust} [GeV];Events",
    qTBins, 0, qTMax
  );

  TH1D* h_qT_thrust_highest_SS = new TH1D(
    "h_qT_thrust_highest_SS",
    "q_{T}^{thrust} (highest), SS;q_{T}^{thrust} [GeV];Events",
    qTBins, 0, qTMax
  );

  h_qT_thrust_closest_OS->SetDirectory(nullptr);
  h_qT_thrust_closest_SS->SetDirectory(nullptr);
  h_qT_thrust_highest_OS->SetDirectory(nullptr);
  h_qT_thrust_highest_SS->SetDirectory(nullptr);

  // Pythia setup (e+e- @ Z pole)
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

  // Analysis settings
  const int nEvents = 20000;

  const double R = 0.4;
  const double jetPtMin = 5.0;

  const double thrustCut = 0.8;
  const double backToBackCut = 2.8; // ~160 deg

  int nProcessed = 0, n2Jets = 0, nBackToBack = 0, nWithAnyPions = 0;
  int nFillClosestOS = 0, nFillClosestSS = 0, nFillHighestOS = 0, nFillHighestSS = 0;

  // explicit overflow diagnostics (don’t hide tails)
  long long nOver10_closest_OS = 0, nOver10_closest_SS = 0;
  long long nOver10_highest_OS = 0, nOver10_highest_SS = 0;

  long long nOver20_closest_OS = 0, nOver20_closest_SS = 0;
  long long nOver20_highest_OS = 0, nOver20_highest_SS = 0;

  bool keepInteractive = true;

  for (int ievt = 0; ievt < nEvents; ++ievt) {
    if (!pythia.next()) continue;
    nProcessed++;

    if (!INTERACTIVE_MODE && (ievt + 1) % 1000 == 0)
      std::cout << "Processed " << (ievt + 1) << " events...\n";

    // Z -> qqbar indices
    std::pair<int,int> zq = findZdecayQuarks(pythia.event);
    int zQuark1 = zq.first;
    int zQuark2 = zq.second;
    if (zQuark1 < 0 || zQuark2 < 0) continue;

    // final visible particles for thrust + jets
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

    // thrust axis + thrust cut
    ThrustResult thr = calculateThrustAxis(fjInputs);
    if (thr.T < thrustCut) continue;

    // define (e1,e2) basis around thrust axis (for phiT)
    Vec3 ref(0,0,1);
    if (std::fabs(dot(ref, thr.nHat)) > 0.95) ref = Vec3(0,1,0);
    Vec3 e1 = unit(cross(ref, thr.nHat));
    Vec3 e2 = cross(thr.nHat, e1);

    // jets
    JetDefinition jetDef(antikt_algorithm, R);
    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));
    if ((int)jets.size() != 2) continue;
    n2Jets++;

    PseudoJet jet0 = jets[0], jet1 = jets[1];
    double dphi_jets = wrapToPi(jet0.phi_std() - jet1.phi_std());
    if (std::fabs(dphi_jets) < backToBackCut) continue;
    nBackToBack++;

    // collect pions inside a given jet, with ancestry to Z quarks
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

    // best pion of a given charge in a jet
    auto bestByCharge = [&](const std::vector<PionInfo>& v, int charge, bool useClosest) -> std::optional<PionInfo> {
      std::optional<PionInfo> best;
      for (const auto& pi : v) {
        if (pi.charge != charge) continue;
        if (!best) best = pi;
        else {
          if (useClosest) { if (betterClosest(pi, *best)) best = pi; }
          else           { if (betterHighest(pi, *best)) best = pi; }
        }
      }
      return best;
    };

    // pick one pion from a jet (either + or -) according to strategy
    auto pickLead = [&](const std::optional<PionInfo>& pos,
                        const std::optional<PionInfo>& neg,
                        bool useClosest) -> std::optional<PionInfo> {
      if (pos && neg) {
        if (useClosest) return betterClosest(*pos, *neg) ? pos : neg;
        return betterHighest(*pos, *neg) ? pos : neg;
      }
      return pos ? pos : neg;
    };

    // ---- closest strategy ----
    auto c0_pos = bestByCharge(pions0, +1, true);
    auto c0_neg = bestByCharge(pions0, -1, true);
    auto c1_pos = bestByCharge(pions1, +1, true);
    auto c1_neg = bestByCharge(pions1, -1, true);

    auto c0_lead = pickLead(c0_pos, c0_neg, true);
    auto c1_lead = pickLead(c1_pos, c1_neg, true);

    // ---- highest-pT strategy ----
    auto h0_pos = bestByCharge(pions0, +1, false);
    auto h0_neg = bestByCharge(pions0, -1, false);
    auto h1_pos = bestByCharge(pions1, +1, false);
    auto h1_neg = bestByCharge(pions1, -1, false);

    auto h0_lead = pickLead(h0_pos, h0_neg, false);
    auto h1_lead = pickLead(h1_pos, h1_neg, false);

    // store per-event strings for interactive print (only if needed)
    std::string lineClosest, lineHighest;
    bool hasAnyPairThisEvent = false;

    if (c0_lead && c1_lead) {
      const Particle& p0 = pythia.event[c0_lead->idx];
      const Particle& p1 = pythia.event[c1_lead->idx];

      double qT = qT_thrust(p0, p1, thr.nHat);
      double phiT0 = phiAroundAxis(p0, thr.nHat, e1, e2);
      double phiT1 = phiAroundAxis(p1, thr.nHat, e1, e2);
      double dphiT = wrapToPi(phiT0 - phiT1);

      bool isOS = (c0_lead->charge != c1_lead->charge);

      if (isOS) {
        if (qT > qTMax) nOver10_closest_OS++;
        if (qT > 20.0)  nOver20_closest_OS++;
        h_qT_thrust_closest_OS->Fill(qT);
        nFillClosestOS++;
      } else {
        if (qT > qTMax) nOver10_closest_SS++;
        if (qT > 20.0)  nOver20_closest_SS++;
        h_qT_thrust_closest_SS->Fill(qT);
        nFillClosestSS++;
      }

      {
        std::ostringstream os;
        os << "[closest] " << (isOS ? "OS" : "SS")
           << "  qT(thrust)=" << std::fixed << std::setprecision(3) << qT
           << "  dphiT=" << std::fixed << std::setprecision(3) << dphiT
           << "  pion0(idx=" << c0_lead->idx << ",q=" << c0_lead->charge << ",steps=" << c0_lead->steps << ")"
           << "  pion1(idx=" << c1_lead->idx << ",q=" << c1_lead->charge << ",steps=" << c1_lead->steps << ")";
        lineClosest = os.str();
      }
      hasAnyPairThisEvent = true;
    }

    if (h0_lead && h1_lead) {
      const Particle& p0 = pythia.event[h0_lead->idx];
      const Particle& p1 = pythia.event[h1_lead->idx];

      double qT = qT_thrust(p0, p1, thr.nHat);
      double phiT0 = phiAroundAxis(p0, thr.nHat, e1, e2);
      double phiT1 = phiAroundAxis(p1, thr.nHat, e1, e2);
      double dphiT = wrapToPi(phiT0 - phiT1);

      bool isOS = (h0_lead->charge != h1_lead->charge);

      if (isOS) {
        if (qT > qTMax) nOver10_highest_OS++;
        if (qT > 20.0)  nOver20_highest_OS++;
        h_qT_thrust_highest_OS->Fill(qT);
        nFillHighestOS++;
      } else {
        if (qT > qTMax) nOver10_highest_SS++;
        if (qT > 20.0)  nOver20_highest_SS++;
        h_qT_thrust_highest_SS->Fill(qT);
        nFillHighestSS++;
      }

      {
        std::ostringstream os;
        os << "[highest] " << (isOS ? "OS" : "SS")
           << "  qT(thrust)=" << std::fixed << std::setprecision(3) << qT
           << "  dphiT=" << std::fixed << std::setprecision(3) << dphiT
           << "  pion0(idx=" << h0_lead->idx << ",q=" << h0_lead->charge << ",steps=" << h0_lead->steps << ")"
           << "  pion1(idx=" << h1_lead->idx << ",q=" << h1_lead->charge << ",steps=" << h1_lead->steps << ")";
        lineHighest = os.str();
      }
      hasAnyPairThisEvent = true;
    }

    // interactive: print info + ALWAYS show the table (paged)
    if (INTERACTIVE_MODE && keepInteractive && hasAnyPairThisEvent) {
      std::cout << "\n=== EVENT " << ievt << " ===\n";
      std::cout << "thrust=" << std::fixed << std::setprecision(4) << thr.T
                << "  thrust_axis=(" << thr.nHat.x << "," << thr.nHat.y << "," << thr.nHat.z << ")"
                << "  dphi(j0,j1)=" << std::fixed << std::setprecision(4) << dphi_jets
                << "\n";

      // keep jet phis in [0,2pi) for nicer reading (matches your screenshot vibe)
      std::cout << "jet0: pt=" << jet0.pt() << " eta=" << jet0.eta() << " phi=" << jet0.phi_02pi() << "\n";
      std::cout << "jet1: pt=" << jet1.pt() << " eta=" << jet1.eta() << " phi=" << jet1.phi_02pi() << "\n";

      if (!lineClosest.empty()) std::cout << lineClosest << "\n";
      if (!lineHighest.empty()) std::cout << lineHighest << "\n";
      std::cout << "src(note: indices refer to Pythia event record)\n";

      int startRow = 0;
      const int rowsPerPage = 25;

      while (true) {
        printEventTableWide(pythia.event, ievt, startRow, rowsPerPage);
        char c = promptPager();
        if (c == 'q') { keepInteractive = false; break; }
        if (c == 'n') break;
        startRow += rowsPerPage;
        if (startRow >= pythia.event.size()) break;
      }
    }
  }

  // summary
  std::cout << "\n========================================\n";
  std::cout << "         EVENT SUMMARY (PYTHIA)\n";
  std::cout << "========================================\n";
  std::cout << "Total events processed:             " << nProcessed << "\n";
  std::cout << "Events with exactly 2 jets:         " << n2Jets << "\n";
  std::cout << "Events back-to-back (>160 deg):     " << nBackToBack << "\n";
  std::cout << "Events with any pions in both jets: " << nWithAnyPions << "\n";
  std::cout << "---- Filled histograms (qT^{thrust}) ----\n";
  std::cout << "Closest  OS filled:                 " << nFillClosestOS << "\n";
  std::cout << "Closest  SS filled:                 " << nFillClosestSS << "\n";
  std::cout << "Highest  OS filled:                 " << nFillHighestOS << "\n";
  std::cout << "Highest  SS filled:                 " << nFillHighestSS << "\n";
  std::cout << "---- Overflow diagnostics (NOT hiding tails) ----\n";
  std::cout << "Histogram x-range is [0," << qTMax << "] GeV\n";
  std::cout << "qT > " << qTMax << "  (closest OS/SS):  " << nOver10_closest_OS << " / " << nOver10_closest_SS << "\n";
  std::cout << "qT > " << qTMax << "  (highest OS/SS):  " << nOver10_highest_OS << " / " << nOver10_highest_SS << "\n";
  std::cout << "qT > 20 (closest OS/SS):            " << nOver20_closest_OS << " / " << nOver20_closest_SS << "\n";
  std::cout << "qT > 20 (highest OS/SS):            " << nOver20_highest_OS << " / " << nOver20_highest_SS << "\n";
  std::cout << "========================================\n\n";

  // write output
  fout->cd();
  h_qT_thrust_closest_OS->Write();
  h_qT_thrust_closest_SS->Write();
  h_qT_thrust_highest_OS->Write();
  h_qT_thrust_highest_SS->Write();
  fout->Close();
  delete fout;

  std::cout << "Output written to: pythia1.root\n";
  pythia.stat();
  return 0;
}
