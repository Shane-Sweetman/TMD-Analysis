// scan_square60_heatmap.cc
// One parameter-point scan for PYTHIA vs fixed z=0.70 theory at the 60% cut.
// Keeps the same event selection logic as the current analysis:
// - e+e- at 91.2 GeV
// - thrust axis from brute-force 50x50 scan
// - anti-kT R=0.4 jets
// - exactly 2 jets with pT > 5 GeV
// - back-to-back delta-phi > 2.8
// - charged pions traced back to Z->q qbar daughters
// - pair choice = maximum qT pair
// - 60% cut applied via min(frac1, frac2) >= 0.6
//
// Usage:
//   ./scan_square60_heatmap NEVENTS SEED SIGMA ALPHAS THEORYFILE OUTTXT
//
// Output line columns written to OUTTXT:
// sigma alphaS nEvents seed scale chi2OS NOS chi2PerOS chi2SS NSS chi2PerSS score

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <limits>

// ROOT
#include "TH1D.h"

// Pythia
#include "Pythia8/Pythia.h"

// FastJet
#include "fastjet/ClusterSequence.hh"

using namespace Pythia8;
using namespace fastjet;

struct TheoryData {
  std::vector<double> qT;
  std::vector<double> os;
  std::vector<double> ss;
};

struct HadronCand {
  Vec4 p4;
  int charge;
  double frac;
};

struct ThrustInfo {
  Vec4 axis;
  double T = 0.0;
};

struct Chi2Result {
  double chi2 = 0.0;
  int nUsed = 0;
};

static inline double norm3(const Vec4& a) {
  return std::sqrt(dot3(a, a));
}

static inline Vec4 unit3(const Vec4& a) {
  double n = norm3(a);
  if (n <= 0.0) return Vec4(0,0,1,0);
  return Vec4(a.px()/n, a.py()/n, a.pz()/n, 0);
}

static ThrustInfo computeThrustInfo(const std::vector<Vec4>& ps) {
  ThrustInfo out;
  if (ps.empty()) {
    out.axis = Vec4(0,0,1,0);
    out.T = 0.0;
    return out;
  }

  double totalP = 0.0;
  for (const auto& p : ps) totalP += norm3(p);
  if (totalP <= 0.0) {
    out.axis = Vec4(0,0,1,0);
    out.T = 0.0;
    return out;
  }

  const int nTheta = 50;
  const int nPhi   = 50;

  double bestT = -1.0;
  Vec4 bestN(0,0,1,0);

  for (int it = 0; it < nTheta; ++it) {
    double theta = M_PI * (it + 0.5) / nTheta;
    double st = std::sin(theta);
    double ct = std::cos(theta);

    for (int ip = 0; ip < nPhi; ++ip) {
      double phi = 2.0 * M_PI * (ip + 0.5) / nPhi;
      Vec4 n(st * std::cos(phi), st * std::sin(phi), ct, 0.0);

      double sum = 0.0;
      for (const auto& p : ps) sum += std::fabs(dot3(p, n));
      double T = sum / totalP;

      if (T > bestT) {
        bestT = T;
        bestN = n;
      }
    }
  }

  out.axis = unit3(bestN);
  out.T = bestT;
  return out;
}

static double qT_pair(const Vec4& p1, const Vec4& p2, const Vec4& n_unit) {
  Vec4 q = p1 + p2;
  double qpar = q.px()*n_unit.px() + q.py()*n_unit.py() + q.pz()*n_unit.pz();
  Vec4 qperp(q.px() - qpar*n_unit.px(),
             q.py() - qpar*n_unit.py(),
             q.pz() - qpar*n_unit.pz(), 0);
  return norm3(qperp);
}

static inline double wrapToPi(double x) {
  while (x <= -M_PI) x += 2.0 * M_PI;
  while (x >   M_PI) x -= 2.0 * M_PI;
  return x;
}

bool loadTheoryData(const std::string& fname, TheoryData& th) {
  std::ifstream fin(fname);
  if (!fin.is_open()) {
    std::cerr << "Could not open theory file: " << fname << "\n";
    return false;
  }

  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') continue;

    std::istringstream iss(line);
    double q, os, ss;
    if (iss >> q >> os >> ss) {
      th.qT.push_back(q);
      th.os.push_back(os);
      th.ss.push_back(ss);
    }
  }

  if (th.qT.empty()) {
    std::cerr << "No theory points found in: " << fname << "\n";
    return false;
  }
  return true;
}

double theoryEval(const TheoryData& th, double x, bool useOS) {
  const std::vector<double>& y = useOS ? th.os : th.ss;

  if (th.qT.empty()) return 0.0;
  if (x <= th.qT.front()) return y.front();
  if (x >= th.qT.back())  return y.back();

  for (size_t i = 0; i + 1 < th.qT.size(); ++i) {
    if (x >= th.qT[i] && x < th.qT[i + 1]) {
      double t = (x - th.qT[i]) / (th.qT[i + 1] - th.qT[i]);
      return y[i] + t * (y[i + 1] - y[i]);
    }
  }
  return 0.0;
}

double theoryPeakOS(const TheoryData& th) {
  if (th.os.empty()) return 0.0;
  return *std::max_element(th.os.begin(), th.os.end());
}

double pythiaPeakOS(const TH1D* h, double xMax) {
  double out = 0.0;
  for (int b = 1; b <= h->GetNbinsX(); ++b) {
    double x = h->GetBinCenter(b);
    if (x > xMax) continue;
    out = std::max(out, h->GetBinContent(b));
  }
  return out;
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

struct AncestryResult {
  int steps = 0;
  bool foundQuark = false;
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
      break;
    }
    if (result.steps > 200) break;
  }
  return result;
}

static Chi2Result chi2AgainstTheory(const TH1D* hData,
                                    const TheoryData& th,
                                    double scale,
                                    bool useOS,
                                    double xMin,
                                    double xMax) {
  Chi2Result out;

  for (int b = 1; b <= hData->GetNbinsX(); ++b) {
    double x = hData->GetBinCenter(b);
    if (x < xMin || x > xMax) continue;

    double d = hData->GetBinContent(b);
    double e = hData->GetBinError(b);
    double t = scale * theoryEval(th, x, useOS);

    if (e <= 0.0) continue;
    if (t <= 0.0) continue;

    out.chi2 += std::pow((d - t) / e, 2);
    out.nUsed++;
  }

  return out;
}

int main(int argc, char* argv[]) {
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0]
              << " NEVENTS SEED SIGMA ALPHAS THEORYFILE OUTTXT\n";
    return 1;
  }

  const long long nEvents = std::atoll(argv[1]);
  const int seed = std::atoi(argv[2]);
  const double sigma = std::atof(argv[3]);
  const double alphaS = std::atof(argv[4]);
  const std::string theoryFile = argv[5];
  const std::string outTxt = argv[6];

  TheoryData th;
  if (!loadTheoryData(theoryFile, th)) return 1;

  const double xPlotMax = 10.0;
  const double xChi2Min = 0.0;
  const double xChi2Max = 10.0;

  const int NBINS = 40;
  const double XMIN = 0.0;
  const double XMAX = 10.0;
  const double cutThr = 0.60;

  auto hOS = new TH1D("h_qT_OS_60", "", NBINS, XMIN, XMAX);
  auto hSS = new TH1D("h_qT_SS_60", "", NBINS, XMIN, XMAX);
  hOS->Sumw2();
  hSS->Sumw2();

  Pythia pythia;
  pythia.readString("Beams:idA = -11");
  pythia.readString("Beams:idB = 11");
  pythia.readString("Beams:eCM = 91.2");
  pythia.readString("PDF:lepton = off");
  pythia.readString("HadronLevel:all = on");
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");

  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));

  pythia.readString("StringPT:sigma = " + std::to_string(sigma));
  pythia.readString("TimeShower:alphaSvalue = " + std::to_string(alphaS));

  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  if (!pythia.init()) {
    std::cerr << "Pythia initialization failed\n";
    return 1;
  }

  const double R = 0.4;
  const double jetPtMin = 5.0;
  const double thrustCut = 0.8;
  const double backToBackCut = 2.8;
  JetDefinition jetDef(antikt_algorithm, R);

  std::vector<PseudoJet> fjInputs;
  fjInputs.reserve(250);
  std::vector<Vec4> thrustInputs;
  thrustInputs.reserve(250);

  long long nAcceptedPairs = 0;

  for (long long iEvt = 0; iEvt < nEvents; ++iEvt) {
    if (!pythia.next()) continue;

    auto zq = findZdecayQuarks(pythia.event);
    int zQuark1 = zq.first;
    int zQuark2 = zq.second;
    if (zQuark1 < 0 || zQuark2 < 0) continue;

    fjInputs.clear();
    thrustInputs.clear();

    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (!p.isFinal()) continue;
      if (!p.isVisible()) continue;

      Vec4 v = p.p();
      PseudoJet pj(v.px(), v.py(), v.pz(), v.e());
      pj.set_user_index(i);
      fjInputs.push_back(pj);
      thrustInputs.push_back(Vec4(v.px(), v.py(), v.pz(), 0));
    }

    if (fjInputs.size() < 2) continue;

    ThrustInfo thr = computeThrustInfo(thrustInputs);
    if (thr.T < thrustCut) continue;
    Vec4 nT = thr.axis;

    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));
    if ((int)jets.size() != 2) continue;

    double dphi_jets = wrapToPi(jets[0].phi_std() - jets[1].phi_std());
    if (std::fabs(dphi_jets) < backToBackCut) continue;

    std::vector<HadronCand> pions1, pions2;
    pions1.reserve(60);
    pions2.reserve(60);

    auto collectPions = [&](const PseudoJet& jet, std::vector<HadronCand>& out) {
      for (const auto& c : jet.constituents()) {
        int idx = c.user_index();
        if (idx < 0 || idx >= pythia.event.size()) continue;

        const Particle& p = pythia.event[idx];
        int id = p.id();
        if (id != 211 && id != -211) continue;

        AncestryResult anc = countStepsToQuark(pythia.event, idx, zQuark1, zQuark2);
        if (!anc.foundQuark) continue;

        Vec4 v = p.p();
        int charge = (id == 211) ? +1 : -1;
        double frac = (jet.modp() > 1e-12) ? (v.pAbs() / jet.modp()) : 0.0;
        out.push_back({v, charge, frac});
      }
    };

    collectPions(jets[0], pions1);
    collectPions(jets[1], pions2);

    if (pions1.empty() || pions2.empty()) continue;

    auto byP = [](const HadronCand& x, const HadronCand& y) {
      return x.p4.pAbs() > y.p4.pAbs();
    };

    std::sort(pions1.begin(), pions1.end(), byP);
    std::sort(pions2.begin(), pions2.end(), byP);

    const int MAX_HADRONS_PER_JET = 50;
    if ((int)pions1.size() > MAX_HADRONS_PER_JET) pions1.resize(MAX_HADRONS_PER_JET);
    if ((int)pions2.size() > MAX_HADRONS_PER_JET) pions2.resize(MAX_HADRONS_PER_JET);

    double bestQT = -1.0;
    HadronCand best1{}, best2{};

    for (const auto& h1 : pions1) {
      for (const auto& h2 : pions2) {
        double qt = qT_pair(h1.p4, h2.p4, nT);
        if (qt > bestQT) {
          bestQT = qt;
          best1 = h1;
          best2 = h2;
        }
      }
    }

    if (bestQT < 0.0) continue;

    double minFrac = std::min(best1.frac, best2.frac);
    if (minFrac < cutThr) continue;

    bool isOS = (best1.charge * best2.charge < 0);
    if (isOS) hOS->Fill(bestQT);
    else      hSS->Fill(bestQT);

    nAcceptedPairs++;
  }

  const double thPeakOS = theoryPeakOS(th);
  const double pyPeakOS = pythiaPeakOS(hOS, xPlotMax);
  const double sharedScale = (thPeakOS > 0.0 ? pyPeakOS / thPeakOS : 0.0);

  Chi2Result chiOS = chi2AgainstTheory(hOS, th, sharedScale, true,  xChi2Min, xChi2Max);
  Chi2Result chiSS = chi2AgainstTheory(hSS, th, sharedScale, false, xChi2Min, xChi2Max);

  const double chi2PerOS = (chiOS.nUsed > 0 ? chiOS.chi2 / chiOS.nUsed : std::numeric_limits<double>::quiet_NaN());
  const double chi2PerSS = (chiSS.nUsed > 0 ? chiSS.chi2 / chiSS.nUsed : std::numeric_limits<double>::quiet_NaN());
  const double score = chi2PerOS + chi2PerSS;

  std::ofstream fout(outTxt);
  fout.setf(std::ios::fixed);
  fout.precision(6);
  fout << sigma << " "
       << alphaS << " "
       << nEvents << " "
       << seed << " "
       << sharedScale << " "
       << chiOS.chi2 << " "
       << chiOS.nUsed << " "
       << chi2PerOS << " "
       << chiSS.chi2 << " "
       << chiSS.nUsed << " "
       << chi2PerSS << " "
       << score << " "
       << hOS->Integral() << " "
       << hSS->Integral() << " "
       << nAcceptedPairs << "\n";
  fout.close();

  std::cout << "sigma = " << sigma
            << " , alphaS = " << alphaS
            << " , scale = " << sharedScale
            << " , OS chi2/N = " << chi2PerOS
            << " , SS chi2/N = " << chi2PerSS
            << " , score = " << score
            << "\n";

  return 0;
}
