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
#include <map>
#include <limits>
#include <cstdio>

// ROOT
#include "TFile.h"
#include "TH1D.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TLine.h"
#include "TStyle.h"
#include "TString.h"
#include "TDirectory.h"
#include "TNamed.h"
#include "TParameter.h"

// Pythia / FastJet
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

// brute-force scan
static ThrustResult calculateThrustAxis(const std::vector<fastjet::PseudoJet>& particles) {
  ThrustResult out;
  if (particles.empty()) return out;

  double totalP = 0.0;
  for (const auto& p : particles) {
    totalP += std::sqrt(p.px()*p.px() + p.py()*p.py() + p.pz()*p.pz());
  }
  if (totalP <= 0.0) return out;

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

// ------------------------- version stamp -------------------------

static const char* BUILD_STAMP =
#if defined(__DATE__) && defined(__TIME__)
  __DATE__ " " __TIME__;
#else
  "unknown";
#endif

// ------------------------- compare-mode helpers -------------------------

static double parseSigmaFromPath(const std::string& path) {
  auto pos = path.find("sigma");
  if (pos == std::string::npos) return std::numeric_limits<double>::quiet_NaN();
  pos += 5;
  std::string num;
  while (pos < path.size()) {
    char ch = path[pos];
    if ((ch >= '0' && ch <= '9') || ch=='.' || ch=='+' || ch=='-') { num.push_back(ch); pos++; }
    else break;
  }
  if (num.empty()) return std::numeric_limits<double>::quiet_NaN();
  try { return std::stod(num); }
  catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}

static std::string sigmaTag(double s) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "sigma%.3f", s);
  return std::string(buf);
}

static TH1D* cloneHist1D(TFile* f, const char* key, const char* newname) {
  if (!f) return nullptr;
  auto* h = dynamic_cast<TH1D*>(f->Get(key));
  if (!h) return nullptr;
  auto* c = dynamic_cast<TH1D*>(h->Clone(newname));
  if (!c) return nullptr;
  c->SetDirectory(nullptr);
  return c;
}

// ------------------------- compare-mode (Option A: SAVE hists + canvases) -------------------------

static int runCompareMode(
  const std::vector<std::string>& inputFiles,
  const std::string& outName,
  double baseSigma,
  double qTMax,
  double xDrawMax
) {
  if (inputFiles.empty()) {
    std::cerr << "COMPARE MODE: no input files.\n";
    return 1;
  }

  std::map<double, std::string> fileOf;
  for (const auto& p : inputFiles) {
    double s = parseSigmaFromPath(p);
    if (!std::isfinite(s)) {
      std::cerr << "COMPARE MODE: could not parse sigma from: " << p << "\n";
      return 1;
    }
    fileOf[s] = p;
  }
  if (!fileOf.count(baseSigma)) {
    std::cerr << "COMPARE MODE: baseSigma=" << baseSigma << " not present in inputs.\n";
    return 1;
  }

  TFile* fout = new TFile(outName.c_str(), "RECREATE");
  if (!fout || fout->IsZombie()) {
    std::cerr << "COMPARE MODE: could not create " << outName << "\n";
    return 1;
  }

  TDirectory* dMeta = fout->mkdir("meta");
  dMeta->cd();
  TParameter<double>("baseSigma", baseSigma).Write("baseSigma");
  TParameter<double>("qTMax", qTMax).Write("qTMax");
  TParameter<double>("xDrawMax", xDrawMax).Write("xDrawMax");

  for (const auto& kv : fileOf) {
    TString nm = TString::Format("file_%s", sigmaTag(kv.first).c_str());
    TNamed n(nm.Data(), kv.second.c_str());
    n.Write();
  }

  TDirectory* dClosest = fout->mkdir("closest");
  TDirectory* dHighest = fout->mkdir("highest");

  TDirectory* dClosestOS = dClosest->mkdir("OS");
  TDirectory* dClosestSS = dClosest->mkdir("SS");
  TDirectory* dClosestDR = dClosest->mkdir("doubleRatio");

  TDirectory* dHighestOS = dHighest->mkdir("OS");
  TDirectory* dHighestSS = dHighest->mkdir("SS");
  TDirectory* dHighestDR = dHighest->mkdir("doubleRatio");

  TFile* fBase = TFile::Open(fileOf[baseSigma].c_str(), "READ");
  if (!fBase || fBase->IsZombie()) {
    std::cerr << "COMPARE MODE: failed to open base file.\n";
    fout->Close();
    return 1;
  }

  TH1D* base_cl_OS = cloneHist1D(fBase, "h_qT_thrust_closest_OS", "base_closest_OS");
  TH1D* base_cl_SS = cloneHist1D(fBase, "h_qT_thrust_closest_SS", "base_closest_SS");
  TH1D* base_hi_OS = cloneHist1D(fBase, "h_qT_thrust_highest_OS", "base_highest_OS");
  TH1D* base_hi_SS = cloneHist1D(fBase, "h_qT_thrust_highest_SS", "base_highest_SS");

  fBase->Close();
  delete fBase;

  if (!base_cl_OS || !base_cl_SS || !base_hi_OS || !base_hi_SS) {
    std::cerr << "COMPARE MODE: missing base histograms.\n";
    fout->Close();
    return 1;
  }

  dClosestOS->cd(); base_cl_OS->Write(TString::Format("counts_%s", sigmaTag(baseSigma).c_str()));
  dClosestSS->cd(); base_cl_SS->Write(TString::Format("counts_%s", sigmaTag(baseSigma).c_str()));
  dHighestOS->cd(); base_hi_OS->Write(TString::Format("counts_%s", sigmaTag(baseSigma).c_str()));
  dHighestSS->cd(); base_hi_SS->Write(TString::Format("counts_%s", sigmaTag(baseSigma).c_str()));

  std::map<double,int> colMap;
  colMap[0.280] = kBlack;
  colMap[0.300] = kRed;
  colMap[0.335] = kGreen+2;
  colMap[0.370] = kBlue;
  colMap[0.400] = kViolet;

  for (const auto& kv : fileOf) {
    const double s = kv.first;
    const auto& path = kv.second;
    const auto tag = sigmaTag(s);

    TFile* fin = TFile::Open(path.c_str(), "READ");
    if (!fin || fin->IsZombie()) {
      std::cerr << "COMPARE MODE: failed to open " << path << "\n";
      fout->Close();
      return 1;
    }

    TH1D* clOS = cloneHist1D(fin, "h_qT_thrust_closest_OS", TString::Format("tmp_clOS_%s", tag.c_str()));
    TH1D* clSS = cloneHist1D(fin, "h_qT_thrust_closest_SS", TString::Format("tmp_clSS_%s", tag.c_str()));
    TH1D* hiOS = cloneHist1D(fin, "h_qT_thrust_highest_OS", TString::Format("tmp_hiOS_%s", tag.c_str()));
    TH1D* hiSS = cloneHist1D(fin, "h_qT_thrust_highest_SS", TString::Format("tmp_hiSS_%s", tag.c_str()));

    fin->Close();
    delete fin;

    if (!clOS || !clSS || !hiOS || !hiSS) {
      std::cerr << "COMPARE MODE: missing hists in " << path << "\n";
      fout->Close();
      return 1;
    }

    dClosestOS->cd(); clOS->Write(TString::Format("counts_%s", tag.c_str()));
    dClosestSS->cd(); clSS->Write(TString::Format("counts_%s", tag.c_str()));
    dHighestOS->cd(); hiOS->Write(TString::Format("counts_%s", tag.c_str()));
    dHighestSS->cd(); hiSS->Write(TString::Format("counts_%s", tag.c_str()));

    auto makeDoubleRatio = [&](TH1D* os_s, TH1D* ss_s, TH1D* os_0, TH1D* ss_0, const char* name) -> TH1D* {
      auto* D = dynamic_cast<TH1D*>(os_s->Clone(name));
      D->Reset("ICES");
      D->SetDirectory(nullptr);
      const int nb = D->GetNbinsX();
      for (int b = 1; b <= nb; ++b) {
        const double A  = os_s->GetBinContent(b);
        const double A0 = os_0->GetBinContent(b);
        const double C  = ss_s->GetBinContent(b);
        const double C0 = ss_0->GetBinContent(b);
        if (A0 <= 0 || C <= 0 || C0 <= 0) continue;
        D->SetBinContent(b, (A * C0) / (A0 * C));
      }
      return D;
    };

    if (s != baseSigma) {
      TH1D* Dcl = makeDoubleRatio(clOS, clSS, base_cl_OS, base_cl_SS,
                                  TString::Format("doubleRatio_%s", tag.c_str()));
      TH1D* Dhi = makeDoubleRatio(hiOS, hiSS, base_hi_OS, base_hi_SS,
                                  TString::Format("doubleRatio_%s", tag.c_str()));
      dClosestDR->cd(); if (Dcl) Dcl->Write();
      dHighestDR->cd(); if (Dhi) Dhi->Write();
    }
  }

  auto makeCanvas = [&](const char* which) -> TCanvas* {
    bool isClosest = (std::string(which) == "closest");
    TH1D* baseOS = isClosest ? base_cl_OS : base_hi_OS;
    TH1D* baseSS = isClosest ? base_cl_SS : base_hi_SS;

    if (!baseOS || !baseSS) return nullptr;

    gStyle->SetOptStat(0);

    TCanvas* c = new TCanvas(Form("c_sigmaScan_%s", which), Form("c_sigmaScan_%s", which), 1100, 800);

    TPad* p1 = new TPad(Form("p_%s_top", which), "", 0, 0.32, 1, 1);
    TPad* p2 = new TPad(Form("p_%s_bot", which), "", 0, 0.00, 1, 0.32);

    p1->SetTopMargin(0.08);
    p1->SetBottomMargin(0.05);
    p1->SetLeftMargin(0.10);
    p1->SetRightMargin(0.06);

    p2->SetTopMargin(0.03);
    p2->SetBottomMargin(0.34);
    p2->SetLeftMargin(0.10);
    p2->SetRightMargin(0.06);

    p1->Draw();
    p2->Draw();

    p1->cd();

    baseOS->SetLineColor(kRed);
    baseSS->SetLineColor(kBlue);

    baseOS->SetTitle(which);
    baseOS->GetYaxis()->SetTitle("Events");

    baseOS->GetYaxis()->SetTitleSize(0.045);
    baseOS->GetYaxis()->SetTitleOffset(1.05);
    baseOS->GetYaxis()->SetLabelSize(0.040);

    baseOS->GetXaxis()->SetLabelSize(0.0);
    baseOS->GetXaxis()->SetTitleSize(0.0);

    const double maxY = std::max(baseOS->GetMaximum(), baseSS->GetMaximum());
    baseOS->SetMaximum(1.25 * maxY);

    baseOS->Draw("hist");
    baseSS->Draw("hist same");
    baseOS->SetTitleSize(0.055, "t");

    TLegend* legTop = new TLegend(0.65, 0.78, 0.90, 0.90);
    legTop->SetBorderSize(0);
    legTop->SetFillStyle(0);
    legTop->AddEntry(baseOS, Form("OS (base #sigma=%.3f)", baseSigma), "l");
    legTop->AddEntry(baseSS, Form("SS (base #sigma=%.3f)", baseSigma), "l");
    legTop->Draw();

    p2->cd();

    const double xAxisMin = 0.0;
    const double xAxisMax = qTMax;

    auto frame = p2->DrawFrame(xAxisMin, 0.6, xAxisMax, 1.4);
    frame->SetStats(0);
    frame->GetYaxis()->SetTitle("ratio to base");
    frame->GetXaxis()->SetTitle("q_{T}^{thrust} [GeV]");

    frame->GetYaxis()->SetNdivisions(505);
    frame->GetYaxis()->SetTitleSize(0.10);
    frame->GetYaxis()->SetTitleOffset(0.50);
    frame->GetYaxis()->SetLabelSize(0.09);

    frame->GetXaxis()->SetTitleSize(0.12);
    frame->GetXaxis()->SetTitleOffset(1.05);
    frame->GetXaxis()->SetLabelSize(0.10);

    TLine* one = new TLine(xAxisMin, 1.0, xAxisMax, 1.0);
    one->SetLineStyle(2);
    one->Draw();

    TLegend* legBot = new TLegend(0.60, 0.62, 0.90, 0.90);
    legBot->SetBorderSize(0);
    legBot->SetFillStyle(0);
    legBot->SetTextSize(0.030);
    legBot->SetHeader("D = (OS/OS_{0})/(SS/SS_{0})", "C");

    for (const auto& kv : fileOf) {
      const double s = kv.first;
      if (s == baseSigma) continue;

      fout->cd(Form("%s/doubleRatio", which));
      const char* hname = Form("doubleRatio_%s", sigmaTag(s).c_str());
      TH1D* D = dynamic_cast<TH1D*>(gDirectory->Get(hname));
      if (!D) continue;

      D->SetLineWidth(2);
      int col = colMap.count(s) ? colMap[s] : kBlack;
      D->SetLineColor(col);

      D->GetXaxis()->SetRangeUser(xAxisMin, xDrawMax);
      D->Draw("hist same");

      legBot->AddEntry(D, TString::Format("#sigma=%.3f", s), "l");
    }

    legBot->Draw();
    c->Update();
    return c;
  };

  dClosest->cd();
  if (auto c1 = makeCanvas("closest")) c1->Write("c_sigmaScan_closest");

  dHighest->cd();
  if (auto c2 = makeCanvas("highest")) c2->Write("c_sigmaScan_highest");

  fout->Close();
  delete fout;

  std::cout << "COMPARE MODE: wrote compare file to " << outName << "\n";
  return 0;
}

// ------------------------- main -------------------------

struct Args {
  bool interactive = false;
  bool compareMode = false;

  double sigma = 0.280;
  int events = 150000;
  std::string outFile = "pythia1.root";

  std::string compareOut = "compare_sigma.root";
  double baseSigma = 0.280;
  double qTMax = 10.0;
  double xDrawMax = 3.0;

  std::vector<std::string> compareFiles;
};

static bool isFlag(const std::string& s) {
  return (s.rfind("--", 0) == 0);
}

int main(int argc, char* argv[]) {
  Args A;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--interactive" || arg == "-i") {
      A.interactive = true;
    } else if (arg == "--sigma" && i+1 < argc) {
      A.sigma = std::stod(argv[++i]);
    } else if (arg == "--events" && i+1 < argc) {
      A.events = std::stoi(argv[++i]);
    } else if (arg == "--out" && i+1 < argc) {
      A.outFile = argv[++i];
    } else if (arg == "--qTMax" && i+1 < argc) {
      A.qTMax = std::stod(argv[++i]);
    } else if (arg == "--compareOut" && i+1 < argc) {
      A.compareMode = true;
      A.compareOut = argv[++i];
    } else if (arg == "--baseSigma" && i+1 < argc) {
      A.baseSigma = std::stod(argv[++i]);
    } else if (arg == "--xDrawMax" && i+1 < argc) {
      A.xDrawMax = std::stod(argv[++i]);
    } else if (arg == "--compare") {
      A.compareMode = true;
      while (i+1 < argc && !isFlag(argv[i+1])) {
        A.compareFiles.push_back(argv[++i]);
      }
    } else if (arg == "--version") {
      std::cout << "BUILD_STAMP: " << BUILD_STAMP << "\n";
      return 0;
    }
  }

  if (A.compareMode) {
    if (A.compareFiles.empty()) {
      std::cerr << "COMPARE MODE: no files passed after --compare\n";
      return 1;
    }
    return runCompareMode(A.compareFiles, A.compareOut, A.baseSigma, A.qTMax, A.xDrawMax);
  }

  // generator mode
  TFile* fout = new TFile(A.outFile.c_str(), "RECREATE");

  const int    qTBins = 200;
  const double qTMax  = A.qTMax;

  TH1D* h_qT_thrust_closest_OS = new TH1D("h_qT_thrust_closest_OS","closest OS;q_{T}^{thrust} [GeV];Events",qTBins,0,qTMax);
  TH1D* h_qT_thrust_closest_SS = new TH1D("h_qT_thrust_closest_SS","closest SS;q_{T}^{thrust} [GeV];Events",qTBins,0,qTMax);
  TH1D* h_qT_thrust_highest_OS = new TH1D("h_qT_thrust_highest_OS","highest OS;q_{T}^{thrust} [GeV];Events",qTBins,0,qTMax);
  TH1D* h_qT_thrust_highest_SS = new TH1D("h_qT_thrust_highest_SS","highest SS;q_{T}^{thrust} [GeV];Events",qTBins,0,qTMax);

  h_qT_thrust_closest_OS->SetDirectory(nullptr);
  h_qT_thrust_closest_SS->SetDirectory(nullptr);
  h_qT_thrust_highest_OS->SetDirectory(nullptr);
  h_qT_thrust_highest_SS->SetDirectory(nullptr);

  Pythia pythia;
  pythia.readString("Beams:idA = -11");
  pythia.readString("Beams:idB = 11");
  pythia.readString("Beams:eCM = 91.2");
  pythia.readString("PDF:lepton = off");
  pythia.readString("HadronLevel:all = on");
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 123456788");
  pythia.readString(TString::Format("StringPT:sigma = %.6f", A.sigma).Data());

  if (!pythia.init()) {
    std::cerr << "Pythia initialization failed\n";
    return 1;
  }

  const int nEvents = A.events;

  const double R = 0.4;
  const double jetPtMin = 5.0;
  const double thrustCut = 0.8;
  const double backToBackCut = 2.8;

  for (int ievt = 0; ievt < nEvents; ++ievt) {
    if (!pythia.next()) continue;

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
      PseudoJet pj(pythia.event[idx].px(), pythia.event[idx].py(), pythia.event[idx].pz(), pythia.event[idx].e());
      pj.set_user_index(idx);
      fjInputs.push_back(pj);
    }

    ThrustResult thr = calculateThrustAxis(fjInputs);
    if (thr.T < thrustCut) continue;

    JetDefinition jetDef(antikt_algorithm, R);
    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));
    if ((int)jets.size() != 2) continue;

    double dphi_jets = wrapToPi(jets[0].phi_std() - jets[1].phi_std());
    if (std::fabs(dphi_jets) < backToBackCut) continue;

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

    auto pions0 = collectPions(jets[0]);
    auto pions1 = collectPions(jets[1]);
    if (pions0.empty() || pions1.empty()) continue;

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

    auto pickLead = [&](const std::optional<PionInfo>& pos, const std::optional<PionInfo>& neg, bool useClosest) -> std::optional<PionInfo> {
      if (pos && neg) {
        if (useClosest) return betterClosest(*pos, *neg) ? pos : neg;
        return betterHighest(*pos, *neg) ? pos : neg;
      }
      return pos ? pos : neg;
    };

    auto c0_pos = bestByCharge(pions0, +1, true);
    auto c0_neg = bestByCharge(pions0, -1, true);
    auto c1_pos = bestByCharge(pions1, +1, true);
    auto c1_neg = bestByCharge(pions1, -1, true);
    auto c0_lead = pickLead(c0_pos, c0_neg, true);
    auto c1_lead = pickLead(c1_pos, c1_neg, true);

    auto h0_pos = bestByCharge(pions0, +1, false);
    auto h0_neg = bestByCharge(pions0, -1, false);
    auto h1_pos = bestByCharge(pions1, +1, false);
    auto h1_neg = bestByCharge(pions1, -1, false);
    auto h0_lead = pickLead(h0_pos, h0_neg, false);
    auto h1_lead = pickLead(h1_pos, h1_neg, false);

    if (c0_lead && c1_lead) {
      const Particle& p0 = pythia.event[c0_lead->idx];
      const Particle& p1 = pythia.event[c1_lead->idx];
      double qT = qT_thrust(p0, p1, thr.nHat);
      bool isOS = (c0_lead->charge != c1_lead->charge);
      if (isOS) h_qT_thrust_closest_OS->Fill(qT);
      else      h_qT_thrust_closest_SS->Fill(qT);
    }

    if (h0_lead && h1_lead) {
      const Particle& p0 = pythia.event[h0_lead->idx];
      const Particle& p1 = pythia.event[h1_lead->idx];
      double qT = qT_thrust(p0, p1, thr.nHat);
      bool isOS = (h0_lead->charge != h1_lead->charge);
      if (isOS) h_qT_thrust_highest_OS->Fill(qT);
      else      h_qT_thrust_highest_SS->Fill(qT);
    }
  }

  fout->cd();
  h_qT_thrust_closest_OS->Write();
  h_qT_thrust_closest_SS->Write();
  h_qT_thrust_highest_OS->Write();
  h_qT_thrust_highest_SS->Write();

  fout->Close();
  delete fout;

  std::cout << "Output written to: " << A.outFile << "\n";
  pythia.stat();
  return 0;
}
