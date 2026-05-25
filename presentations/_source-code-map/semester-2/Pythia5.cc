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

static inline Vec3 transverseTo(const Vec3& p, const Vec3& n) {
  return p - dot(p,n)*n;
}

static inline double qT_axis(const Vec3& a, const Vec3& b, const Vec3& nHat) {
  Vec3 P = a + b;
  Vec3 PT = transverseTo(P, nHat);
  return norm(PT);
}

struct ThrustResult {
  double T = 0.0;
  Vec3 nHat{0,0,1};
};

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

static inline bool betterHighest(const PionInfo& a, const PionInfo& b) {
  return a.pT > b.pT;
}

// ------------------------- version stamp -------------------------

static const char* BUILD_STAMP =
#if defined(__DATE__) && defined(__TIME__)
  __DATE__ " " __TIME__
#else
  "unknown"
#endif
;

// ------------------------- shared helpers -------------------------

static double parseValueFromPath(const std::string& path, const std::string& key) {
  auto pos = path.find(key);
  if (pos == std::string::npos) return std::numeric_limits<double>::quiet_NaN();
  pos += key.size();

  std::string num;
  while (pos < path.size()) {
    char ch = path[pos];
    if ((ch >= '0' && ch <= '9') || ch=='.' || ch=='+' || ch=='-') {
      num.push_back(ch);
      pos++;
    } else {
      break;
    }
  }

  if (num.empty()) return std::numeric_limits<double>::quiet_NaN();
  try { return std::stod(num); }
  catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}

static const char* tuneName(int t) {
  switch (t) {
    case -1: return "Reset-ee";
    case  0: return "NoTune";
    case  1: return "OldJETSET";
    case  2: return "Montull2007";
    case  3: return "Hoeth2009";
    case  4: return "Skands2013";
    case  5: return "Fischer2013-1";
    case  6: return "Fischer2013-2";
    case  7: return "Monash2013-ee";
    default: return "Unknown";
  }
}

static TString fmtValLabel(double v, double baseVal) {
  int iv = (int) std::lround(v);
  int ib = (int) std::lround(baseVal);
  if (iv == ib) return TString::Format("Tune:ee=%d (%s, base)", iv, tuneName(iv));
  return TString::Format("Tune:ee=%d (%s)", iv, tuneName(iv));
}

static std::string stripRootExt(const std::string& s) {
  if (s.size() >= 5 && s.substr(s.size()-5) == ".root") return s.substr(0, s.size()-5);
  return s;
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

static TH1D* makeShapeFromCounts(const TH1D* hCounts, const char* newname) {
  if (!hCounts) return nullptr;
  auto* h = dynamic_cast<TH1D*>(hCounts->Clone(newname));
  if (!h) return nullptr;
  h->SetDirectory(nullptr);
  const int nb = h->GetNbinsX();
  const double N = hCounts->Integral(1, nb);
  if (N > 0.0) h->Scale(1.0 / N, "width");
  return h;
}

static void styleTopAxis(TH1D* h, const char* ytitle) {
  h->SetStats(0);
  h->GetYaxis()->SetTitle(ytitle);
  h->GetYaxis()->SetTitleSize(0.055);
  h->GetYaxis()->SetTitleOffset(1.00);
  h->GetYaxis()->SetLabelSize(0.045);
  h->GetXaxis()->SetLabelSize(0.0);
  h->GetXaxis()->SetTitleSize(0.0);
}

static void styleRatioFrame(TH1D* frame, const char* ytitle) {
  frame->SetStats(0);
  frame->GetYaxis()->SetTitle(ytitle);
  frame->GetXaxis()->SetTitle("q_{T} [GeV]");
  frame->GetYaxis()->SetNdivisions(505);
  frame->GetYaxis()->SetTitleSize(0.12);
  frame->GetYaxis()->SetTitleOffset(0.55);
  frame->GetYaxis()->SetLabelSize(0.10);
  frame->GetXaxis()->SetTitleSize(0.14);
  frame->GetXaxis()->SetTitleOffset(1.05);
  frame->GetXaxis()->SetLabelSize(0.12);
}

// ------------------------- compare-mode canvases -------------------------

static TCanvas* makeSingle_OS_or_SS(
  const char* which,
  const char* kind,
  const std::vector<double>& vals,
  double baseTune,
  double qTMax,
  double xDrawMax,
  const std::map<double,int>& colMap,
  const std::vector<TH1D*>& H,
  TH1D* Hbase
) {
  gStyle->SetOptStat(0);
  (void)colMap;

  auto colorFor = [&](double s) -> int {
    int iv = (int) std::lround(s);
    if (iv == 3) return kRed;
    if (iv == 4) return kBlue;
    if (iv == 7) return kGreen + 2;
    return kMagenta + 2;
  };

  TString plotTitle = TString::Format("Highest %s", which);

  TString cname = TString::Format("c_highest_%s_%s", which, kind);
  TCanvas* c = new TCanvas(cname, cname, 1100, 800);

  TPad* p1 = new TPad("top", "", 0, 0.32, 1, 1);
  TPad* p2 = new TPad("bot", "", 0, 0.00, 1, 0.32);

  p1->SetTopMargin(0.08); p1->SetBottomMargin(0.05);
  p1->SetLeftMargin(0.12); p1->SetRightMargin(0.06);
  p2->SetTopMargin(0.03); p2->SetBottomMargin(0.34);
  p2->SetLeftMargin(0.12); p2->SetRightMargin(0.06);

  p1->Draw();
  p2->Draw();

  const char* yTop = (std::string(kind) == "shape")
    ? "(1/N) dN/dq_{T} [GeV^{-1}]"
    : "Events";

  // top pad
  p1->cd();

  Hbase->SetTitle(plotTitle);
  Hbase->SetLineColor(colorFor(baseTune));
  Hbase->SetLineWidth(2);
  styleTopAxis(Hbase, yTop);

  double maxY = 0.0;
  for (auto* h : H) {
    if (!h) continue;
    maxY = std::max(maxY, h->GetMaximum());
  }
  Hbase->SetMaximum(maxY > 0.0 ? 1.25 * maxY : 1.0);

  Hbase->Draw("hist");

  for (size_t i = 0; i < vals.size(); ++i) {
    double s = vals[i];
    if (std::fabs(s - baseTune) < 1e-12) continue;

    TH1D* h = H[i];
    if (!h) continue;

    h->SetLineColor(colorFor(s));
    h->SetLineWidth(2);
    h->Draw("hist same");
  }

  Hbase->SetLineColor(colorFor(baseTune));
  Hbase->SetLineWidth(2);
  Hbase->Draw("hist same");

  TLegend* leg = new TLegend(0.62, 0.68, 0.90, 0.90);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.040);

  TLine* l1 = new TLine(0,0,1,0);
  TLine* l2 = new TLine(0,0,1,0);
  TLine* l3 = new TLine(0,0,1,0);
  l1->SetLineColor(kRed);       l1->SetLineWidth(2);
  l2->SetLineColor(kBlue);      l2->SetLineWidth(2);
  l3->SetLineColor(kGreen + 2); l3->SetLineWidth(2);

  leg->AddEntry(l1, "Tune:ee=3 (Hoeth2009)", "l");
  leg->AddEntry(l2, "Tune:ee=4 (Skands2013)", "l");
  leg->AddEntry(l3, "Tune:ee=7 (Monash2013-ee, base)", "l");
  leg->Draw();

  // bottom pad
  p2->cd();

  TH1D* frame = (TH1D*)Hbase->Clone(Form("frame_%s_%s", which, kind));
  frame->Reset("ICES");
  frame->SetDirectory(nullptr);
  frame->SetTitle("");
  frame->GetXaxis()->SetLimits(0.0, qTMax);
  frame->SetMinimum(0.6);
  frame->SetMaximum(1.4);
  styleRatioFrame(frame, "ratio to base");
  frame->Draw();

  TLine* one = new TLine(0.0, 1.0, qTMax, 1.0);
  one->SetLineStyle(2);
  one->SetLineWidth(1);
  one->SetLineColor(kBlack);
  one->Draw();

  for (size_t i = 0; i < vals.size(); ++i) {
    double s = vals[i];
    if (std::fabs(s - baseTune) < 1e-12) continue;

    TH1D* h = H[i];
    if (!h) continue;

    int iv = (int) std::lround(s);
    TH1D* r = (TH1D*)h->Clone(Form("ratio_%s_%s_%d", which, kind, iv));
    r->SetDirectory(nullptr);
    r->SetTitle("");
    r->Divide(Hbase);

    r->SetLineColor(colorFor(s));
    r->SetLineWidth(1);
    r->GetXaxis()->SetRangeUser(0.0, xDrawMax);
    r->Draw("hist same");
  }

  c->Update();
  return c;
}

static TCanvas* makeOSSS_bias(
  const char* kind,
  const std::vector<double>& vals,
  double baseTune,
  double qTMax,
  double xDrawMax,
  const std::map<double,int>& colMap,
  const std::vector<TH1D*>& OS,
  const std::vector<TH1D*>& SS,
  TH1D* OS0,
  TH1D* SS0
) {
  gStyle->SetOptStat(0);

  TString plotTitle = "Highest OS vs SS";

  TString cname = TString::Format("c_highest_OSSS_bias_%s", kind);
  TCanvas* c = new TCanvas(cname, cname, 1100, 800);

  TPad* p1 = new TPad("top", "", 0, 0.32, 1, 1);
  TPad* p2 = new TPad("bot", "", 0, 0.00, 1, 0.32);

  p1->SetTopMargin(0.08); p1->SetBottomMargin(0.05);
  p1->SetLeftMargin(0.12); p1->SetRightMargin(0.06);
  p2->SetTopMargin(0.03); p2->SetBottomMargin(0.34);
  p2->SetLeftMargin(0.12); p2->SetRightMargin(0.06);

  p1->Draw();
  p2->Draw();

  const bool isShape = (std::string(kind) == "shape");
  const char* yTop = isShape
    ? "(1/N) dN/dq_{T} [GeV^{-1}]"
    : "Events";

  // top
  p1->cd();

  TH1D* axis = (TH1D*)OS0->Clone("axisTop");
  axis->SetDirectory(nullptr);
  axis->SetTitle(plotTitle);
  styleTopAxis(axis, yTop);

  TH1D* OS0draw = (TH1D*)OS0->Clone(Form("OS0draw_%s", kind));
  TH1D* SS0draw = (TH1D*)SS0->Clone(Form("SS0draw_%s", kind));
  OS0draw->SetDirectory(nullptr);
  SS0draw->SetDirectory(nullptr);

  axis->SetMaximum(isShape ? 1.0 : 1400.0);
  axis->Draw("hist");

  OS0draw->SetLineColor(kRed);
  OS0draw->SetLineWidth(2);
  OS0draw->Draw("hist same");

  SS0draw->SetLineColor(kBlue);
  SS0draw->SetLineWidth(2);
  SS0draw->Draw("hist same");

  TLegend* legTop = new TLegend(0.62, 0.78, 0.90, 0.90);
  legTop->SetBorderSize(0);
  legTop->SetFillStyle(0);
  legTop->SetTextSize(0.040);

  for (double s : vals) {
    if (std::fabs(s - baseTune) < 1e-12) continue;

    int col = colMap.count(s) ? colMap.at(s) : kBlack;
    if (col != kRed && col != kBlue) continue;

    TLine* l = new TLine(0,0,1,0);
    l->SetLineColor(col);
    l->SetLineWidth(2);
    legTop->AddEntry(l, fmtValLabel(s, baseTune), "l");
  }

  legTop->Draw();

  // bottom
  p2->cd();
  TH1D* frame = (TH1D*)OS0->Clone(Form("frame_bias_%s", kind));
  frame->Reset("ICES");
  frame->SetDirectory(nullptr);
  frame->SetTitle("");
  frame->GetXaxis()->SetLimits(0.0, qTMax);
  frame->SetMinimum(0.6);
  frame->SetMaximum(1.4);
  styleRatioFrame(frame, "ratio to base");
  frame->Draw();

  TLine* one = new TLine(0.0, 1.0, qTMax, 1.0);
  one->SetLineStyle(2);
  one->SetLineWidth(1);
  one->SetLineColor(kBlack);
  one->Draw();

  for (size_t i = 0; i < vals.size(); ++i) {
    double s = vals[i];
    if (std::fabs(s - baseTune) < 1e-12) continue;

    TH1D* OSs = OS[i];
    TH1D* SSs = SS[i];
    if (!OSs || !SSs) continue;

    int iv = (int) std::lround(s);
    TH1D* D = (TH1D*)OSs->Clone(Form("D_%s_%d", kind, iv));
    D->Reset("ICES");
    D->SetDirectory(nullptr);
    D->SetTitle("");

    const int nb = D->GetNbinsX();
    for (int b = 1; b <= nb; ++b) {
      double os_s = OSs->GetBinContent(b);
      double ss_s = SSs->GetBinContent(b);
      double os_0 = OS0->GetBinContent(b);
      double ss_0 = SS0->GetBinContent(b);

      if (os_s <= 0 || ss_s <= 0 || os_0 <= 0 || ss_0 <= 0) continue;

      D->SetBinContent(b, (os_s * ss_0) / (ss_s * os_0));
    }

    int col = colMap.count(s) ? colMap.at(s) : kBlack;
    D->SetLineColor(col);
    D->SetLineWidth(1);
    D->GetXaxis()->SetRangeUser(0.0, xDrawMax);
    D->Draw("hist same");
  }

  c->Update();
  return c;
}

// ------------------------- compare-mode -------------------------

static int runCompareMode(
  const std::vector<std::string>& inputFiles,
  const std::string& outName,
  double baseTune,
  double qTMax,
  double xDrawMax
) {
  if (inputFiles.empty()) {
    std::cerr << "COMPARE MODE: no input files.\n";
    return 1;
  }

  std::map<double, std::string> fileOf;
  for (const auto& p : inputFiles) {
    double s = parseValueFromPath(p, "tune");
    if (!std::isfinite(s)) {
      std::cerr << "COMPARE MODE: could not parse tune from: " << p << "\n";
      return 1;
    }
    fileOf[s] = p;
  }

  if (!fileOf.count(baseTune)) {
    std::cerr << "COMPARE MODE: baseTune=" << baseTune << " not present in inputs.\n";
    return 1;
  }

  std::vector<double> vals;
  for (const auto& kv : fileOf) vals.push_back(kv.first);
  std::sort(vals.begin(), vals.end());

  std::map<double,int> colMap;
  colMap[3.0] = kRed;
  colMap[4.0] = kBlue;
  colMap[7.0] = kBlack;
  for (double s : vals) if (!colMap.count(s)) colMap[s] = kBlack;

  TFile* fout = new TFile(outName.c_str(), "RECREATE");
  if (!fout || fout->IsZombie()) {
    std::cerr << "COMPARE MODE: could not create " << outName << "\n";
    return 1;
  }

  TDirectory* dMeta = fout->mkdir("meta");
  dMeta->cd();
  TParameter<int>("baseTuneEE", (int)std::lround(baseTune)).Write("baseTuneEE");
  TParameter<double>("qTMax", qTMax).Write("qTMax");
  TParameter<double>("xDrawMax", xDrawMax).Write("xDrawMax");

  TDirectory* dHigh = fout->mkdir("highest");
  TDirectory* dOS   = dHigh->mkdir("OS");
  TDirectory* dSS   = dHigh->mkdir("SS");

  std::vector<TH1D*> OS_counts(vals.size(), nullptr), SS_counts(vals.size(), nullptr);
  std::vector<TH1D*> OS_shape (vals.size(), nullptr), SS_shape (vals.size(), nullptr);

  TH1D* OS0_counts = nullptr; TH1D* SS0_counts = nullptr;
  TH1D* OS0_shape  = nullptr; TH1D* SS0_shape  = nullptr;

  for (size_t i = 0; i < vals.size(); ++i) {
    double s = vals[i];
    int iv = (int) std::lround(s);

    TFile* fin = TFile::Open(fileOf[s].c_str(), "READ");
    if (!fin || fin->IsZombie()) {
      std::cerr << "COMPARE MODE: failed to open " << fileOf[s] << "\n";
      fout->Close();
      return 1;
    }

    OS_counts[i] = cloneHist1D(fin, "h_qT_thrust_highest_OS", Form("OS_counts_%d", iv));
    SS_counts[i] = cloneHist1D(fin, "h_qT_thrust_highest_SS", Form("SS_counts_%d", iv));

    fin->Close();
    delete fin;

    if (!OS_counts[i] || !SS_counts[i]) {
      std::cerr << "COMPARE MODE: missing highest histograms in " << fileOf[s] << "\n";
      fout->Close();
      return 1;
    }

    OS_shape[i] = makeShapeFromCounts(OS_counts[i], Form("OS_shape_%d", iv));
    SS_shape[i] = makeShapeFromCounts(SS_counts[i], Form("SS_shape_%d", iv));

    dOS->cd();
    OS_counts[i]->Write(Form("counts_tune%d", iv));
    if (OS_shape[i]) OS_shape[i]->Write(Form("shape_tune%d", iv));

    dSS->cd();
    SS_counts[i]->Write(Form("counts_tune%d", iv));
    if (SS_shape[i]) SS_shape[i]->Write(Form("shape_tune%d", iv));

    if (std::fabs(s - baseTune) < 1e-12) {
      OS0_counts = OS_counts[i];
      SS0_counts = SS_counts[i];
      OS0_shape  = OS_shape[i];
      SS0_shape  = SS_shape[i];
    }
  }

  if (!OS0_counts || !SS0_counts || !OS0_shape || !SS0_shape) {
    std::cerr << "COMPARE MODE: could not identify base histograms.\n";
    fout->Close();
    return 1;
  }

  TCanvas* cOS_counts = makeSingle_OS_or_SS("OS", "counts", vals, baseTune, qTMax, xDrawMax, colMap, OS_counts, OS0_counts);
  TCanvas* cSS_counts = makeSingle_OS_or_SS("SS", "counts", vals, baseTune, qTMax, xDrawMax, colMap, SS_counts, SS0_counts);
  TCanvas* cOS_shape  = makeSingle_OS_or_SS("OS", "shape",  vals, baseTune, qTMax, xDrawMax, colMap, OS_shape,  OS0_shape);
  TCanvas* cSS_shape  = makeSingle_OS_or_SS("SS", "shape",  vals, baseTune, qTMax, xDrawMax, colMap, SS_shape,  SS0_shape);

  TCanvas* cBias_counts = makeOSSS_bias("counts", vals, baseTune, qTMax, xDrawMax, colMap, OS_counts, SS_counts, OS0_counts, SS0_counts);
  TCanvas* cBias_shape  = makeOSSS_bias("shape",  vals, baseTune, qTMax, xDrawMax, colMap, OS_shape,  SS_shape,  OS0_shape,  SS0_shape);

  dHigh->cd();
  if (cOS_counts)   cOS_counts->Write("c_highest_OS_counts");
  if (cSS_counts)   cSS_counts->Write("c_highest_SS_counts");
  if (cOS_shape)    cOS_shape->Write("c_highest_OS_shape");
  if (cSS_shape)    cSS_shape->Write("c_highest_SS_shape");
  if (cBias_counts) cBias_counts->Write("c_highest_OSSS_bias_counts");
  if (cBias_shape)  cBias_shape->Write("c_highest_OSSS_bias_shape");

  std::string prefix = stripRootExt(outName);
  auto saveBoth = [&](TCanvas* c, const std::string& name) {
    if (!c) return;
    c->SaveAs((prefix + "_" + name + ".png").c_str());
    c->SaveAs((prefix + "_" + name + ".pdf").c_str());
  };

  saveBoth(cOS_counts,   "highest_OS_counts");
  saveBoth(cSS_counts,   "highest_SS_counts");
  saveBoth(cOS_shape,    "highest_OS_shape");
  saveBoth(cSS_shape,    "highest_SS_shape");
  saveBoth(cBias_counts, "highest_OSSS_bias_counts");
  saveBoth(cBias_shape,  "highest_OSSS_bias_shape");

  fout->Close();
  delete fout;

  std::cout << "COMPARE MODE: wrote compare file + images to " << outName << "\n";
  return 0;
}

// ------------------------- main -------------------------

struct Args {
  bool compareMode = false;

  int eeTune = 7;
  int events = 150000;
  double qTMax = 10.0;
  std::string outFile = "pythia1.root";

  std::string compareOut = "compare_tuneEE.root";
  int baseTune = 7;
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

    if (arg == "--eeTune" && i+1 < argc) {
      A.eeTune = std::stoi(argv[++i]);
    } else if (arg == "--events" && i+1 < argc) {
      A.events = std::stoi(argv[++i]);
    } else if (arg == "--qTMax" && i+1 < argc) {
      A.qTMax = std::stod(argv[++i]);
    } else if (arg == "--out" && i+1 < argc) {
      A.outFile = argv[++i];
    } else if (arg == "--compareOut" && i+1 < argc) {
      A.compareMode = true;
      A.compareOut = argv[++i];
    } else if (arg == "--baseTune" && i+1 < argc) {
      A.baseTune = std::stoi(argv[++i]);
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
    return runCompareMode(A.compareFiles, A.compareOut, A.baseTune, A.qTMax, A.xDrawMax);
  }

  // generator mode
  TFile* fout = new TFile(A.outFile.c_str(), "RECREATE");

  {
    TDirectory* dMeta = fout->mkdir("meta");
    dMeta->cd();
    TParameter<int>("Tune_ee", A.eeTune).Write("Tune_ee");
    TParameter<int>("nEvents", A.events).Write("nEvents");
    TParameter<double>("qTMax", A.qTMax).Write("qTMax");
  }

  const int    qTBins = 200;
  const double qTMax  = A.qTMax;

  TH1D* h_qT_thrust_highest_OS = new TH1D("h_qT_thrust_highest_OS","highest OS;q_{T} [GeV];Events",qTBins,0,qTMax);
  TH1D* h_qT_thrust_highest_SS = new TH1D("h_qT_thrust_highest_SS","highest SS;q_{T} [GeV];Events",qTBins,0,qTMax);

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

  pythia.readString(TString::Format("Tune:ee = %d", A.eeTune).Data());

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

    auto pickBestAnyCharge = [&](const std::vector<PionInfo>& v) -> std::optional<PionInfo> {
      std::optional<PionInfo> best;
      for (const auto& pi : v) {
        if (!best) best = pi;
        else if (betterHighest(pi, *best)) best = pi;
      }
      return best;
    };

    auto lead0 = pickBestAnyCharge(pions0);
    auto lead1 = pickBestAnyCharge(pions1);
    if (!lead0 || !lead1) continue;

    const Particle& p0 = pythia.event[lead0->idx];
    const Particle& p1 = pythia.event[lead1->idx];

    Vec3 v0(p0.px(), p0.py(), p0.pz());
    Vec3 v1(p1.px(), p1.py(), p1.pz());

    double qT = qT_axis(v0, v1, thr.nHat);

    bool isOS = (lead0->charge != lead1->charge);
    if (isOS) h_qT_thrust_highest_OS->Fill(qT);
    else      h_qT_thrust_highest_SS->Fill(qT);
  }

  fout->cd();
  h_qT_thrust_highest_OS->Write();
  h_qT_thrust_highest_SS->Write();

  fout->Close();
  delete fout;

  std::cout << "Output written to: " << A.outFile << "\n";
  pythia.stat();
  return 0;
}