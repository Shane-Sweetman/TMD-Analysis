// Semester-2/Pythia1.cc
// Produces a ROOT file with:
//   - tPionPairs (reduced selected-pair TTree)
//   - h_qT_highest_OS_pion_cut{0,20,40,60}
//   - h_qT_highest_SS_pion_cut{0,20,40,60}
//   - c_qT_OSSS_4cuts_pion_counts
//   - c_qT_OSSS_4cuts_pion_norm
//   - c_qT_OSSS_pion_cut{0,20,40,60}_counts
//   - c_qT_OSSS_pion_cut{0,20,40,60}_norm
//
// Run: ./TMD [NEVENTS] [SEED] [OUTFILE] [PROGRESSFILE]
// Example: ./TMD 25000000 11111 output_1.root progress_1.txt

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <limits>
#include <set>
#include <fstream>

// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TLine.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TString.h"

// Pythia
#include "Pythia8/Pythia.h"

// FastJet
#include "fastjet/ClusterSequence.hh"

using namespace Pythia8;
using namespace fastjet;

// -------------------- helpers --------------------

struct HadronCand {
  Vec4 p4;
  int charge;
  double frac; // |p_hadron| / |p_jet|
};

struct ThrustInfo {
  Vec4 axis;
  double T = 0.0;
};

static inline double norm3(const Vec4& a) {
  return std::sqrt(dot3(a, a));
}

static inline Vec4 unit3(const Vec4& a) {
  double n = norm3(a);
  if (n <= 0.0) return Vec4(0,0,1,0);
  return Vec4(a.px()/n, a.py()/n, a.pz()/n, 0);
}

// brute-force 50x50 scan
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

static TGraphAsymmErrors* makeBand(const TH1D* h, Color_t col, double alpha) {
  auto g = new TGraphAsymmErrors(h);
  g->SetFillColorAlpha(col, alpha);
  g->SetLineColor(col);
  g->SetLineWidth(2);
  g->SetMarkerColor(col);
  g->SetMarkerSize(0.0);
  return g;
}

static TGraphErrors* makePointErrors(const TH1D* h, Color_t col, int mstyle, double msize) {
  auto g = new TGraphErrors(h);
  for (int i = 0; i < g->GetN(); ++i)
    g->SetPointError(i, 0.0, g->GetErrorY(i));

  g->SetLineColor(col);
  g->SetLineWidth(2);
  g->SetMarkerColor(col);
  g->SetMarkerStyle(mstyle);
  g->SetMarkerSize(msize);
  return g;
}

static double histMaxWithErrors(const TH1D* h) {
  double out = 0.0;
  for (int b = 1; b <= h->GetNbinsX(); ++b)
    out = std::max(out, h->GetBinContent(b) + h->GetBinError(b));
  return out;
}

static std::pair<double,double> findRatioRange(const TH1D* hOS, const TH1D* hSS, double xMax) {
  double ymin = std::numeric_limits<double>::max();
  double ymax = -std::numeric_limits<double>::max();

  const int nb = hOS->GetNbinsX();
  for (int b = 1; b <= nb; ++b) {
    double x = hOS->GetBinCenter(b);
    if (x > xMax) continue;

    double os = hOS->GetBinContent(b);
    double ss = hSS->GetBinContent(b);
    if (os <= 0.0 || ss <= 0.0) continue;

    double r = os / ss;
    double e = hSS->GetBinContent(b) > 0.0 ? hOS->GetBinError(b) / ss : 0.0;
    if (os > 0.0 && ss > 0.0)
      e = r * std::sqrt(std::pow(hOS->GetBinError(b)/os,2) + std::pow(hSS->GetBinError(b)/ss,2));

    ymin = std::min(ymin, r - e);
    ymax = std::max(ymax, r + e);
  }

  if (!(ymin < ymax)) return {0.5, 1.5};

  ymin = std::min(ymin, 1.0);
  ymax = std::max(ymax, 1.0);

  double pad = 0.15 * (ymax - ymin);
  ymin = std::max(0.0, ymin - pad);
  ymax = ymax + pad;

  return {ymin, ymax};
}

static TH1D* makeNormHist(const TH1D* hIn, const char* newname) {
  auto h = (TH1D*)hIn->Clone(newname);
  h->SetDirectory(nullptr);
  h->Sumw2();
  double integral = h->Integral("width");
  if (integral > 0.0) h->Scale(1.0 / integral);
  return h;
}

static void drawCellOSSS(TPad* cell,
                         TH1D* hOSIn,
                         TH1D* hSSIn,
                         const char* tag,
                         const char* title,
                         double xMax,
                         bool normalised) {
  cell->cd();
  cell->SetMargin(0,0,0,0);

  auto pTop = new TPad(Form("pTop_%s", tag), "", 0, 0.30, 1, 1);
  auto pBot = new TPad(Form("pBot_%s", tag), "", 0, 0.00, 1, 0.30);

  pTop->SetBottomMargin(0.02);
  pTop->SetLeftMargin(0.13);
  pTop->SetRightMargin(0.03);

  pBot->SetTopMargin(0.02);
  pBot->SetBottomMargin(0.35);
  pBot->SetLeftMargin(0.13);
  pBot->SetRightMargin(0.03);

  pTop->Draw();
  pBot->Draw();

  TH1D* hOS = nullptr;
  TH1D* hSS = nullptr;

  if (normalised) {
    hOS = makeNormHist(hOSIn, Form("hOS_norm_%s", tag));
    hSS = makeNormHist(hSSIn, Form("hSS_norm_%s", tag));
  } else {
    hOS = (TH1D*)hOSIn->Clone(Form("hOS_%s", tag));
    hSS = (TH1D*)hSSIn->Clone(Form("hSS_%s", tag));
    hOS->SetDirectory(nullptr);
    hSS->SetDirectory(nullptr);
    hOS->Sumw2();
    hSS->Sumw2();
  }

  pTop->cd();

  auto frameTop = (TH1D*)hOS->Clone(Form("frameTop_%s", tag));
  frameTop->Reset("ICES");
  frameTop->SetTitle(title);
  frameTop->GetXaxis()->SetTitle("");
  frameTop->GetXaxis()->SetLabelSize(0.0);
  frameTop->GetXaxis()->SetTitleSize(0.0);
  frameTop->GetYaxis()->SetTitle(normalised ? "(1/N) dN/dq_{T} [GeV^{-1}]" : "Events");
  frameTop->GetYaxis()->SetTitleSize(0.060);
  frameTop->GetYaxis()->SetLabelSize(0.050);
  frameTop->GetYaxis()->SetTitleOffset(0.90);
  frameTop->GetXaxis()->SetRangeUser(0.0, xMax);
  frameTop->SetMinimum(0.0);
  frameTop->SetMaximum(1.18 * std::max(histMaxWithErrors(hOS), histMaxWithErrors(hSS)));
  frameTop->Draw();

  auto gOSBand = makeBand(hOS, kRed+1, 0.40);
  auto gSSBand = makeBand(hSS, kBlue+1, 0.40);
  auto gOSPts  = makePointErrors(hOS, kRed+1, 20, 0.30);
  auto gSSPts  = makePointErrors(hSS, kBlue+1, 24, 0.30);

  gOSBand->Draw("2 SAME");
  gSSBand->Draw("2 SAME");
  gOSPts->Draw("P E1 SAME");
  gSSPts->Draw("P E1 SAME");

  auto leg = new TLegend(0.64, 0.74, 0.92, 0.90);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(gOSBand, "OS", "pf");
  leg->AddEntry(gSSBand, "SS", "pf");
  leg->Draw();

  pBot->cd();

  auto hR = (TH1D*)hOS->Clone(Form("hRatio_%s", tag));
  hR->Sumw2();
  hR->Divide(hSS);

  auto frameBot = (TH1D*)hR->Clone(Form("frameBot_%s", tag));
  frameBot->Reset("ICES");
  frameBot->SetTitle("");
  frameBot->GetXaxis()->SetTitle("q_{T} [GeV]");
  frameBot->GetYaxis()->SetTitle("OS/SS");
  frameBot->GetYaxis()->SetNdivisions(505);
  frameBot->GetYaxis()->SetTitleSize(0.12);
  frameBot->GetYaxis()->SetLabelSize(0.10);
  frameBot->GetYaxis()->SetTitleOffset(0.45);
  frameBot->GetXaxis()->SetTitleSize(0.12);
  frameBot->GetXaxis()->SetLabelSize(0.10);
  frameBot->GetXaxis()->SetRangeUser(0.0, xMax);

  auto yr = findRatioRange(hOS, hSS, xMax);
  frameBot->SetMinimum(yr.first);
  frameBot->SetMaximum(yr.second);
  frameBot->Draw();

  auto one = new TLine(0.0, 1.0, xMax, 1.0);
  one->SetLineColor(kBlack);
  one->SetLineWidth(2);
  one->Draw("SAME");

  auto gRatio = makePointErrors(hR, kBlack, 20, 0.28);
  gRatio->Draw("P E1 SAME");
}

static TCanvas* makeSingleCutCanvas(TH1D* hOS,
                                    TH1D* hSS,
                                    int cut,
                                    double xMax,
                                    bool normalised) {
  TString cname = normalised
    ? TString::Format("c_qT_OSSS_pion_cut%d_norm", cut)
    : TString::Format("c_qT_OSSS_pion_cut%d_counts", cut);

  TString title = TString::Format("Highest pair   cut %d%%", cut);

  auto c = new TCanvas(cname, cname, 900, 700);
  drawCellOSSS((TPad*)c, hOS, hSS,
               Form("pion_cut%d_%s", cut, normalised ? "norm" : "counts"),
               title,
               xMax,
               normalised);
  return c;
}

// -------------------- main --------------------

int main(int argc, char* argv[]) {
  long long nEvents = 50000000;
  int seed = 12345;
  std::string outFileName = "output.root";
  std::string progressFileName = "";

  if (argc > 1) nEvents = std::atoll(argv[1]);
  if (argc > 2) seed = std::atoi(argv[2]);
  if (argc > 3) outFileName = argv[3];
  if (argc > 4) progressFileName = argv[4];

  auto writeProgress = [&](long long value) {
    if (!progressFileName.empty()) {
      std::ofstream pf(progressFileName);
      pf << value << "\n";
    }
  };

  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(5);

  std::vector<int> cutHist = {0,20,40,60};

  const int NBINS = 40;
  const double XMIN = 0.0;
  const double XMAX = 10.0;

  std::map<int,TH1D*> hOS_pion, hSS_pion;

  for (int c : cutHist) {
    hOS_pion[c] = new TH1D(Form("h_qT_highest_OS_pion_cut%d", c), "", NBINS, XMIN, XMAX);
    hSS_pion[c] = new TH1D(Form("h_qT_highest_SS_pion_cut%d", c), "", NBINS, XMIN, XMAX);
    hOS_pion[c]->Sumw2();
    hSS_pion[c]->Sumw2();
  }

  // Reduced selected-pair tree
  Long64_t tr_evt = 0;
  Int_t    tr_seed = seed;
  Double_t tr_thrust = 0.0;
  Double_t tr_thrust_nx = 0.0, tr_thrust_ny = 0.0, tr_thrust_nz = 0.0;
  Double_t tr_abs_dphi_jets = 0.0;
  Double_t tr_jet1_pt = 0.0, tr_jet2_pt = 0.0;
  Double_t tr_jet1_p = 0.0, tr_jet2_p = 0.0;
  Int_t    tr_nPionsJet1 = 0, tr_nPionsJet2 = 0;

  Double_t tr_qT = 0.0;
  Int_t    tr_charge1 = 0, tr_charge2 = 0;
  Int_t    tr_isOS = 0;
  Double_t tr_frac1 = 0.0, tr_frac2 = 0.0;
  Double_t tr_minFrac = 0.0, tr_maxFrac = 0.0;

  Double_t tr_p1_px = 0.0, tr_p1_py = 0.0, tr_p1_pz = 0.0, tr_p1_e = 0.0;
  Double_t tr_p2_px = 0.0, tr_p2_py = 0.0, tr_p2_pz = 0.0, tr_p2_e = 0.0;

  // Pythia setup
  Pythia pythia;
  pythia.readString("Beams:idA = -11");
  pythia.readString("Beams:idB = 11");
  pythia.readString("Beams:eCM = 91.2");
  pythia.readString("PDF:lepton = off");
  pythia.readString("HadronLevel:all = on");
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");

  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));

  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  if (!pythia.init()) {
    std::cerr << "Pythia initialization failed\n";
    return 1;
  }

  TFile fout(outFileName.c_str(), "RECREATE");
  auto tPionPairs = new TTree("tPionPairs", "Reduced selected-pair pion tree");

  tPionPairs->Branch("evt",           &tr_evt,           "evt/L");
  tPionPairs->Branch("seed",          &tr_seed,          "seed/I");
  tPionPairs->Branch("thrust",        &tr_thrust,        "thrust/D");
  tPionPairs->Branch("thrust_nx",     &tr_thrust_nx,     "thrust_nx/D");
  tPionPairs->Branch("thrust_ny",     &tr_thrust_ny,     "thrust_ny/D");
  tPionPairs->Branch("thrust_nz",     &tr_thrust_nz,     "thrust_nz/D");
  tPionPairs->Branch("abs_dphi_jets", &tr_abs_dphi_jets, "abs_dphi_jets/D");
  tPionPairs->Branch("jet1_pt",       &tr_jet1_pt,       "jet1_pt/D");
  tPionPairs->Branch("jet2_pt",       &tr_jet2_pt,       "jet2_pt/D");
  tPionPairs->Branch("jet1_p",        &tr_jet1_p,        "jet1_p/D");
  tPionPairs->Branch("jet2_p",        &tr_jet2_p,        "jet2_p/D");
  tPionPairs->Branch("nPionsJet1",    &tr_nPionsJet1,    "nPionsJet1/I");
  tPionPairs->Branch("nPionsJet2",    &tr_nPionsJet2,    "nPionsJet2/I");

  tPionPairs->Branch("qT",            &tr_qT,            "qT/D");
  tPionPairs->Branch("charge1",       &tr_charge1,       "charge1/I");
  tPionPairs->Branch("charge2",       &tr_charge2,       "charge2/I");
  tPionPairs->Branch("isOS",          &tr_isOS,          "isOS/I");
  tPionPairs->Branch("frac1",         &tr_frac1,         "frac1/D");
  tPionPairs->Branch("frac2",         &tr_frac2,         "frac2/D");
  tPionPairs->Branch("minFrac",       &tr_minFrac,       "minFrac/D");
  tPionPairs->Branch("maxFrac",       &tr_maxFrac,       "maxFrac/D");

  tPionPairs->Branch("p1_px",         &tr_p1_px,         "p1_px/D");
  tPionPairs->Branch("p1_py",         &tr_p1_py,         "p1_py/D");
  tPionPairs->Branch("p1_pz",         &tr_p1_pz,         "p1_pz/D");
  tPionPairs->Branch("p1_e",          &tr_p1_e,          "p1_e/D");
  tPionPairs->Branch("p2_px",         &tr_p2_px,         "p2_px/D");
  tPionPairs->Branch("p2_py",         &tr_p2_py,         "p2_py/D");
  tPionPairs->Branch("p2_pz",         &tr_p2_pz,         "p2_pz/D");
  tPionPairs->Branch("p2_e",          &tr_p2_e,          "p2_e/D");

  // FastJet baseline
  const double R = 0.4;
  const double jetPtMin = 5.0;
  const double thrustCut = 0.8;
  const double backToBackCut = 2.8;
  JetDefinition jetDef(antikt_algorithm, R);

  std::vector<PseudoJet> fjInputs;
  fjInputs.reserve(250);
  std::vector<Vec4> thrustInputs;
  thrustInputs.reserve(250);

  writeProgress(0);

  for (long long iEvt = 0; iEvt < nEvents; ++iEvt) {
    if ((iEvt + 1) % 250 == 0) {
      writeProgress(iEvt + 1);
    }

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

    bool isOS = (best1.charge * best2.charge < 0);

    // Fill reduced tree
    tr_evt           = iEvt;
    tr_seed          = seed;
    tr_thrust        = thr.T;
    tr_thrust_nx     = nT.px();
    tr_thrust_ny     = nT.py();
    tr_thrust_nz     = nT.pz();
    tr_abs_dphi_jets = std::fabs(dphi_jets);
    tr_jet1_pt       = jets[0].pt();
    tr_jet2_pt       = jets[1].pt();
    tr_jet1_p        = jets[0].modp();
    tr_jet2_p        = jets[1].modp();
    tr_nPionsJet1    = (int)pions1.size();
    tr_nPionsJet2    = (int)pions2.size();

    tr_qT            = bestQT;
    tr_charge1       = best1.charge;
    tr_charge2       = best2.charge;
    tr_isOS          = isOS ? 1 : 0;
    tr_frac1         = best1.frac;
    tr_frac2         = best2.frac;
    tr_minFrac       = std::min(best1.frac, best2.frac);
    tr_maxFrac       = std::max(best1.frac, best2.frac);

    tr_p1_px         = best1.p4.px();
    tr_p1_py         = best1.p4.py();
    tr_p1_pz         = best1.p4.pz();
    tr_p1_e          = best1.p4.e();

    tr_p2_px         = best2.p4.px();
    tr_p2_py         = best2.p4.py();
    tr_p2_pz         = best2.p4.pz();
    tr_p2_e          = best2.p4.e();

    tPionPairs->Fill();

    // Keep the original default histogram filling
    for (int c : cutHist) {
      double thrCut = c / 100.0;
      if (best1.frac >= thrCut && best2.frac >= thrCut) {
        if (isOS) hOS_pion[c]->Fill(bestQT);
        else      hSS_pion[c]->Fill(bestQT);
      }
    }
  }

  writeProgress(nEvents);

  auto cPionCounts = new TCanvas("c_qT_OSSS_4cuts_pion_counts", "Pion OS vs SS counts", 1400, 900);
  cPionCounts->Divide(2,2,0.01,0.01);
  for (int i = 0; i < 4; ++i) {
    int c = cutHist[i];
    cPionCounts->cd(i+1);
    drawCellOSSS((TPad*)gPad, hOS_pion[c], hSS_pion[c],
                 Form("pion_cut%d_counts", c),
                 Form("Highest pair   cut %d%%", c),
                 XMAX,
                 false);
  }

  auto cPionNorm = new TCanvas("c_qT_OSSS_4cuts_pion_norm", "Pion OS vs SS normalised", 1400, 900);
  cPionNorm->Divide(2,2,0.01,0.01);
  for (int i = 0; i < 4; ++i) {
    int c = cutHist[i];
    cPionNorm->cd(i+1);
    drawCellOSSS((TPad*)gPad, hOS_pion[c], hSS_pion[c],
                 Form("pion_cut%d_norm", c),
                 Form("Highest pair   cut %d%%", c),
                 XMAX,
                 true);
  }

  std::map<int,TCanvas*> cSingleCounts, cSingleNorm;
  for (int c : cutHist) {
    cSingleCounts[c] = makeSingleCutCanvas(hOS_pion[c], hSS_pion[c], c, XMAX, false);
    cSingleNorm[c]   = makeSingleCutCanvas(hOS_pion[c], hSS_pion[c], c, XMAX, true);
  }

  fout.cd();

  tPionPairs->Write();

  for (auto& kv : hOS_pion) kv.second->Write();
  for (auto& kv : hSS_pion) kv.second->Write();

  cPionCounts->Write();
  cPionNorm->Write();

  for (auto& kv : cSingleCounts) kv.second->Write();
  for (auto& kv : cSingleNorm)   kv.second->Write();

  fout.Close();
  std::cout << "Saved " << outFileName << " for viewing in TBrowser\n";
  return 0;
}