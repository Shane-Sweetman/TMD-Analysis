// Semester-2/Pythia1.cc
// Produces output.root with:
//   - c_qT_OSSS_4cuts_pion : 2x2 of pion qT plots (cuts 0,20,40,60), OS+SS + ratio OS/SS
//   - c_ratio_vs_cut_pion  : pion OS/SS ratio vs cut (0..80), points only
//   - c_qT_OSSS_4cuts_kaon : 2x2 of kaon qT plots (cuts 0,20,40,60), OS+SS + ratio OS/SS
//   - c_ratio_vs_cut_kaon  : kaon OS/SS ratio vs cut (0..80), points only
//   - c_shapes_pion_40_60  : pion event-shape histograms for cuts 40% and 60%
//   - c_shapes_kaon_40_60  : kaon event-shape histograms for cuts 40% and 60%
//
// Run: ./TMD [NEVENTS] [SEED]
// Example: ./TMD 5000000 12345

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <limits>

// ROOT
#include "TFile.h"
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

static ThrustInfo computeThrustInfo(const std::vector<Vec4>& ps) {
  ThrustInfo out;
  if (ps.empty()) {
    out.axis = Vec4(0,0,1,0);
    out.T = 0.0;
    return out;
  }

  int imax = 0;
  double pmax = 0.0;
  for (int i = 0; i < (int)ps.size(); ++i) {
    double p = norm3(ps[i]);
    if (p > pmax) {
      pmax = p;
      imax = i;
    }
  }

  Vec4 n = unit3(ps[imax]);

  for (int it = 0; it < 100; ++it) {
    Vec4 sum(0,0,0,0);
    for (const auto& p : ps) {
      double s = (dot3(p, n) >= 0.0) ? 1.0 : -1.0;
      sum += s * p;
    }
    Vec4 nNew = unit3(sum);
    Vec4 diff(nNew.px()-n.px(), nNew.py()-n.py(), nNew.pz()-n.pz(), 0);
    if (norm3(diff) < 1e-6) {
      n = nNew;
      break;
    }
    n = nNew;
  }

  double num = 0.0;
  double den = 0.0;
  for (const auto& p : ps) {
    num += std::fabs(dot3(p, n));
    den += norm3(p);
  }

  out.axis = n;
  out.T = (den > 0.0) ? num / den : 0.0;
  return out;
}

static double computeSphericity(const std::vector<Vec4>& ps) {
  if (ps.empty()) return 0.0;

  double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
  double denom = 0.0;

  for (const auto& p : ps) {
    const double px = p.px();
    const double py = p.py();
    const double pz = p.pz();
    const double p2 = px*px + py*py + pz*pz;
    denom += p2;

    M[0][0] += px*px; M[0][1] += px*py; M[0][2] += px*pz;
    M[1][0] += py*px; M[1][1] += py*py; M[1][2] += py*pz;
    M[2][0] += pz*px; M[2][1] += pz*py; M[2][2] += pz*pz;
  }

  if (denom <= 0.0) return 0.0;

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      M[i][j] /= denom;

  double v[3] = {1.0, 1.0, 1.0};
  {
    double n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] /= n; v[1] /= n; v[2] /= n;
  }

  for (int it = 0; it < 100; ++it) {
    double w[3] = {
      M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
      M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
      M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]
    };

    double wn = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    if (wn <= 0.0) break;

    w[0] /= wn; w[1] /= wn; w[2] /= wn;

    double diff = std::sqrt((w[0]-v[0])*(w[0]-v[0]) + (w[1]-v[1])*(w[1]-v[1]) + (w[2]-v[2])*(w[2]-v[2]));
    v[0] = w[0]; v[1] = w[1]; v[2] = w[2];
    if (diff < 1e-10) break;
  }

  const double lambdaMax =
      v[0]*(M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2]) +
      v[1]*(M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2]) +
      v[2]*(M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]);

  double S = 1.5 * (1.0 - lambdaMax);
  if (S < 0.0) S = 0.0;
  if (S > 1.0) S = 1.0;
  return S;
}

static double qT_pair(const Vec4& p1, const Vec4& p2, const Vec4& n_unit) {
  Vec4 q = p1 + p2;
  double qpar = q.px()*n_unit.px() + q.py()*n_unit.py() + q.pz()*n_unit.pz();
  Vec4 qperp(q.px() - qpar*n_unit.px(),
             q.py() - qpar*n_unit.py(),
             q.pz() - qpar*n_unit.pz(), 0);
  return norm3(qperp);
}

static inline double dR(const PseudoJet& a, const PseudoJet& b) {
  double dphi = std::fabs(a.phi_std() - b.phi_std());
  if (dphi > M_PI) dphi = 2*M_PI - dphi;
  double deta = a.eta() - b.eta();
  return std::sqrt(deta*deta + dphi*dphi);
}

// darker filled band
static TGraphAsymmErrors* makeBand(const TH1D* h, Color_t col, double alpha) {
  auto g = new TGraphAsymmErrors(h);
  g->SetFillColorAlpha(col, alpha);
  g->SetLineColor(col);
  g->SetLineWidth(2);
  g->SetMarkerColor(col);
  g->SetMarkerSize(0.0);
  return g;
}

// smaller points + clearer error bars
static TGraphErrors* makePointErrors(const TH1D* h, Color_t col, int mstyle, double msize) {
  auto g = new TGraphErrors(h);
  for (int i = 0; i < g->GetN(); ++i) {
    g->SetPointError(i, 0.0, g->GetErrorY(i));
  }
  g->SetLineColor(col);
  g->SetLineWidth(2);
  g->SetMarkerColor(col);
  g->SetMarkerStyle(mstyle);
  g->SetMarkerSize(msize);
  return g;
}

static double histMaxWithErrors(const TH1D* h) {
  double out = 0.0;
  for (int b = 1; b <= h->GetNbinsX(); ++b) {
    out = std::max(out, h->GetBinContent(b) + h->GetBinError(b));
  }
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
    double e = r * std::sqrt(1.0/os + 1.0/ss);

    ymin = std::min(ymin, r - e);
    ymax = std::max(ymax, r + e);
  }

  if (!(ymin < ymax)) return {0.5, 1.5};

  ymin = std::min(ymin, 1.0);
  ymax = std::max(ymax, 1.0);

  double pad = 0.15 * (ymax - ymin);
  ymin = std::max(0.0, ymin - pad);
  ymax = ymax + pad;

  if (ymax - ymin < 0.4) {
    ymin = 0.8;
    ymax = 1.2;
  }

  return {ymin, ymax};
}

// One cell (top: OS+SS, bottom: OS/SS)
static void drawCellOSSS(TPad* cell, TH1D* hOS, TH1D* hSS, const char* tag, const char* title, double xMax) {
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

  hOS->Sumw2();
  hSS->Sumw2();

  // ---- top pad
  pTop->cd();

  auto frameTop = (TH1D*)hOS->Clone(Form("frameTop_%s", tag));
  frameTop->Reset("ICES");
  frameTop->SetTitle(title);
  frameTop->GetXaxis()->SetTitle("");
  frameTop->GetXaxis()->SetLabelSize(0.0);
  frameTop->GetXaxis()->SetTitleSize(0.0);
  frameTop->GetYaxis()->SetTitle("Events");
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

  // ---- bottom pad: OS/SS
  pBot->cd();

  auto hR = (TH1D*)hOS->Clone(Form("hRatio_%s", tag));
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

static TGraphErrors* makeRatioGraph(const std::vector<int>& cutScan,
                                    const std::map<int,long long>& nOS,
                                    const std::map<int,long long>& nSS,
                                    const char* name) {
  auto g = new TGraphErrors((int)cutScan.size());
  g->SetName(name);

  for (int i = 0; i < (int)cutScan.size(); ++i) {
    int c = cutScan[i];
    double NOS = (double)nOS.at(c);
    double NSS = (double)nSS.at(c);

    double r = 0.0;
    double e = 0.0;
    if (NOS > 0.0 && NSS > 0.0) {
      r = NOS / NSS;
      e = r * std::sqrt(1.0/NOS + 1.0/NSS);
    }

    g->SetPoint(i, c, r);
    g->SetPointError(i, 0.0, e);
  }

  g->SetMarkerStyle(20);
  g->SetMarkerSize(0.55);
  g->SetMarkerColor(kBlack);
  g->SetLineColor(kBlack);
  g->SetLineWidth(0);
  return g;
}

static TCanvas* makeRatioVsCutCanvas(TGraphErrors* g, const char* cname, const char* title) {
  auto c = new TCanvas(cname, cname, 900, 700);

  double ymax = 1.0;
  for (int i = 0; i < g->GetN(); ++i) {
    double x = 0.0, y = 0.0;
    g->GetPoint(i, x, y);
    ymax = std::max(ymax, y + g->GetErrorY(i));
  }

  auto frame = new TH1D(Form("frame_%s", cname), title, 100, 0.0, 100.0);
  frame->SetDirectory(nullptr);
  frame->SetStats(0);
  frame->SetMinimum(0.0);
  frame->SetMaximum(1.15 * ymax);
  frame->GetXaxis()->SetTitle("cut [%]");
  frame->GetYaxis()->SetTitle("N_{OS}/N_{SS}");
  frame->GetXaxis()->SetTitleSize(0.050);
  frame->GetYaxis()->SetTitleSize(0.050);
  frame->GetXaxis()->SetLabelSize(0.042);
  frame->GetYaxis()->SetLabelSize(0.042);
  frame->GetYaxis()->SetTitleOffset(1.05);
  frame->Draw();

  g->Draw("P E1 SAME");
  return c;
}

static TCanvas* makeShapeCanvas(const char* cname,
                                const char* species,
                                TH1D* hMult40, TH1D* hThr40, TH1D* hSph40,
                                TH1D* hMult60, TH1D* hThr60, TH1D* hSph60) {
  gStyle->SetOptStat(1110);

  auto c = new TCanvas(cname, cname, 1500, 900);
  c->Divide(3,2,0.01,0.01);

  auto drawHist = [&](int padIdx, TH1D* h, const char* title, const char* xtitle) {
    c->cd(padIdx);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    gPad->SetBottomMargin(0.12);
    gPad->SetTopMargin(0.10);
    h->SetTitle(title);
    h->GetXaxis()->SetTitle(xtitle);
    h->GetYaxis()->SetTitle("Entries");
    h->GetXaxis()->SetTitleSize(0.050);
    h->GetYaxis()->SetTitleSize(0.050);
    h->GetXaxis()->SetLabelSize(0.042);
    h->GetYaxis()->SetLabelSize(0.042);
    h->SetLineColor(kBlue+1);
    h->SetLineWidth(2);
    h->Draw("hist");
  };

  drawHist(1, hMult40, Form("%s jet multiplicity, cut 40%%", species), "N_{jets}");
  drawHist(2, hThr40,  Form("%s thrust, cut 40%%", species), "T");
  drawHist(3, hSph40,  Form("%s sphericity, cut 40%%", species), "S");
  drawHist(4, hMult60, Form("%s jet multiplicity, cut 60%%", species), "N_{jets}");
  drawHist(5, hThr60,  Form("%s thrust, cut 60%%", species), "T");
  drawHist(6, hSph60,  Form("%s sphericity, cut 60%%", species), "S");

  gStyle->SetOptStat(0);
  return c;
}

static void processSpecies(const std::vector<int>& cutHist,
                           const std::vector<int>& cutScan,
                           const std::vector<HadronCand>& jet1,
                           const std::vector<HadronCand>& jet2,
                           const Vec4& thrustAxis,
                           double thrustVal,
                           double sphericityVal,
                           int jetMultiplicity,
                           std::map<int,TH1D*>& hOS,
                           std::map<int,TH1D*>& hSS,
                           std::map<int,long long>& nOS,
                           std::map<int,long long>& nSS,
                           std::map<int,TH1D*>& hJetMult,
                           std::map<int,TH1D*>& hThrust,
                           std::map<int,TH1D*>& hSphericity) {
  if (jet1.empty() || jet2.empty()) return;

  std::vector<HadronCand> a = jet1;
  std::vector<HadronCand> b = jet2;

  auto byP = [](const HadronCand& x, const HadronCand& y) {
    return x.p4.pAbs() > y.p4.pAbs();
  };

  std::sort(a.begin(), a.end(), byP);
  std::sort(b.begin(), b.end(), byP);

  const int MAX_HADRONS_PER_JET = 50;
  if ((int)a.size() > MAX_HADRONS_PER_JET) a.resize(MAX_HADRONS_PER_JET);
  if ((int)b.size() > MAX_HADRONS_PER_JET) b.resize(MAX_HADRONS_PER_JET);

  double bestQT = -1.0;
  HadronCand best1{}, best2{};

  for (const auto& h1 : a) {
    for (const auto& h2 : b) {
      double qt = qT_pair(h1.p4, h2.p4, thrustAxis);
      if (qt > bestQT) {
        bestQT = qt;
        best1 = h1;
        best2 = h2;
      }
    }
  }

  if (bestQT < 0.0) return;

  bool isOS = (best1.charge * best2.charge < 0);

  for (int c : cutHist) {
    double thr = c / 100.0;
    if (best1.frac >= thr && best2.frac >= thr) {
      if (isOS) hOS[c]->Fill(bestQT);
      else      hSS[c]->Fill(bestQT);
    }
  }

  for (int c : cutScan) {
    double thr = c / 100.0;
    if (best1.frac >= thr && best2.frac >= thr) {
      if (isOS) nOS[c]++;
      else      nSS[c]++;

      if (hJetMult.count(c))    hJetMult[c]->Fill(jetMultiplicity);
      if (hThrust.count(c))     hThrust[c]->Fill(thrustVal);
      if (hSphericity.count(c)) hSphericity[c]->Fill(sphericityVal);
    }
  }
}

int main(int argc, char* argv[]) {
  long long nEvents = 5000000;
  int seed = 12345;
  if (argc > 1) nEvents = std::atoll(argv[1]);
  if (argc > 2) seed = std::atoi(argv[2]);

  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(5);

  // Cuts
  std::vector<int> cutHist = {0,20,40,60};
  std::vector<int> cutScan;
  for (int c = 0; c <= 80; c += 5) cutScan.push_back(c);
  std::vector<int> shapeCuts = {40,60};

  // qT hist settings: doubled bins over 0-5
  const int NBINS = 20;
  const double XMIN = 0.0;
  const double XMAX = 5.0;

  // Shape hist settings
  const int MULT_BINS = 20;
  const double MULT_MIN = 0.0;
  const double MULT_MAX = 20.0;

  const int THR_BINS = 50;
  const double THR_MIN = 0.0;
  const double THR_MAX = 1.0;

  const int SPH_BINS = 50;
  const double SPH_MIN = 0.0;
  const double SPH_MAX = 1.0;

  // Pion histograms
  std::map<int,TH1D*> hOS_pion, hSS_pion;
  std::map<int,TH1D*> hJetMult_pion, hThrust_pion, hSphericity_pion;

  // Kaon histograms
  std::map<int,TH1D*> hOS_kaon, hSS_kaon;
  std::map<int,TH1D*> hJetMult_kaon, hThrust_kaon, hSphericity_kaon;

  for (int c : cutHist) {
    hOS_pion[c] = new TH1D(Form("h_qT_highest_OS_pion_cut%d", c), "", NBINS, XMIN, XMAX);
    hSS_pion[c] = new TH1D(Form("h_qT_highest_SS_pion_cut%d", c), "", NBINS, XMIN, XMAX);
    hOS_kaon[c] = new TH1D(Form("h_qT_highest_OS_kaon_cut%d", c), "", NBINS, XMIN, XMAX);
    hSS_kaon[c] = new TH1D(Form("h_qT_highest_SS_kaon_cut%d", c), "", NBINS, XMIN, XMAX);

    hOS_pion[c]->Sumw2(); hSS_pion[c]->Sumw2();
    hOS_kaon[c]->Sumw2(); hSS_kaon[c]->Sumw2();
  }

  for (int c : shapeCuts) {
    hJetMult_pion[c]   = new TH1D(Form("h_jetMult_pion_cut%d", c), "", MULT_BINS, MULT_MIN, MULT_MAX);
    hThrust_pion[c]    = new TH1D(Form("h_thrust_pion_cut%d", c), "", THR_BINS, THR_MIN, THR_MAX);
    hSphericity_pion[c]= new TH1D(Form("h_sphericity_pion_cut%d", c), "", SPH_BINS, SPH_MIN, SPH_MAX);

    hJetMult_kaon[c]   = new TH1D(Form("h_jetMult_kaon_cut%d", c), "", MULT_BINS, MULT_MIN, MULT_MAX);
    hThrust_kaon[c]    = new TH1D(Form("h_thrust_kaon_cut%d", c), "", THR_BINS, THR_MIN, THR_MAX);
    hSphericity_kaon[c]= new TH1D(Form("h_sphericity_kaon_cut%d", c), "", SPH_BINS, SPH_MIN, SPH_MAX);
  }

  std::map<int,long long> nOS_pion, nSS_pion, nOS_kaon, nSS_kaon;
  for (int c : cutScan) {
    nOS_pion[c] = 0; nSS_pion[c] = 0;
    nOS_kaon[c] = 0; nSS_kaon[c] = 0;
  }

  // Pythia setup
  Pythia pythia;
  pythia.readString("Beams:idA = 11");
  pythia.readString("Beams:idB = -11");
  pythia.readString("Beams:eCM = 91.1876");
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");

  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));

  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  pythia.init();

  // FastJet
  const double R = 0.4;
  const double JET_PTMIN_MULT = 1.0; // for jet multiplicity only
  JetDefinition jetDef(antikt_algorithm, R);

  std::vector<PseudoJet> fjInputs;
  fjInputs.reserve(250);
  std::vector<Vec4> thrustInputs;
  thrustInputs.reserve(250);

  for (long long iEvt = 0; iEvt < nEvents; ++iEvt) {
    if (!pythia.next()) continue;

    fjInputs.clear();
    thrustInputs.clear();

    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (!p.isFinal()) continue;
      if (!p.isVisible()) continue;

      Vec4 v = p.p();
      fjInputs.emplace_back(v.px(), v.py(), v.pz(), v.e());
      thrustInputs.push_back(Vec4(v.px(), v.py(), v.pz(), 0));
    }

    if (fjInputs.size() < 2) continue;

    ThrustInfo thr = computeThrustInfo(thrustInputs);
    Vec4 nT = thr.axis;
    double thrustVal = thr.T;
    double sphericityVal = computeSphericity(thrustInputs);

    ClusterSequence cs(fjInputs, jetDef);
    std::vector<PseudoJet> jets = sorted_by_E(cs.inclusive_jets());
    if (jets.size() < 2) continue;

    std::vector<PseudoJet> jetsForMult = sorted_by_E(cs.inclusive_jets(JET_PTMIN_MULT));
    int jetMultiplicity = (int)jetsForMult.size();

    PseudoJet j1 = jets[0];
    PseudoJet j2 = jets[1];

    // mild back-to-back
    {
      Vec4 v1(j1.px(), j1.py(), j1.pz(), 0);
      Vec4 v2(j2.px(), j2.py(), j2.pz(), 0);
      double cang = dot3(v1, v2) / (norm3(v1)*norm3(v2) + 1e-12);
      if (cang > -0.5) continue;
    }

    std::vector<HadronCand> pions1, pions2;
    std::vector<HadronCand> kaons1, kaons2;
    pions1.reserve(60); pions2.reserve(60);
    kaons1.reserve(60); kaons2.reserve(60);

    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (!p.isFinal()) continue;
      if (!p.isVisible()) continue;

      Vec4 v = p.p();
      PseudoJet pj(v.px(), v.py(), v.pz(), v.e());

      double dr1 = dR(pj, j1);
      double dr2 = dR(pj, j2);

      bool inJet1 = (dr1 < dr2 && dr1 < R);
      bool inJet2 = (dr2 <= dr1 && dr2 < R);

      int id = p.id();

      if (id == 211 || id == -211) {
        int charge = (id == 211) ? +1 : -1;
        if (inJet1) {
          double frac = (j1.modp() > 1e-12) ? (pj.modp() / j1.modp()) : 0.0;
          pions1.push_back({v, charge, frac});
        } else if (inJet2) {
          double frac = (j2.modp() > 1e-12) ? (pj.modp() / j2.modp()) : 0.0;
          pions2.push_back({v, charge, frac});
        }
      }

      if (id == 321 || id == -321) {
        int charge = (id == 321) ? +1 : -1;
        if (inJet1) {
          double frac = (j1.modp() > 1e-12) ? (pj.modp() / j1.modp()) : 0.0;
          kaons1.push_back({v, charge, frac});
        } else if (inJet2) {
          double frac = (j2.modp() > 1e-12) ? (pj.modp() / j2.modp()) : 0.0;
          kaons2.push_back({v, charge, frac});
        }
      }
    }

    processSpecies(cutHist, cutScan, pions1, pions2, nT, thrustVal, sphericityVal, jetMultiplicity,
                   hOS_pion, hSS_pion, nOS_pion, nSS_pion,
                   hJetMult_pion, hThrust_pion, hSphericity_pion);

    processSpecies(cutHist, cutScan, kaons1, kaons2, nT, thrustVal, sphericityVal, jetMultiplicity,
                   hOS_kaon, hSS_kaon, nOS_kaon, nSS_kaon,
                   hJetMult_kaon, hThrust_kaon, hSphericity_kaon);
  }

  // Ratio-vs-cut graphs
  auto gRatioPion = makeRatioGraph(cutScan, nOS_pion, nSS_pion, "g_ratio_vs_cut_pion");
  auto gRatioKaon = makeRatioGraph(cutScan, nOS_kaon, nSS_kaon, "g_ratio_vs_cut_kaon");

  // 2x2 pion qT canvas
  auto cPion = new TCanvas("c_qT_OSSS_4cuts_pion", "Pion OS vs SS qT for cuts", 1400, 900);
  cPion->Divide(2,2,0.01,0.01);
  for (int i = 0; i < 4; ++i) {
    int c = cutHist[i];
    cPion->cd(i+1);
    drawCellOSSS((TPad*)gPad, hOS_pion[c], hSS_pion[c],
                 Form("pion_cut%d", c),
                 Form("Highest pair   cut %d%%", c),
                 XMAX);
  }

  // 2x2 kaon qT canvas
  auto cKaon = new TCanvas("c_qT_OSSS_4cuts_kaon", "Kaon OS vs SS qT for cuts", 1400, 900);
  cKaon->Divide(2,2,0.01,0.01);
  for (int i = 0; i < 4; ++i) {
    int c = cutHist[i];
    cKaon->cd(i+1);
    drawCellOSSS((TPad*)gPad, hOS_kaon[c], hSS_kaon[c],
                 Form("kaon_cut%d", c),
                 Form("Highest pair   cut %d%%", c),
                 XMAX);
  }

  // ratio-vs-cut canvases
  auto cRatioPion = makeRatioVsCutCanvas(
    gRatioPion,
    "c_ratio_vs_cut_pion",
    "OS/SS ratio vs pion p_{T} fraction cut;cut [%];N_{OS}/N_{SS}"
  );

  auto cRatioKaon = makeRatioVsCutCanvas(
    gRatioKaon,
    "c_ratio_vs_cut_kaon",
    "OS/SS ratio vs kaon p_{T} fraction cut;cut [%];N_{OS}/N_{SS}"
  );

  // shape canvases
  auto cShapesPion = makeShapeCanvas(
    "c_shapes_pion_40_60", "Pion",
    hJetMult_pion[40], hThrust_pion[40], hSphericity_pion[40],
    hJetMult_pion[60], hThrust_pion[60], hSphericity_pion[60]
  );

  auto cShapesKaon = makeShapeCanvas(
    "c_shapes_kaon_40_60", "Kaon",
    hJetMult_kaon[40], hThrust_kaon[40], hSphericity_kaon[40],
    hJetMult_kaon[60], hThrust_kaon[60], hSphericity_kaon[60]
  );

  // write output
  TFile fout("output.root", "RECREATE");

  for (auto& kv : hOS_pion) kv.second->Write();
  for (auto& kv : hSS_pion) kv.second->Write();
  for (auto& kv : hOS_kaon) kv.second->Write();
  for (auto& kv : hSS_kaon) kv.second->Write();

  for (auto& kv : hJetMult_pion) kv.second->Write();
  for (auto& kv : hThrust_pion) kv.second->Write();
  for (auto& kv : hSphericity_pion) kv.second->Write();

  for (auto& kv : hJetMult_kaon) kv.second->Write();
  for (auto& kv : hThrust_kaon) kv.second->Write();
  for (auto& kv : hSphericity_kaon) kv.second->Write();

  gRatioPion->Write();
  gRatioKaon->Write();

  cPion->Write();
  cKaon->Write();
  cRatioPion->Write();
  cRatioKaon->Write();
  cShapesPion->Write();
  cShapesKaon->Write();

  fout.Close();

  // save images
  cPion->SaveAs("c_qT_OSSS_4cuts_pion.pdf");
  cPion->SaveAs("c_qT_OSSS_4cuts_pion.png");

  cKaon->SaveAs("c_qT_OSSS_4cuts_kaon.pdf");
  cKaon->SaveAs("c_qT_OSSS_4cuts_kaon.png");

  cRatioPion->SaveAs("c_ratio_vs_cut_pion.pdf");
  cRatioPion->SaveAs("c_ratio_vs_cut_pion.png");

  cRatioKaon->SaveAs("c_ratio_vs_cut_kaon.pdf");
  cRatioKaon->SaveAs("c_ratio_vs_cut_kaon.png");

  cShapesPion->SaveAs("c_shapes_pion_40_60.pdf");
  cShapesPion->SaveAs("c_shapes_pion_40_60.png");

  cShapesKaon->SaveAs("c_shapes_kaon_40_60.pdf");
  cShapesKaon->SaveAs("c_shapes_kaon_40_60.png");

  std::cout << "Saved output.root\n";
  return 0;
}