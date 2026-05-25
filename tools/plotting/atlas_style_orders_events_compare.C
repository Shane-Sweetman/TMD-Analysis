#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TLine.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TDatime.h"
#include "TString.h"

struct TheoryData {
  std::vector<double> qT;
  std::vector<double> os;
  std::vector<double> ss;
};

struct BinnedSeries {
  std::vector<double> x1, x2, xc, xerr;
  std::vector<double> y, ey;
};

bool loadTheoryData(const std::string& fname, TheoryData& th) {
  std::ifstream fin(fname);
  if (!fin.is_open()) {
    std::cerr << "Could not open theory file: " << fname << "\n";
    return false;
  }
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    double q, os, ss;
    if (iss >> q >> os >> ss) {
      th.qT.push_back(q);
      th.os.push_back(os);
      th.ss.push_back(ss);
    }
  }
  return !th.qT.empty();
}

double interpTheory(const TheoryData& th, double x, bool useOS) {
  const std::vector<double>& y = useOS ? th.os : th.ss;
  if (th.qT.empty()) return 0.0;
  if (x <= th.qT.front()) return y.front();
  if (x >= th.qT.back()) return y.back();
  for (size_t i = 0; i + 1 < th.qT.size(); ++i) {
    if (x >= th.qT[i] && x <= th.qT[i + 1]) {
      double dx = th.qT[i + 1] - th.qT[i];
      double t = (dx > 0.0) ? (x - th.qT[i]) / dx : 0.0;
      return y[i] + t * (y[i + 1] - y[i]);
    }
  }
  return y.back();
}

double integrateTheory(const TheoryData& th, double x1, double x2, bool useOS) {
  if (x2 <= x1) return 0.0;
  std::vector<double> xs;
  xs.push_back(x1);
  for (double q : th.qT) if (q > x1 && q < x2) xs.push_back(q);
  xs.push_back(x2);
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  double integral = 0.0;
  for (size_t i = 0; i + 1 < xs.size(); ++i) {
    double xa = xs[i], xb = xs[i + 1];
    double ya = interpTheory(th, xa, useOS);
    double yb = interpTheory(th, xb, useOS);
    integral += 0.5 * (ya + yb) * (xb - xa);
  }
  return integral;
}

double theoryPeakOS(const TheoryData& th) {
  return th.os.empty() ? 0.0 : *std::max_element(th.os.begin(), th.os.end());
}

double histPeak(const TH1D* h, double xMax) {
  double out = 0.0;
  for (int b = 1; b <= h->GetNbinsX(); ++b) {
    if (h->GetBinCenter(b) > xMax) continue;
    out = std::max(out, h->GetBinContent(b));
  }
  return out;
}

bool buildCutHistFromTree(TFile* f, int cut, TH1D*& hOS, TH1D*& hSS) {
  TTree* t = dynamic_cast<TTree*>(f->Get("tPionPairs"));
  if (!t) return false;

  double qT = 0.0;
  double minFrac = 0.0;
  int isOS = 0;
  t->SetBranchAddress("qT", &qT);
  t->SetBranchAddress("minFrac", &minFrac);
  t->SetBranchAddress("isOS", &isOS);

  hOS = new TH1D("hOS_cut60_rebuilt", "", 40, 0.0, 10.0);
  hSS = new TH1D("hSS_cut60_rebuilt", "", 40, 0.0, 10.0);
  hOS->SetDirectory(nullptr);
  hSS->SetDirectory(nullptr);
  hOS->Sumw2();
  hSS->Sumw2();

  const double thr = cut / 100.0;
  Long64_t n = t->GetEntries();
  for (Long64_t i = 0; i < n; ++i) {
    t->GetEntry(i);
    if (minFrac < thr) continue;
    if (isOS) hOS->Fill(qT);
    else      hSS->Fill(qT);
  }
  return true;
}

bool loadCut60Hists(const char* pythiaFile, TH1D*& hOS, TH1D*& hSS) {
  TFile* f = TFile::Open(pythiaFile, "READ");
  if (!f || f->IsZombie()) {
    std::cerr << "Could not open " << pythiaFile << "\n";
    return false;
  }
  auto hRawOS = dynamic_cast<TH1D*>(f->Get("h_qT_highest_OS_pion_cut60"));
  auto hRawSS = dynamic_cast<TH1D*>(f->Get("h_qT_highest_SS_pion_cut60"));
  if (hRawOS && hRawSS) {
    hOS = (TH1D*)hRawOS->Clone("hOS60_local");
    hSS = (TH1D*)hRawSS->Clone("hSS60_local");
    hOS->SetDirectory(nullptr);
    hSS->SetDirectory(nullptr);
    f->Close();
    return true;
  }
  std::cout << "Cut60 histograms not found directly; rebuilding from tPionPairs.\n";
  bool ok = buildCutHistFromTree(f, 60, hOS, hSS);
  f->Close();
  return ok;
}

BinnedSeries makeBinnedData(const TH1D* h, const std::vector<double>& edges) {
  BinnedSeries out;
  const double eps = 1e-9;
  for (size_t i = 0; i + 1 < edges.size(); ++i) {
    double x1 = edges[i], x2 = edges[i+1];
    double y = 0.0, ey2 = 0.0;
    for (int b = 1; b <= h->GetNbinsX(); ++b) {
      double lo = h->GetXaxis()->GetBinLowEdge(b);
      double hi = h->GetXaxis()->GetBinUpEdge(b);
      if (lo >= x1 - eps && hi <= x2 + eps) {
        y += h->GetBinContent(b);
        ey2 += std::pow(h->GetBinError(b), 2);
      }
    }
    out.x1.push_back(x1);
    out.x2.push_back(x2);
    out.xc.push_back(0.5 * (x1 + x2));
    out.xerr.push_back(0.5 * (x2 - x1));
    out.y.push_back(y);
    out.ey.push_back(std::sqrt(ey2));
  }
  return out;
}

BinnedSeries makeBinnedTheory(const TheoryData& th, const std::vector<double>& edges, bool useOS, double scale) {
  BinnedSeries out;
  for (size_t i = 0; i + 1 < edges.size(); ++i) {
    double x1 = edges[i], x2 = edges[i+1];
    double y = scale * integrateTheory(th, x1, x2, useOS);
    out.x1.push_back(x1);
    out.x2.push_back(x2);
    out.xc.push_back(0.5 * (x1 + x2));
    out.xerr.push_back(0.5 * (x2 - x1));
    out.y.push_back(y);
    out.ey.push_back(0.0);
  }
  return out;
}

TGraphErrors* makeDataGraph(const BinnedSeries& s, int color, int marker, double msize) {
  int n = (int)s.y.size();
  auto* g = new TGraphErrors(n);
  for (int i = 0; i < n; ++i) {
    g->SetPoint(i, s.xc[i], s.y[i]);
    g->SetPointError(i, s.xerr[i], s.ey[i]);
  }
  g->SetLineColor(color);
  g->SetMarkerColor(color);
  g->SetMarkerStyle(marker);
  g->SetMarkerSize(msize);
  g->SetLineWidth(2);
  return g;
}

TGraphErrors* makeUnityGraph(const BinnedSeries& s) {
  int n = (int)s.y.size();
  auto* g = new TGraphErrors(n);
  for (int i = 0; i < n; ++i) {
    double rel = (s.y[i] > 0.0) ? s.ey[i] / s.y[i] : 0.0;
    g->SetPoint(i, s.xc[i], 1.0);
    g->SetPointError(i, s.xerr[i], rel);
  }
  g->SetLineColor(kBlack);
  g->SetMarkerColor(kBlack);
  g->SetMarkerStyle(20);
  g->SetMarkerSize(0.8);
  g->SetLineWidth(1);
  return g;
}

TGraph* makeStepGraph(const BinnedSeries& s, int color, int lineStyle, int lineWidth) {
  int n = (int)s.y.size();
  auto* g = new TGraph(2*n);
  for (int i = 0; i < n; ++i) {
    g->SetPoint(2*i,   s.x1[i], s.y[i]);
    g->SetPoint(2*i+1, s.x2[i], s.y[i]);
  }
  g->SetLineColor(color);
  g->SetLineStyle(lineStyle);
  g->SetLineWidth(lineWidth);
  return g;
}

BinnedSeries makeRatioSeries(const BinnedSeries& th, const BinnedSeries& data) {
  BinnedSeries out;
  for (size_t i = 0; i < th.y.size() && i < data.y.size(); ++i) {
    double r = (data.y[i] > 0.0) ? th.y[i] / data.y[i] : 0.0;
    out.x1.push_back(data.x1[i]);
    out.x2.push_back(data.x2[i]);
    out.xc.push_back(data.xc[i]);
    out.xerr.push_back(data.xerr[i]);
    out.y.push_back(r);
    out.ey.push_back(0.0);
  }
  return out;
}

void drawHeader(const char* line1, const char* line2 = nullptr) {
  TLatex tl;
  tl.SetNDC();
  tl.SetTextFont(42);
  tl.SetTextSize(0.070);
  tl.DrawLatex(0.08, 0.90, line1);
  if (line2 && std::string(line2).size()) {
    tl.SetTextSize(0.040);
    tl.DrawLatex(0.08, 0.82, line2);
  }
}

void styleTopFrame(TH1D* h, const char* ytitle, double ymin, double ymax) {
  h->SetTitle("");
  h->GetXaxis()->SetTitle("");
  h->GetXaxis()->SetLabelSize(0.0);
  h->GetXaxis()->SetTitleSize(0.0);
  h->GetYaxis()->SetTitle(ytitle);
  h->GetYaxis()->SetTitleSize(0.075);
  h->GetYaxis()->SetLabelSize(0.055);
  h->GetYaxis()->SetTitleOffset(0.78);
  h->GetYaxis()->SetRangeUser(ymin, ymax);
  h->GetXaxis()->SetRangeUser(0.0, 10.0);
}

void styleBottomFrame(TH1D* h, double ymin, double ymax) {
  h->SetTitle("");
  h->GetXaxis()->SetTitle("q_{T} [GeV]");
  h->GetYaxis()->SetTitle("Theory / PYTHIA");
  h->GetYaxis()->SetNdivisions(505);
  h->GetYaxis()->SetTitleSize(0.11);
  h->GetYaxis()->SetLabelSize(0.10);
  h->GetYaxis()->SetTitleOffset(0.42);
  h->GetXaxis()->SetTitleSize(0.13);
  h->GetXaxis()->SetLabelSize(0.11);
  h->GetXaxis()->SetTitleOffset(1.0);
  h->GetYaxis()->SetRangeUser(ymin, ymax);
  h->GetXaxis()->SetRangeUser(0.0, 10.0);
}

void saveSummaryTxt(const char* fname,
                    const std::vector<double>& edges,
                    const BinnedSeries& dataOS,
                    const BinnedSeries& dataSS,
                    const BinnedSeries& loOS,
                    const BinnedSeries& nloOS,
                    const BinnedSeries& nnloOS,
                    const BinnedSeries& n3loOS,
                    const BinnedSeries& loSS,
                    const BinnedSeries& nloSS,
                    const BinnedSeries& nnloSS,
                    const BinnedSeries& n3loSS,
                    double scaleLO, double scaleNLO, double scaleNNLO, double scaleN3LO) {
  std::ofstream out(fname);
  out << "# cut 60\n# z 0.70\n";
  out << "# scale_LO " << scaleLO << "\n";
  out << "# scale_NLO " << scaleNLO << "\n";
  out << "# scale_NNLO " << scaleNNLO << "\n";
  out << "# scale_N3LO " << scaleN3LO << "\n";
  out << "# columns: x1 x2 dataOS errOS dataSS errSS loOS nloOS nnloOS n3loOS loSS nloSS nnloSS n3loSS\n";
  for (size_t i = 0; i + 1 < edges.size(); ++i) {
    out << edges[i] << ' ' << edges[i+1] << ' '
        << dataOS.y[i] << ' ' << dataOS.ey[i] << ' '
        << dataSS.y[i] << ' ' << dataSS.ey[i] << ' '
        << loOS.y[i] << ' ' << nloOS.y[i] << ' ' << nnloOS.y[i] << ' ' << n3loOS.y[i] << ' '
        << loSS.y[i] << ' ' << nloSS.y[i] << ' ' << nnloSS.y[i] << ' ' << n3loSS.y[i] << '\n';
  }
}

void drawOSOnlyCanvas(const TString& outBase,
                      const BinnedSeries& dataOS,
                      const BinnedSeries& loOS,
                      const BinnedSeries& nloOS,
                      const BinnedSeries& nnloOS,
                      const BinnedSeries& n3loOS) {
  auto* c = new TCanvas("c_os_only", "c_os_only", 1100, 850);
  auto* top = new TPad("top_os", "", 0.0, 0.30, 1.0, 1.0);
  auto* bot = new TPad("bot_os", "", 0.0, 0.0, 1.0, 0.30);
  top->SetBottomMargin(0.02); top->SetLeftMargin(0.12); top->SetRightMargin(0.03); top->SetTopMargin(0.04);
  bot->SetTopMargin(0.03); bot->SetBottomMargin(0.30); bot->SetLeftMargin(0.12); bot->SetRightMargin(0.03);
  top->Draw(); bot->Draw();

  double ymax = 0.0;
  for (size_t i = 0; i < dataOS.y.size(); ++i) ymax = std::max(ymax, dataOS.y[i] + dataOS.ey[i]);
  for (double v : loOS.y)   ymax = std::max(ymax, v);
  for (double v : nloOS.y)  ymax = std::max(ymax, v);
  for (double v : nnloOS.y) ymax = std::max(ymax, v);
  for (double v : n3loOS.y) ymax = std::max(ymax, v);
  ymax *= 1.22;

  auto* frameTop = new TH1D("frameTop_os", "", 100, 0.0, 10.0);
  styleTopFrame(frameTop, "Events", 0.0, ymax);

  auto* gData = makeDataGraph(dataOS, kBlack, 20, 0.85);
  auto* gLO   = makeStepGraph(loOS,   kRed+1,   1, 3);
  auto* gNLO  = makeStepGraph(nloOS,  kGreen+2, 1, 3);
  auto* gNNLO = makeStepGraph(nnloOS, kOrange+7,1, 3);
  auto* gN3LO = makeStepGraph(n3loOS, kBlue+1,  1, 3);

  top->cd();
  frameTop->Draw();
  gLO->Draw("L SAME");
  gNLO->Draw("L SAME");
  gNNLO->Draw("L SAME");
  gN3LO->Draw("L SAME");
  gData->Draw("P SAME");
  drawHeader("Benchmark   cut 60%   z = 0.70", "OS only, wide-bin order comparison");

  auto* leg = new TLegend(0.63, 0.66, 0.94, 0.90);
  leg->SetBorderSize(0); leg->SetFillStyle(0); leg->SetTextSize(0.045);
  leg->AddEntry(gLO,   "LO",   "l");
  leg->AddEntry(gNLO,  "NLO",  "l");
  leg->AddEntry(gNNLO, "NNLO", "l");
  leg->AddEntry(gN3LO, "N^{3}LO", "l");
  leg->AddEntry(gData, "PYTHIA", "pe");
  leg->Draw();

  auto* frameBot = new TH1D("frameBot_os", "", 100, 0.0, 10.0);
  styleBottomFrame(frameBot, 0.78, 1.28);
  auto* one = new TLine(0.0, 1.0, 10.0, 1.0); one->SetLineStyle(2); one->SetLineWidth(2);
  auto* gUnity = makeUnityGraph(dataOS);
  auto* gRLO   = makeStepGraph(makeRatioSeries(loOS,   dataOS), kRed+1,   1, 3);
  auto* gRNLO  = makeStepGraph(makeRatioSeries(nloOS,  dataOS), kGreen+2, 1, 3);
  auto* gRNNLO = makeStepGraph(makeRatioSeries(nnloOS, dataOS), kOrange+7,1, 3);
  auto* gRN3LO = makeStepGraph(makeRatioSeries(n3loOS, dataOS), kBlue+1,  1, 3);

  bot->cd();
  frameBot->Draw();
  one->Draw("SAME");
  gRLO->Draw("L SAME");
  gRNLO->Draw("L SAME");
  gRNNLO->Draw("L SAME");
  gRN3LO->Draw("L SAME");
  gUnity->Draw("P SAME");

  c->SaveAs(outBase + ".pdf");
  c->SaveAs(outBase + ".png");
}

void drawOSSSCanvas(const TString& outBase,
                    const BinnedSeries& dataOS,
                    const BinnedSeries& dataSS,
                    const BinnedSeries& loOS,
                    const BinnedSeries& nloOS,
                    const BinnedSeries& nnloOS,
                    const BinnedSeries& n3loOS,
                    const BinnedSeries& loSS,
                    const BinnedSeries& nloSS,
                    const BinnedSeries& nnloSS,
                    const BinnedSeries& n3loSS) {
  auto* c = new TCanvas("c_osss", "c_osss", 1180, 880);
  auto* top = new TPad("top_osss", "", 0.0, 0.30, 1.0, 1.0);
  auto* bot = new TPad("bot_osss", "", 0.0, 0.0, 1.0, 0.30);
  top->SetBottomMargin(0.02); top->SetLeftMargin(0.12); top->SetRightMargin(0.03); top->SetTopMargin(0.04);
  bot->SetTopMargin(0.03); bot->SetBottomMargin(0.30); bot->SetLeftMargin(0.12); bot->SetRightMargin(0.03);
  top->Draw(); bot->Draw();

  double ymax = 0.0;
  for (size_t i = 0; i < dataOS.y.size(); ++i) ymax = std::max(ymax, dataOS.y[i] + dataOS.ey[i]);
  for (size_t i = 0; i < dataSS.y.size(); ++i) ymax = std::max(ymax, dataSS.y[i] + dataSS.ey[i]);
  for (double v : loOS.y) ymax = std::max(ymax, v);
  for (double v : nloOS.y) ymax = std::max(ymax, v);
  for (double v : nnloOS.y) ymax = std::max(ymax, v);
  for (double v : n3loOS.y) ymax = std::max(ymax, v);
  ymax *= 1.25;

  auto* frameTop = new TH1D("frameTop_osss", "", 100, 0.0, 10.0);
  styleTopFrame(frameTop, "Events", 0.0, ymax);

  auto* gDataOS = makeDataGraph(dataOS, kBlack, 20, 0.85);
  auto* gDataSS = makeDataGraph(dataSS, kBlack, 24, 0.85);

  auto* gLOOS   = makeStepGraph(loOS,   kRed+1,   1, 3);
  auto* gNLOOS  = makeStepGraph(nloOS,  kGreen+2, 1, 3);
  auto* gNNLOOS = makeStepGraph(nnloOS, kOrange+7,1, 3);
  auto* gN3LOOS = makeStepGraph(n3loOS, kBlue+1,  1, 3);
  auto* gLOSS   = makeStepGraph(loSS,   kRed+1,   2, 3);
  auto* gNLOSS  = makeStepGraph(nloSS,  kGreen+2, 2, 3);
  auto* gNNLOSS = makeStepGraph(nnloSS, kOrange+7,2, 3);
  auto* gN3LOSS = makeStepGraph(n3loSS, kBlue+1,  2, 3);

  top->cd();
  frameTop->Draw();
  gLOOS->Draw("L SAME"); gNLOOS->Draw("L SAME"); gNNLOOS->Draw("L SAME"); gN3LOOS->Draw("L SAME");
  gLOSS->Draw("L SAME"); gNLOSS->Draw("L SAME"); gNNLOSS->Draw("L SAME"); gN3LOSS->Draw("L SAME");
  gDataOS->Draw("P SAME"); gDataSS->Draw("P SAME");
  drawHeader("Benchmark   cut 60%   z = 0.70", "OS + SS together (solid = OS, dashed = SS)");

  auto* leg1 = new TLegend(0.57, 0.61, 0.80, 0.90);
  leg1->SetBorderSize(0); leg1->SetFillStyle(0); leg1->SetTextSize(0.040);
  leg1->AddEntry(gLOOS,   "LO",   "l");
  leg1->AddEntry(gNLOOS,  "NLO",  "l");
  leg1->AddEntry(gNNLOOS, "NNLO", "l");
  leg1->AddEntry(gN3LOOS, "N^{3}LO", "l");
  leg1->AddEntry(gDataOS, "PYTHIA OS", "pe");
  leg1->AddEntry(gDataSS, "PYTHIA SS", "pe");
  leg1->Draw();

  auto* leg2 = new TLegend(0.80, 0.69, 0.96, 0.90);
  leg2->SetBorderSize(0); leg2->SetFillStyle(0); leg2->SetTextSize(0.038);
  leg2->AddEntry(gLOOS,  "solid = OS", "l");
  leg2->AddEntry(gLOSS,  "dashed = SS", "l");
  leg2->Draw();

  auto* frameBot = new TH1D("frameBot_osss", "", 100, 0.0, 10.0);
  styleBottomFrame(frameBot, 0.70, 1.45);
  auto* one = new TLine(0.0, 1.0, 10.0, 1.0); one->SetLineStyle(2); one->SetLineWidth(2);
  auto* gUnityOS = makeUnityGraph(dataOS); gUnityOS->SetMarkerStyle(20);
  auto* gUnitySS = makeUnityGraph(dataSS); gUnitySS->SetMarkerStyle(24);
  auto* gRLOOS   = makeStepGraph(makeRatioSeries(loOS,   dataOS), kRed+1,   1, 3);
  auto* gRNLOOS  = makeStepGraph(makeRatioSeries(nloOS,  dataOS), kGreen+2, 1, 3);
  auto* gRNNLOOS = makeStepGraph(makeRatioSeries(nnloOS, dataOS), kOrange+7,1, 3);
  auto* gRN3LOOS = makeStepGraph(makeRatioSeries(n3loOS, dataOS), kBlue+1,  1, 3);
  auto* gRLOSS   = makeStepGraph(makeRatioSeries(loSS,   dataSS), kRed+1,   2, 3);
  auto* gRNLOSS  = makeStepGraph(makeRatioSeries(nloSS,  dataSS), kGreen+2, 2, 3);
  auto* gRNNLOSS = makeStepGraph(makeRatioSeries(nnloSS, dataSS), kOrange+7,2, 3);
  auto* gRN3LOSS = makeStepGraph(makeRatioSeries(n3loSS, dataSS), kBlue+1,  2, 3);

  bot->cd();
  frameBot->Draw();
  one->Draw("SAME");
  gRLOOS->Draw("L SAME"); gRNLOOS->Draw("L SAME"); gRNNLOOS->Draw("L SAME"); gRN3LOOS->Draw("L SAME");
  gRLOSS->Draw("L SAME"); gRNLOSS->Draw("L SAME"); gRNNLOSS->Draw("L SAME"); gRN3LOSS->Draw("L SAME");
  gUnityOS->Draw("P SAME"); gUnitySS->Draw("P SAME");

  c->SaveAs(outBase + ".pdf");
  c->SaveAs(outBase + ".png");
}

void atlas_style_orders_events_compare(
  const char* pythiaFile = "output_100M.root",
  const char* theoryLO   = "../epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_LO.dat",
  const char* theoryNLO  = "../epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_NLO.dat",
  const char* theoryNNLO = "../epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_NNLO.dat",
  const char* theoryN3LO = "../epemTMD-main-Final/theory_zscan/theory_z_0p70.dat") {

  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(6);
  gStyle->SetLineWidth(2);

  TH1D *hOS = nullptr, *hSS = nullptr;
  if (!loadCut60Hists(pythiaFile, hOS, hSS)) {
    std::cerr << "Failed to load benchmark PYTHIA histograms/tree from " << pythiaFile << "\n";
    return;
  }

  TheoryData thLO, thNLO, thNNLO, thN3LO;
  if (!loadTheoryData(theoryLO, thLO)) return;
  if (!loadTheoryData(theoryNLO, thNLO)) return;
  if (!loadTheoryData(theoryNNLO, thNNLO)) return;
  if (!loadTheoryData(theoryN3LO, thN3LO)) return;

  std::vector<double> edges = {0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0};

  double scaleLO   = (theoryPeakOS(thLO)   > 0.0) ? histPeak(hOS, 10.0) / theoryPeakOS(thLO)   : 0.0;
  double scaleNLO  = (theoryPeakOS(thNLO)  > 0.0) ? histPeak(hOS, 10.0) / theoryPeakOS(thNLO)  : 0.0;
  double scaleNNLO = (theoryPeakOS(thNNLO) > 0.0) ? histPeak(hOS, 10.0) / theoryPeakOS(thNNLO) : 0.0;
  double scaleN3LO = (theoryPeakOS(thN3LO) > 0.0) ? histPeak(hOS, 10.0) / theoryPeakOS(thN3LO) : 0.0;

  auto dataOS = makeBinnedData(hOS, edges);
  auto dataSS = makeBinnedData(hSS, edges);
  auto loOS   = makeBinnedTheory(thLO,   edges, true,  scaleLO);
  auto nloOS  = makeBinnedTheory(thNLO,  edges, true,  scaleNLO);
  auto nnloOS = makeBinnedTheory(thNNLO, edges, true,  scaleNNLO);
  auto n3loOS = makeBinnedTheory(thN3LO, edges, true,  scaleN3LO);
  auto loSS   = makeBinnedTheory(thLO,   edges, false, scaleLO);
  auto nloSS  = makeBinnedTheory(thNLO,  edges, false, scaleNLO);
  auto nnloSS = makeBinnedTheory(thNNLO, edges, false, scaleNNLO);
  auto n3loSS = makeBinnedTheory(thN3LO, edges, false, scaleN3LO);

  TDatime dt;
  TString stamp = Form("%04d%02d%02d_%02d%02d%02d",
                       dt.GetYear(), dt.GetMonth(), dt.GetDay(),
                       dt.GetHour(), dt.GetMinute(), dt.GetSecond());

  TString outOSOnly = TString::Format("atlas_style_orders_events_os_only_%s", stamp.Data());
  TString outOSSS   = TString::Format("atlas_style_orders_events_osss_%s", stamp.Data());
  TString outTxt    = TString::Format("atlas_style_orders_events_summary_%s.txt", stamp.Data());

  drawOSOnlyCanvas(outOSOnly, dataOS, loOS, nloOS, nnloOS, n3loOS);
  drawOSSSCanvas(outOSSS, dataOS, dataSS, loOS, nloOS, nnloOS, n3loOS, loSS, nloSS, nnloSS, n3loSS);
  saveSummaryTxt(outTxt, edges, dataOS, dataSS, loOS, nloOS, nnloOS, n3loOS, loSS, nloSS, nnloSS, n3loSS, scaleLO, scaleNLO, scaleNNLO, scaleN3LO);

  std::cout << "Saved " << outOSOnly << ".pdf/.png\n";
  std::cout << "Saved " << outOSSS   << ".pdf/.png\n";
  std::cout << "Saved " << outTxt    << "\n";
}
