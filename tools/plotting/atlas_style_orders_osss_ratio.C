#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TLine.h"
#include "TGraphErrors.h"
#include "TLatex.h"
#include "TString.h"
#include "TStyle.h"
#include "TDatime.h"

struct TheoryData {
  std::vector<double> qT;
  std::vector<double> os;
  std::vector<double> ss;
};

struct RatioBin {
  double x1 = 0.0;
  double x2 = 0.0;
  double xc = 0.0;
  double xerr = 0.0;
  double value = 0.0;
  double err = 0.0;
  bool valid = false;
};

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

double interpTheory(const TheoryData& th, double x, bool useOS) {
  const std::vector<double>& y = useOS ? th.os : th.ss;
  if (th.qT.empty()) return 0.0;
  if (x <= th.qT.front()) return y.front();
  if (x >= th.qT.back())  return y.back();

  for (size_t i = 0; i + 1 < th.qT.size(); ++i) {
    if (x >= th.qT[i] && x <= th.qT[i + 1]) {
      double dx = th.qT[i + 1] - th.qT[i];
      if (dx <= 0.0) return y[i];
      double t = (x - th.qT[i]) / dx;
      return y[i] + t * (y[i + 1] - y[i]);
    }
  }
  return y.back();
}

double integrateTheory(const TheoryData& th, double x1, double x2, bool useOS) {
  if (x2 <= x1) return 0.0;
  std::vector<double> xs;
  xs.push_back(x1);
  for (double q : th.qT) {
    if (q > x1 && q < x2) xs.push_back(q);
  }
  xs.push_back(x2);
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

  double integral = 0.0;
  for (size_t i = 0; i + 1 < xs.size(); ++i) {
    double xa = xs[i];
    double xb = xs[i + 1];
    double ya = interpTheory(th, xa, useOS);
    double yb = interpTheory(th, xb, useOS);
    integral += 0.5 * (ya + yb) * (xb - xa);
  }
  return integral;
}

RatioBin makeTheoryRatioBin(const TheoryData& th, double x1, double x2) {
  RatioBin out;
  out.x1 = x1;
  out.x2 = x2;
  out.xc = 0.5 * (x1 + x2);
  out.xerr = 0.5 * (x2 - x1);

  double ios = integrateTheory(th, x1, x2, true);
  double iss = integrateTheory(th, x1, x2, false);
  if (iss <= 0.0 || !std::isfinite(ios) || !std::isfinite(iss)) return out;

  out.value = ios / iss;
  out.err = 0.0;
  out.valid = std::isfinite(out.value);
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

  const int NBINS = 40;
  const double XMIN = 0.0;
  const double XMAX = 10.0;
  const double thr = cut / 100.0;

  hOS = new TH1D(Form("hOS_cut%d_rebuilt", cut), "", NBINS, XMIN, XMAX);
  hSS = new TH1D(Form("hSS_cut%d_rebuilt", cut), "", NBINS, XMIN, XMAX);
  hOS->SetDirectory(nullptr);
  hSS->SetDirectory(nullptr);
  hOS->Sumw2();
  hSS->Sumw2();

  const Long64_t nEntries = t->GetEntries();
  for (Long64_t i = 0; i < nEntries; ++i) {
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
    hOS = (TH1D*)hRawOS->Clone("hOS60_ratio_plot");
    hSS = (TH1D*)hRawSS->Clone("hSS60_ratio_plot");
    hOS->SetDirectory(nullptr);
    hSS->SetDirectory(nullptr);
    f->Close();
    return true;
  }

  std::cout << "60% histograms not found directly; rebuilding from tPionPairs.\n";
  bool ok = buildCutHistFromTree(f, 60, hOS, hSS);
  f->Close();
  if (!ok) {
    std::cerr << "Could not find cut60 histograms or tPionPairs in " << pythiaFile << "\n";
    return false;
  }
  return true;
}

RatioBin makePythiaRatioBin(const TH1D* hOS, const TH1D* hSS, double x1, double x2) {
  RatioBin out;
  out.x1 = x1;
  out.x2 = x2;
  out.xc = 0.5 * (x1 + x2);
  out.xerr = 0.5 * (x2 - x1);

  double os = 0.0, ss = 0.0;
  double eos2 = 0.0, ess2 = 0.0;

  const double eps = 1e-9;
  for (int b = 1; b <= hOS->GetNbinsX(); ++b) {
    double lo = hOS->GetXaxis()->GetBinLowEdge(b);
    double hi = hOS->GetXaxis()->GetBinUpEdge(b);
    if (lo >= x1 - eps && hi <= x2 + eps) {
      os += hOS->GetBinContent(b);
      ss += hSS->GetBinContent(b);
      eos2 += std::pow(hOS->GetBinError(b), 2);
      ess2 += std::pow(hSS->GetBinError(b), 2);
    }
  }

  if (os <= 0.0 || ss <= 0.0) return out;

  double eos = std::sqrt(eos2);
  double ess = std::sqrt(ess2);
  double r = os / ss;
  double er = r * std::sqrt(std::pow(eos / os, 2) + std::pow(ess / ss, 2));

  out.value = r;
  out.err = er;
  out.valid = std::isfinite(r) && std::isfinite(er);
  return out;
}

void styleTopFrame(TH1D* h, double ymin, double ymax) {
  h->SetTitle("");
  h->GetXaxis()->SetTitle("");
  h->GetXaxis()->SetLabelSize(0.0);
  h->GetXaxis()->SetTitleSize(0.0);
  h->GetYaxis()->SetTitle("R_{OS/SS}(q_{T})");
  h->GetYaxis()->SetTitleSize(0.070);
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
  h->GetYaxis()->SetTitleSize(0.115);
  h->GetYaxis()->SetLabelSize(0.095);
  h->GetYaxis()->SetTitleOffset(0.48);
  h->GetXaxis()->SetTitleSize(0.12);
  h->GetXaxis()->SetLabelSize(0.10);
  h->GetXaxis()->SetTitleOffset(1.0);
  h->GetXaxis()->SetRangeUser(0.0, 10.0);
  h->GetYaxis()->SetRangeUser(ymin, ymax);
}

TGraphErrors* makePythiaGraph(const std::vector<RatioBin>& bins) {
  int n = 0;
  for (const auto& b : bins) if (b.valid) ++n;
  auto g = new TGraphErrors(n);
  int j = 0;
  for (const auto& b : bins) {
    if (!b.valid) continue;
    g->SetPoint(j, b.xc, b.value);
    g->SetPointError(j, b.xerr, b.err);
    ++j;
  }
  g->SetLineColor(kBlack);
  g->SetLineWidth(2);
  g->SetMarkerColor(kBlack);
  g->SetMarkerStyle(20);
  g->SetMarkerSize(1.0);
  return g;
}

TGraphErrors* makeUnityGraphFromPythia(const std::vector<RatioBin>& bins) {
  int n = 0;
  for (const auto& b : bins) if (b.valid && b.value > 0.0) ++n;
  auto g = new TGraphErrors(n);
  int j = 0;
  for (const auto& b : bins) {
    if (!(b.valid && b.value > 0.0)) continue;
    double rel = b.err / b.value;
    g->SetPoint(j, b.xc, 1.0);
    g->SetPointError(j, b.xerr, rel);
    ++j;
  }
  g->SetLineColor(kBlack);
  g->SetLineWidth(2);
  g->SetMarkerColor(kBlack);
  g->SetMarkerStyle(20);
  g->SetMarkerSize(0.95);
  return g;
}

void drawTheorySegments(const std::vector<RatioBin>& bins, Color_t col, int width = 4) {
  for (const auto& b : bins) {
    if (!b.valid) continue;
    auto line = new TLine(b.x1, b.value, b.x2, b.value);
    line->SetLineColor(col);
    line->SetLineWidth(width);
    line->Draw("SAME");
  }
}

std::vector<RatioBin> makeTheoryRatioBins(const TheoryData& th, const std::vector<double>& edges) {
  std::vector<RatioBin> out;
  for (size_t i = 0; i + 1 < edges.size(); ++i)
    out.push_back(makeTheoryRatioBin(th, edges[i], edges[i + 1]));
  return out;
}

std::vector<RatioBin> makePythiaRatioBins(const TH1D* hOS, const TH1D* hSS, const std::vector<double>& edges) {
  std::vector<RatioBin> out;
  for (size_t i = 0; i + 1 < edges.size(); ++i)
    out.push_back(makePythiaRatioBin(hOS, hSS, edges[i], edges[i + 1]));
  return out;
}

std::vector<RatioBin> makeTheoryOverPythia(const std::vector<RatioBin>& th, const std::vector<RatioBin>& py) {
  std::vector<RatioBin> out;
  for (size_t i = 0; i < th.size() && i < py.size(); ++i) {
    RatioBin b;
    b.x1 = th[i].x1;
    b.x2 = th[i].x2;
    b.xc = th[i].xc;
    b.xerr = th[i].xerr;
    if (th[i].valid && py[i].valid && py[i].value > 0.0) {
      b.value = th[i].value / py[i].value;
      b.err = 0.0;
      b.valid = std::isfinite(b.value);
    }
    out.push_back(b);
  }
  return out;
}

void writeTable(const char* outTxt,
                const std::vector<RatioBin>& py,
                const std::vector<RatioBin>& lo,
                const std::vector<RatioBin>& nlo,
                const std::vector<RatioBin>& nnlo,
                const std::vector<RatioBin>& n3lo,
                const std::vector<RatioBin>& rlo,
                const std::vector<RatioBin>& rnlo,
                const std::vector<RatioBin>& rnnlo,
                const std::vector<RatioBin>& rn3lo) {
  std::ofstream out(outTxt);
  out << "# ATLAS-style OS/SS ratio benchmark table\n";
  out << "# cut = 60%, z = 0.70\n";
  out << "# columns: xlow xhigh pythiaRatio pythiaErr LO NLO NNLO N3LO LO_over_PY NLO_over_PY NNLO_over_PY N3LO_over_PY\n";

  for (size_t i = 0; i < py.size(); ++i) {
    out << py[i].x1 << ' ' << py[i].x2 << ' '
        << (py[i].valid ? py[i].value : -1.0) << ' '
        << (py[i].valid ? py[i].err   : -1.0) << ' '
        << (lo[i].valid   ? lo[i].value   : -1.0) << ' '
        << (nlo[i].valid  ? nlo[i].value  : -1.0) << ' '
        << (nnlo[i].valid ? nnlo[i].value : -1.0) << ' '
        << (n3lo[i].valid ? n3lo[i].value : -1.0) << ' '
        << (rlo[i].valid   ? rlo[i].value   : -1.0) << ' '
        << (rnlo[i].valid  ? rnlo[i].value  : -1.0) << ' '
        << (rnnlo[i].valid ? rnnlo[i].value : -1.0) << ' '
        << (rn3lo[i].valid ? rn3lo[i].value : -1.0) << '\n';
  }
}

void atlas_style_orders_osss_ratio(
  const char* pythiaFile = "output_100M.root",
  const char* fileLO    = "../epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_LO.dat",
  const char* fileNLO   = "../epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_NLO.dat",
  const char* fileNNLO  = "../epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_NNLO.dat",
  const char* fileN3LO  = "../epemTMD-main-Final/theory_zscan/theory_z_0p70.dat"
) {
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(5);
  gStyle->SetLineScalePS(1.2);

  TH1D* hOS = nullptr;
  TH1D* hSS = nullptr;
  if (!loadCut60Hists(pythiaFile, hOS, hSS)) return;

  TheoryData thLO, thNLO, thNNLO, thN3LO;
  if (!loadTheoryData(fileLO, thLO)) return;
  if (!loadTheoryData(fileNLO, thNLO)) return;
  if (!loadTheoryData(fileNNLO, thNNLO)) return;
  if (!loadTheoryData(fileN3LO, thN3LO)) return;

  const std::vector<double> edges = {0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0};

  auto pyBins   = makePythiaRatioBins(hOS, hSS, edges);
  auto loBins   = makeTheoryRatioBins(thLO,   edges);
  auto nloBins  = makeTheoryRatioBins(thNLO,  edges);
  auto nnloBins = makeTheoryRatioBins(thNNLO, edges);
  auto n3loBins = makeTheoryRatioBins(thN3LO, edges);

  auto rloBins   = makeTheoryOverPythia(loBins,   pyBins);
  auto rnloBins  = makeTheoryOverPythia(nloBins,  pyBins);
  auto rnnloBins = makeTheoryOverPythia(nnloBins, pyBins);
  auto rn3loBins = makeTheoryOverPythia(n3loBins, pyBins);

  double topMin =  std::numeric_limits<double>::max();
  double topMax = -std::numeric_limits<double>::max();
  for (const auto& b : pyBins) {
    if (!b.valid) continue;
    topMin = std::min(topMin, b.value - b.err);
    topMax = std::max(topMax, b.value + b.err);
  }
  for (const auto& v : {loBins, nloBins, nnloBins, n3loBins}) {
    for (const auto& b : v) {
      if (!b.valid) continue;
      topMin = std::min(topMin, b.value);
      topMax = std::max(topMax, b.value);
    }
  }
  if (!(topMin < topMax)) {
    topMin = 0.5;
    topMax = 2.0;
  }
  double topPad = 0.18 * (topMax - topMin);
  topMin = std::max(0.0, topMin - topPad);
  topMax = topMax + topPad;

  double botMin =  std::numeric_limits<double>::max();
  double botMax = -std::numeric_limits<double>::max();
  for (const auto& b : pyBins) {
    if (!(b.valid && b.value > 0.0)) continue;
    double rel = b.err / b.value;
    botMin = std::min(botMin, 1.0 - rel);
    botMax = std::max(botMax, 1.0 + rel);
  }
  for (const auto& v : {rloBins, rnloBins, rnnloBins, rn3loBins}) {
    for (const auto& b : v) {
      if (!b.valid) continue;
      botMin = std::min(botMin, b.value);
      botMax = std::max(botMax, b.value);
    }
  }
  if (!(botMin < botMax)) {
    botMin = 0.9;
    botMax = 1.1;
  }
  double botPad = 0.20 * (botMax - botMin);
  botMin -= botPad;
  botMax += botPad;
  botMin = std::min(botMin, 0.98);
  botMax = std::max(botMax, 1.02);

  auto c = new TCanvas("cAtlasStyleOrders", "cAtlasStyleOrders", 1500, 1050);
  c->SetMargin(0, 0, 0, 0);

  auto pTop = new TPad("pTop_atlas_orders", "", 0.0, 0.28, 1.0, 1.0);
  auto pBot = new TPad("pBot_atlas_orders", "", 0.0, 0.0, 1.0, 0.28);

  pTop->SetBottomMargin(0.02);
  pTop->SetLeftMargin(0.12);
  pTop->SetRightMargin(0.03);
  pTop->SetTopMargin(0.03);

  pBot->SetTopMargin(0.02);
  pBot->SetBottomMargin(0.33);
  pBot->SetLeftMargin(0.12);
  pBot->SetRightMargin(0.03);

  pTop->Draw();
  pBot->Draw();

  auto frameTop = new TH1D("frameTop_atlas_orders", "", 100, 0.0, 10.0);
  styleTopFrame(frameTop, topMin, topMax);

  auto frameBot = new TH1D("frameBot_atlas_orders", "", 100, 0.0, 10.0);
  styleBottomFrame(frameBot, botMin, botMax);

  auto gPyTop = makePythiaGraph(pyBins);
  auto gPyBot = makeUnityGraphFromPythia(pyBins);

  pTop->cd();
  frameTop->Draw();

  drawTheorySegments(loBins,   kRed + 1, 5);
  drawTheorySegments(nloBins,  kGreen + 2, 5);
  drawTheorySegments(nnloBins, kOrange + 7, 5);
  drawTheorySegments(n3loBins, kBlue + 1, 5);
  gPyTop->Draw("P E1 SAME");

  TLatex lat;
  lat.SetNDC();
  lat.SetTextFont(42);
  lat.SetTextSize(0.055);
  lat.DrawLatex(0.14, 0.90, "Benchmark   cut 60%   z = 0.70");

  TLatex lat2;
  lat2.SetNDC();
  lat2.SetTextFont(42);
  lat2.SetTextSize(0.035);
  lat2.DrawLatex(0.14, 0.84, "OS/SS ratio comparison across perturbative order");

  auto leg = new TLegend(0.68, 0.70, 0.95, 0.93);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextFont(42);
  leg->SetTextSize(0.040);

  auto lLO   = new TLine(0,0,1,0); lLO->SetLineColor(kRed + 1);     lLO->SetLineWidth(5);
  auto lNLO  = new TLine(0,0,1,0); lNLO->SetLineColor(kGreen + 2);  lNLO->SetLineWidth(5);
  auto lNNLO = new TLine(0,0,1,0); lNNLO->SetLineColor(kOrange + 7);lNNLO->SetLineWidth(5);
  auto lN3LO = new TLine(0,0,1,0); lN3LO->SetLineColor(kBlue + 1);  lN3LO->SetLineWidth(5);

  leg->AddEntry(lLO,   "LO", "l");
  leg->AddEntry(lNLO,  "NLO", "l");
  leg->AddEntry(lNNLO, "NNLO", "l");
  leg->AddEntry(lN3LO, "N^{3}LO", "l");
  leg->AddEntry(gPyTop, "PYTHIA", "pe");
  leg->Draw();

  gPad->RedrawAxis();

  pBot->cd();
  frameBot->Draw();

  auto line1 = new TLine(0.0, 1.0, 10.0, 1.0);
  line1->SetLineStyle(2);
  line1->SetLineWidth(2);
  line1->SetLineColor(kBlack);
  line1->Draw("SAME");

  drawTheorySegments(rloBins,   kRed + 1, 5);
  drawTheorySegments(rnloBins,  kGreen + 2, 5);
  drawTheorySegments(rnnloBins, kOrange + 7, 5);
  drawTheorySegments(rn3loBins, kBlue + 1, 5);
  gPyBot->Draw("P E1 SAME");

  gPad->RedrawAxis();

  TDatime now;
  TString tag = Form("%04d%02d%02d_%02d%02d%02d",
                     now.GetYear(), now.GetMonth(), now.GetDay(),
                     now.GetHour(), now.GetMinute(), now.GetSecond());

  TString pdfName  = Form("atlas_style_orders_osss_ratio_%s.pdf", tag.Data());
  TString pngName  = Form("atlas_style_orders_osss_ratio_%s.png", tag.Data());
  TString rootName = Form("atlas_style_orders_osss_ratio_%s.root", tag.Data());
  TString txtName  = Form("atlas_style_orders_osss_ratio_%s.txt", tag.Data());

  c->SaveAs(pdfName);
  c->SaveAs(pngName);

  TFile fout(rootName, "RECREATE");
  c->Write();
  gPyTop->Write("gPythiaTop");
  gPyBot->Write("gPythiaBottom");
  fout.Close();

  writeTable(txtName.Data(), pyBins, loBins, nloBins, nnloBins, n3loBins,
             rloBins, rnloBins, rnnloBins, rn3loBins);

  std::cout << "Saved " << pdfName  << "\n";
  std::cout << "Saved " << pngName  << "\n";
  std::cout << "Saved " << rootName << "\n";
  std::cout << "Saved " << txtName  << "\n";
}
