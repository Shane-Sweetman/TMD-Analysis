#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "TROOT.h"
#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TLine.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TString.h"
#include "TStyle.h"
#include "TDatime.h"

struct TheoryData {
  std::vector<double> qT;
  std::vector<double> os;
  std::vector<double> ss;
};

struct Chi2Result {
  double chi2 = 0.0;
  int nPoints = 0;
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

double theoryEval(const TheoryData& th, double x, bool useOS = true) {
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

TH1D* makeDisplayHist(const TH1D* hIn, const char* newname, bool normalised) {
  TH1D* h = (TH1D*)hIn->Clone(newname);
  h->SetDirectory(nullptr);
  if (h->GetSumw2N() == 0) h->Sumw2();

  if (normalised) {
    double integral = h->Integral("width");
    if (integral > 0.0) h->Scale(1.0 / integral);
  }
  return h;
}

double histMaxWithErrors(const TH1D* h, double xMax) {
  double out = 0.0;
  for (int b = 1; b <= h->GetNbinsX(); ++b) {
    double x = h->GetBinCenter(b);
    if (x > xMax) continue;
    out = std::max(out, h->GetBinContent(b) + h->GetBinError(b));
  }
  return out;
}

double theoryMax(const TheoryData& th, double scale, bool useOS, double xMax) {
  double out = 0.0;
  const std::vector<double>& y = useOS ? th.os : th.ss;
  for (size_t i = 0; i < th.qT.size(); ++i) {
    if (th.qT[i] > xMax) continue;
    out = std::max(out, scale * y[i]);
  }
  return out;
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

Chi2Result computeChi2(const TH1D* hMC,
                       const TheoryData& th,
                       double scale,
                       bool useOS,
                       double xMax,
                       double theoryFloorFrac = 0.0) {
  Chi2Result out;

  double thPeak = theoryMax(th, scale, useOS, xMax);
  double thFloor = theoryFloorFrac * thPeak;

  for (int b = 1; b <= hMC->GetNbinsX(); ++b) {
    double x = hMC->GetBinCenter(b);
    if (x > xMax) continue;

    double mc  = hMC->GetBinContent(b);
    double err = hMC->GetBinError(b);
    double tv  = scale * theoryEval(th, x, useOS);

    if (tv <= thFloor) continue;
    if (err <= 0.0) continue;

    double pull = (mc - tv) / err;
    if (!std::isfinite(pull)) continue;

    out.chi2 += pull * pull;
    out.nPoints++;
  }

  return out;
}

TGraph* makeScaledTheoryGraph(const TheoryData& th, double scale, bool useOS, Color_t col) {
  const std::vector<double>& y = useOS ? th.os : th.ss;
  TGraph* g = new TGraph((int)th.qT.size());

  for (int i = 0; i < (int)th.qT.size(); ++i) {
    g->SetPoint(i, th.qT[i], scale * y[i]);
  }

  g->SetLineColor(col);
  g->SetLineWidth(2);
  g->SetLineStyle(2);
  return g;
}

TGraphAsymmErrors* makeBand(const TH1D* h, Color_t col, double alpha) {
  TGraphAsymmErrors* g = new TGraphAsymmErrors(h);
  g->SetFillColorAlpha(col, alpha);
  g->SetLineColor(col);
  g->SetLineWidth(1);
  g->SetMarkerSize(0.0);
  return g;
}

TGraphErrors* makeBlackPointErrors(const TH1D* h, int markerStyle = 20, double markerSize = 0.35) {
  TGraphErrors* g = new TGraphErrors(h);

  for (int i = 0; i < g->GetN(); ++i) {
    g->SetPointError(i, 0.0, g->GetErrorY(i));
  }

  g->SetLineColor(kBlack);
  g->SetLineWidth(1);
  g->SetMarkerColor(kBlack);
  g->SetMarkerStyle(markerStyle);
  g->SetMarkerSize(markerSize);
  return g;
}

TGraphErrors* makeRatioGraph(const TH1D* hMC,
                             const TheoryData& th,
                             double scale,
                             bool useOS,
                             Color_t col,
                             double xPlotMax) {
  std::vector<double> xs, ys, exs, eys;

  for (int b = 1; b <= hMC->GetNbinsX(); ++b) {
    double x  = hMC->GetBinCenter(b);
    if (x > xPlotMax) continue;

    double mc = hMC->GetBinContent(b);
    double me = hMC->GetBinError(b);
    double tv = scale * theoryEval(th, x, useOS);

    if (tv <= 0.0) continue;

    xs.push_back(x);
    ys.push_back(mc / tv);
    exs.push_back(0.0);
    eys.push_back(me / tv);
  }

  TGraphErrors* g = new TGraphErrors((int)xs.size());
  for (int i = 0; i < (int)xs.size(); ++i) {
    g->SetPoint(i, xs[i], ys[i]);
    g->SetPointError(i, exs[i], eys[i]);
  }

  g->SetLineWidth(0);
  g->SetMarkerColor(col);
  g->SetMarkerStyle(20);
  g->SetMarkerSize(0.55);
  return g;
}

TGraphAsymmErrors* makeLegendBandProxy(Color_t col, double alpha) {
  TGraphAsymmErrors* g = new TGraphAsymmErrors(1);
  g->SetPoint(0, 0.5, 0.5);
  g->SetPointError(0, 0.25, 0.25, 0.18, 0.18);
  g->SetFillColorAlpha(col, alpha);
  g->SetLineColor(col);
  g->SetLineWidth(1);
  g->SetMarkerColor(kBlack);
  g->SetMarkerStyle(20);
  g->SetMarkerSize(0.7);
  return g;
}

TLine* makeLegendTheoryLine(Color_t col) {
  TLine* l = new TLine(0.0, 0.0, 1.0, 0.0);
  l->SetLineColor(col);
  l->SetLineStyle(2);
  l->SetLineWidth(2);
  return l;
}

void styleTopFrame(TH1D* h) {
  h->SetLineColor(0);
  h->SetLineWidth(0);
  h->SetMarkerSize(0);
}

void styleBottomFrame(TH1D* h) {
  h->SetLineColor(0);
  h->SetLineWidth(0);
  h->SetMarkerSize(0);
}

void drawCellCombined(TPad* cell,
                      TH1D* hRawOS,
                      TH1D* hRawSS,
                      const TheoryData& th,
                      int cut,
                      double sharedScale,
                      double xPlotMax,
                      double xRatioAxisMax,
                      double xRatioPlotMax,
                      bool normalised) {
  cell->cd();
  cell->SetMargin(0, 0, 0, 0);

  TPad* pTop = new TPad(Form("pTop_combined_cut%d_%d", cut, (int)normalised), "", 0, 0.30, 1, 1);
  TPad* pBot = new TPad(Form("pBot_combined_cut%d_%d", cut, (int)normalised), "", 0, 0.00, 1, 0.30);

  pTop->SetBottomMargin(0.02);
  pTop->SetLeftMargin(0.13);
  pTop->SetRightMargin(0.03);

  pBot->SetTopMargin(0.02);
  pBot->SetBottomMargin(0.35);
  pBot->SetLeftMargin(0.13);
  pBot->SetRightMargin(0.03);

  pTop->Draw();
  pBot->Draw();

  TH1D* hOS = makeDisplayHist(hRawOS, Form("hOS_combined_cut%d_%d", cut, (int)normalised), normalised);
  TH1D* hSS = makeDisplayHist(hRawSS, Form("hSS_combined_cut%d_%d", cut, (int)normalised), normalised);

  TGraph* gThOS = makeScaledTheoryGraph(th, sharedScale, true,  kRed + 1);
  TGraph* gThSS = makeScaledTheoryGraph(th, sharedScale, false, kBlue + 1);

  TGraphAsymmErrors* gOSBand = makeBand(hOS, kRed + 2, 0.28);
  TGraphAsymmErrors* gSSBand = makeBand(hSS, kBlue + 2, 0.28);

  TGraphErrors* gOSPts = makeBlackPointErrors(hOS);
  TGraphErrors* gSSPts = makeBlackPointErrors(hSS);

  pTop->cd();

  TH1D* frameTop = (TH1D*)hOS->Clone(Form("frameTop_combined_cut%d_%d", cut, (int)normalised));
  frameTop->Reset("ICES");
  frameTop->SetDirectory(nullptr);
  styleTopFrame(frameTop);
  frameTop->SetTitle(Form("Highest pair   cut %d%%", cut));
  frameTop->GetXaxis()->SetTitle("");
  frameTop->GetXaxis()->SetLabelSize(0.0);
  frameTop->GetXaxis()->SetTitleSize(0.0);
  frameTop->GetYaxis()->SetTitle(normalised ? "#scale[0.75]{#frac{1}{N}} #frac{dN}{dq_{T}} [GeV^{-1}]" : "Events");
  frameTop->GetYaxis()->SetTitleSize(0.060);
  frameTop->GetYaxis()->SetLabelSize(0.050);
  frameTop->GetYaxis()->SetTitleOffset(0.95);
  frameTop->GetYaxis()->CenterTitle(true);
  frameTop->GetXaxis()->SetRangeUser(0.0, xPlotMax);

  double ymax = std::max({
    histMaxWithErrors(hOS, xPlotMax),
    histMaxWithErrors(hSS, xPlotMax),
    theoryMax(th, sharedScale, true,  xPlotMax),
    theoryMax(th, sharedScale, false, xPlotMax)
  });

  frameTop->SetMinimum(0.0);
  frameTop->SetMaximum(1.18 * ymax);
  frameTop->Draw();

  gOSBand->Draw("2 SAME");
  gSSBand->Draw("2 SAME");
  gThOS->Draw("L SAME");
  gThSS->Draw("L SAME");
  gOSPts->Draw("P E1 SAME");
  gSSPts->Draw("P E1 SAME");
  gPad->RedrawAxis();

  TLegend* leg = new TLegend(0.80, 0.58, 0.992, 0.89);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.031);

  TGraphAsymmErrors* legOS = makeLegendBandProxy(kRed + 1, 0.25);
  TGraphAsymmErrors* legSS = makeLegendBandProxy(kBlue + 1, 0.25);
  TLine* legThOS = makeLegendTheoryLine(kRed + 1);
  TLine* legThSS = makeLegendTheoryLine(kBlue + 1);

  leg->AddEntry(legOS, "PYTHIA OS", "fp");
  leg->AddEntry(legThOS, Form("TMD OS #times %.6g", sharedScale), "l");
  leg->AddEntry(legSS, "PYTHIA SS", "fp");
  leg->AddEntry(legThSS, Form("TMD SS #times %.6g", sharedScale), "l");
  leg->Draw();

  pBot->cd();

  TGraphErrors* gROS = makeRatioGraph(hOS, th, sharedScale, true,  kRed + 1, xRatioPlotMax);
  TGraphErrors* gRSS = makeRatioGraph(hSS, th, sharedScale, false, kBlue + 1, xRatioPlotMax);

  TH1D* frameBot = (TH1D*)hOS->Clone(Form("frameBot_combined_cut%d_%d", cut, (int)normalised));
  frameBot->Reset("ICES");
  frameBot->SetDirectory(nullptr);
  styleBottomFrame(frameBot);
  frameBot->SetTitle("");
  frameBot->GetXaxis()->SetTitle("q_{T} [GeV]");
  frameBot->GetYaxis()->SetTitle("Ratio");
  frameBot->GetYaxis()->SetNdivisions(505);
  frameBot->GetYaxis()->SetTitleSize(0.12);
  frameBot->GetYaxis()->SetLabelSize(0.10);
  frameBot->GetYaxis()->SetTitleOffset(0.50);
  frameBot->GetXaxis()->SetTitleSize(0.12);
  frameBot->GetXaxis()->SetLabelSize(0.10);
  frameBot->GetXaxis()->SetRangeUser(0.0, xRatioAxisMax);
  frameBot->SetMinimum(0.5);
  frameBot->SetMaximum(1.5);
  frameBot->Draw();

  TLine* one = new TLine(0.0, 1.0, xRatioAxisMax, 1.0);
  one->SetLineColor(kBlack);
  one->SetLineStyle(2);
  one->SetLineWidth(1);
  one->Draw("SAME");

  gROS->Draw("P E1 SAME");
  gRSS->Draw("P E1 SAME");
  gPad->RedrawAxis();
}

void drawCellSingle(TPad* cell,
                    TH1D* hRaw,
                    const TheoryData& th,
                    int cut,
                    double sharedScale,
                    bool useOS,
                    double xPlotMax,
                    double xRatioAxisMax,
                    double xRatioPlotMax,
                    bool normalised) {
  cell->cd();
  cell->SetMargin(0, 0, 0, 0);

  TPad* pTop = new TPad(Form("pTop_single_%d_cut%d_%d", (int)useOS, cut, (int)normalised), "", 0, 0.30, 1, 1);
  TPad* pBot = new TPad(Form("pBot_single_%d_cut%d_%d", (int)useOS, cut, (int)normalised), "", 0, 0.00, 1, 0.30);

  pTop->SetBottomMargin(0.02);
  pTop->SetLeftMargin(0.13);
  pTop->SetRightMargin(0.03);

  pBot->SetTopMargin(0.02);
  pBot->SetBottomMargin(0.35);
  pBot->SetLeftMargin(0.13);
  pBot->SetRightMargin(0.03);

  pTop->Draw();
  pBot->Draw();

  TH1D* h = makeDisplayHist(hRaw,
                            Form("hSingle_%d_cut%d_%d", (int)useOS, cut, (int)normalised),
                            normalised);

  Color_t col = useOS ? kRed + 1 : kBlue + 1;
  const char* lab = useOS ? "OS" : "SS";

  TGraph* gTh = makeScaledTheoryGraph(th, sharedScale, useOS, col);
  Color_t bandCol = useOS ? kRed + 2 : kBlue + 2;
  TGraphAsymmErrors* gBand = makeBand(h, bandCol, 0.28);
  TGraphErrors* gPts = makeBlackPointErrors(h, 20, 0.30);

  pTop->cd();

  TH1D* frameTop = (TH1D*)h->Clone(Form("frameTop_single_%d_cut%d_%d", (int)useOS, cut, (int)normalised));
  frameTop->Reset("ICES");
  frameTop->SetDirectory(nullptr);
  styleTopFrame(frameTop);
  frameTop->SetTitle(Form("Highest pair   cut %d%%", cut));
  frameTop->GetXaxis()->SetTitle("");
  frameTop->GetXaxis()->SetLabelSize(0.0);
  frameTop->GetXaxis()->SetTitleSize(0.0);
  frameTop->GetYaxis()->SetTitle(normalised ? "#scale[0.75]{#frac{1}{N}} #frac{dN}{dq_{T}} [GeV^{-1}]" : "Events");
  frameTop->GetYaxis()->SetTitleSize(0.060);
  frameTop->GetYaxis()->SetLabelSize(0.050);
  frameTop->GetYaxis()->SetTitleOffset(0.95);
  frameTop->GetYaxis()->CenterTitle(true);
  frameTop->GetXaxis()->SetRangeUser(0.0, xPlotMax);

  double ymax = std::max(histMaxWithErrors(h, xPlotMax), theoryMax(th, sharedScale, useOS, xPlotMax));
  frameTop->SetMinimum(0.0);
  frameTop->SetMaximum(1.18 * ymax);
  frameTop->Draw();

  gBand->Draw("2 SAME");
  gTh->Draw("L SAME");
  gPts->Draw("P E1 SAME");
  gPad->RedrawAxis();

  TLegend* leg = new TLegend(0.81, 0.61, 0.992, 0.89);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.032);

  TGraphAsymmErrors* legPy = makeLegendBandProxy(col, 0.25);
  TLine* legTh = makeLegendTheoryLine(col);

  leg->AddEntry(legPy, Form("PYTHIA %s", lab), "fp");
  leg->AddEntry(legTh, Form("TMD %s #times %.6g", lab, sharedScale), "l");
  leg->Draw();

  pBot->cd();

  TGraphErrors* gR = makeRatioGraph(h, th, sharedScale, useOS, col, xRatioPlotMax);

  TH1D* frameBot = (TH1D*)h->Clone(Form("frameBot_single_%d_cut%d_%d", (int)useOS, cut, (int)normalised));
  frameBot->Reset("ICES");
  frameBot->SetDirectory(nullptr);
  styleBottomFrame(frameBot);
  frameBot->SetTitle("");
  frameBot->GetXaxis()->SetTitle("q_{T} [GeV]");
  frameBot->GetYaxis()->SetTitle("Ratio");
  frameBot->GetYaxis()->SetNdivisions(505);
  frameBot->GetYaxis()->SetTitleSize(0.12);
  frameBot->GetYaxis()->SetLabelSize(0.10);
  frameBot->GetYaxis()->SetTitleOffset(0.50);
  frameBot->GetXaxis()->SetTitleSize(0.12);
  frameBot->GetXaxis()->SetLabelSize(0.10);
  frameBot->GetXaxis()->SetRangeUser(0.0, xRatioAxisMax);
  frameBot->SetMinimum(0.0);
  frameBot->SetMaximum(2.0);
  frameBot->Draw();

  TLine* one = new TLine(0.0, 1.0, xRatioAxisMax, 1.0);
  one->SetLineColor(kBlack);
  one->SetLineStyle(2);
  one->SetLineWidth(1);
  one->Draw("SAME");

  gR->Draw("P E1 SAME");
  gPad->RedrawAxis();
}

void resultsfinal_smoothTheory_multi_edited(
  const char* pythiaFile = "output.root",
  const char* theoryFile = "data/theory/epemCrossSection_z0p70.dat",
  const char* outputTag = "public"
) {
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(1);

  const double xPlotMax      = 10.0;
  const double xRatioAxisMax = 10.0;
  const double xRatioPlotMax = 2.0;

  TheoryData th;
  if (!loadTheoryData(theoryFile, th)) return;

  double thPeakOS = theoryPeakOS(th);
  if (thPeakOS <= 0.0) {
    std::cerr << "Theory OS peak is not positive.\n";
    return;
  }

  TFile* f = TFile::Open(pythiaFile, "READ");
  if (!f || f->IsZombie()) {
    std::cerr << "Could not open " << pythiaFile << "\n";
    return;
  }

  std::vector<int> cuts = {0, 20, 40, 60};
  std::map<int,TH1D*> hRawOS, hRawSS;
  std::map<int,double> sharedScaleShape, sharedScaleCounts;

  for (int c : cuts) {
    TString nameOS = Form("h_qT_highest_OS_pion_cut%d", c);
    TString nameSS = Form("h_qT_highest_SS_pion_cut%d", c);

    hRawOS[c] = dynamic_cast<TH1D*>(f->Get(nameOS));
    hRawSS[c] = dynamic_cast<TH1D*>(f->Get(nameSS));

    if (!hRawOS[c] || !hRawSS[c]) {
      std::cerr << "Could not find " << nameOS << " or " << nameSS << " in " << pythiaFile << "\n";
      return;
    }

    TH1D* hOSShape = makeDisplayHist(hRawOS[c], Form("hOSShape_scale_%d", c), true);
    TH1D* hSSShape = makeDisplayHist(hRawSS[c], Form("hSSShape_scale_%d", c), true);

    double pyPeakShapeOS = pythiaPeakOS(hOSShape, xPlotMax);
    double pyPeakCountsOS = pythiaPeakOS(hRawOS[c], xPlotMax);

    sharedScaleShape[c]  = (pyPeakShapeOS > 0.0)  ? pyPeakShapeOS  / thPeakOS : 0.0;
    sharedScaleCounts[c] = (pyPeakCountsOS > 0.0) ? pyPeakCountsOS / thPeakOS : 0.0;

    std::cout << "cut " << c
              << "% : PYTHIA OS peak = " << pyPeakCountsOS
              << " , theory OS peak = " << thPeakOS
              << " , peak-matched counts scale K = " << sharedScaleCounts[c]
              << " , peak-matched shape scale K = " << sharedScaleShape[c]
              << "\n";
  }

  TCanvas* cCombinedNorm = new TCanvas("cCombinedNorm", "combined norm", 1400, 900);
  cCombinedNorm->Divide(2, 2, 0.01, 0.01);

  TCanvas* cOSNorm = new TCanvas("cOSNorm", "os norm", 1400, 900);
  cOSNorm->Divide(2, 2, 0.01, 0.01);

  TCanvas* cSSNorm = new TCanvas("cSSNorm", "ss norm", 1400, 900);
  cSSNorm->Divide(2, 2, 0.01, 0.01);

  TCanvas* cCombinedCounts = new TCanvas("cCombinedCounts", "combined counts", 1400, 900);
  cCombinedCounts->Divide(2, 2, 0.01, 0.01);

  TCanvas* cOSCounts = new TCanvas("cOSCounts", "os counts", 1400, 900);
  cOSCounts->Divide(2, 2, 0.01, 0.01);

  TCanvas* cSSCounts = new TCanvas("cSSCounts", "ss counts", 1400, 900);
  cSSCounts->Divide(2, 2, 0.01, 0.01);

  for (int i = 0; i < 4; ++i) {
    int cut = cuts[i];

    cCombinedNorm->cd(i + 1);
    drawCellCombined((TPad*)gPad,
                     hRawOS[cut], hRawSS[cut], th, cut,
                     sharedScaleShape[cut],
                     xPlotMax, xRatioAxisMax, xRatioPlotMax,
                     true);

    cOSNorm->cd(i + 1);
    drawCellSingle((TPad*)gPad,
                   hRawOS[cut], th, cut,
                   sharedScaleShape[cut], true,
                   xPlotMax, xRatioAxisMax, xRatioPlotMax,
                   true);

    cSSNorm->cd(i + 1);
    drawCellSingle((TPad*)gPad,
                   hRawSS[cut], th, cut,
                   sharedScaleShape[cut], false,
                   xPlotMax, xRatioAxisMax, xRatioPlotMax,
                   true);

    cCombinedCounts->cd(i + 1);
    drawCellCombined((TPad*)gPad,
                     hRawOS[cut], hRawSS[cut], th, cut,
                     sharedScaleCounts[cut],
                     xPlotMax, xRatioAxisMax, xRatioPlotMax,
                     false);

    cOSCounts->cd(i + 1);
    drawCellSingle((TPad*)gPad,
                   hRawOS[cut], th, cut,
                   sharedScaleCounts[cut], true,
                   xPlotMax, xRatioAxisMax, xRatioPlotMax,
                   false);

    cSSCounts->cd(i + 1);
    drawCellSingle((TPad*)gPad,
                   hRawSS[cut], th, cut,
                   sharedScaleCounts[cut], false,
                   xPlotMax, xRatioAxisMax, xRatioPlotMax,
                   false);
  }

  const int chiCut = 60;
  Chi2Result osCountsAll = computeChi2(hRawOS[chiCut], th, sharedScaleCounts[chiCut], true, 10.0, 0.0);
  Chi2Result ssCountsAll = computeChi2(hRawSS[chiCut], th, sharedScaleCounts[chiCut], false, 10.0, 0.0);

  double osChi2PerN = (osCountsAll.nPoints > 0) ? osCountsAll.chi2 / osCountsAll.nPoints : 0.0;
  double ssChi2PerN = (ssCountsAll.nPoints > 0) ? ssCountsAll.chi2 / ssCountsAll.nPoints : 0.0;

  double combinedChi2 = osCountsAll.chi2 + ssCountsAll.chi2;
  int combinedN = osCountsAll.nPoints + ssCountsAll.nPoints;
  double combinedReduced = (combinedN > 1) ? combinedChi2 / (combinedN - 1) : 0.0;

  TString tag(outputTag);
  if (tag.Length() == 0) tag = "public";

  TString txtName = Form("tmd_theory_overlay_chi2_%s.txt", tag.Data());
  std::ofstream foutTxt(txtName.Data());

  auto& os = foutTxt;
  os << "OS-peak-matched z=0.7 summary\n";
  os << "PYTHIA file: " << pythiaFile << "\n";
  os << "Theory file: " << theoryFile << "\n\n";

  os << "Peak-matched scales per cut\n";
  for (int c : cuts) {
    os << "  cut " << c << "% : shape scale = " << sharedScaleShape[c]
       << ", counts scale = " << sharedScaleCounts[c] << "\n";
  }
  os << "\n";

  os << "60% cut chi2 using all bins (counts)\n";
  os << "  peak-matched counts scale = " << sharedScaleCounts[chiCut] << "\n";
  os << "  OS: chi2 = " << osCountsAll.chi2
     << ", N = " << osCountsAll.nPoints
     << ", chi2/N = " << osChi2PerN << "\n";
  os << "  SS: chi2 = " << ssCountsAll.chi2
     << ", N = " << ssCountsAll.nPoints
     << ", chi2/N = " << ssChi2PerN << "\n";
  os << "  Combined: chi2 = " << combinedChi2
     << ", Ntot = " << combinedN
     << ", chi2/(Ntot-1) = " << combinedReduced << "\n";
  foutTxt.close();

  std::cout << "\n60% cut chi2 using all bins (counts)\n";
  std::cout << "  peak-matched counts scale = " << sharedScaleCounts[chiCut] << "\n";
  std::cout << "  OS: chi2 = " << osCountsAll.chi2
            << ", N = " << osCountsAll.nPoints
            << ", chi2/N = " << osChi2PerN << "\n";
  std::cout << "  SS: chi2 = " << ssCountsAll.chi2
            << ", N = " << ssCountsAll.nPoints
            << ", chi2/N = " << ssChi2PerN << "\n";
  std::cout << "  Combined: chi2 = " << combinedChi2
            << ", Ntot = " << combinedN
            << ", chi2/(Ntot-1) = " << combinedReduced << "\n\n";

  TString rootName = Form("tmd_theory_overlay_%s.root", tag.Data());

  TString pdfCombinedNorm   = Form("tmd_theory_overlay_norm_%s.pdf",      tag.Data());
  TString pdfOSNorm         = Form("tmd_theory_overlay_os_norm_%s.pdf",   tag.Data());
  TString pdfSSNorm         = Form("tmd_theory_overlay_ss_norm_%s.pdf",   tag.Data());
  TString pdfCombinedCounts = Form("tmd_theory_overlay_counts_%s.pdf",    tag.Data());
  TString pdfOSCounts       = Form("tmd_theory_overlay_os_counts_%s.pdf", tag.Data());
  TString pdfSSCounts       = Form("tmd_theory_overlay_ss_counts_%s.pdf", tag.Data());

  TString pngCombinedNorm   = Form("tmd_theory_overlay_norm_%s.png",      tag.Data());
  TString pngOSNorm         = Form("tmd_theory_overlay_os_norm_%s.png",   tag.Data());
  TString pngSSNorm         = Form("tmd_theory_overlay_ss_norm_%s.png",   tag.Data());
  TString pngCombinedCounts = Form("tmd_theory_overlay_counts_%s.png",    tag.Data());
  TString pngOSCounts       = Form("tmd_theory_overlay_os_counts_%s.png", tag.Data());
  TString pngSSCounts       = Form("tmd_theory_overlay_ss_counts_%s.png", tag.Data());

  TFile fout(rootName, "RECREATE");
  cCombinedNorm->Write();
  cOSNorm->Write();
  cSSNorm->Write();
  cCombinedCounts->Write();
  cOSCounts->Write();
  cSSCounts->Write();
  fout.Close();

  cCombinedNorm->SaveAs(pdfCombinedNorm);
  cOSNorm->SaveAs(pdfOSNorm);
  cSSNorm->SaveAs(pdfSSNorm);
  cCombinedCounts->SaveAs(pdfCombinedCounts);
  cOSCounts->SaveAs(pdfOSCounts);
  cSSCounts->SaveAs(pdfSSCounts);

  cCombinedNorm->SaveAs(pngCombinedNorm);
  cOSNorm->SaveAs(pngOSNorm);
  cSSNorm->SaveAs(pngSSNorm);
  cCombinedCounts->SaveAs(pngCombinedCounts);
  cOSCounts->SaveAs(pngOSCounts);
  cSSCounts->SaveAs(pngSSCounts);

  std::cout << "Saved:\n";
  std::cout << "  " << rootName << "\n";
  std::cout << "  " << pdfCombinedNorm   << "\n";
  std::cout << "  " << pdfOSNorm         << "\n";
  std::cout << "  " << pdfSSNorm         << "\n";
  std::cout << "  " << pdfCombinedCounts << "\n";
  std::cout << "  " << pdfOSCounts       << "\n";
  std::cout << "  " << pdfSSCounts       << "\n";
  std::cout << "  " << txtName           << "\n";
}
