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

double histMaxWithErrors(const TH1D* h, double xMax) {
  double out = 0.0;
  for (int b = 1; b <= h->GetNbinsX(); ++b) {
    double x = h->GetBinCenter(b);
    if (x > xMax) continue;
    out = std::max(out, h->GetBinContent(b) + h->GetBinError(b));
  }
  return out;
}

double theoryMaxScaled(const TheoryData& th, double scale, bool useOS, double xMax) {
  const std::vector<double>& y = useOS ? th.os : th.ss;
  double out = 0.0;
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

Chi2Result computeChi2AllBins(const TH1D* hMC,
                              const TheoryData& th,
                              double sharedScale,
                              bool useOS,
                              double xMax,
                              double theoryFloorFrac = 1e-3) {
  Chi2Result out;

  double thPeak = theoryMaxScaled(th, sharedScale, useOS, xMax);
  double thFloor = theoryFloorFrac * thPeak;

  for (int b = 1; b <= hMC->GetNbinsX(); ++b) {
    double x  = hMC->GetBinCenter(b);
    if (x > xMax) continue;

    double mc = hMC->GetBinContent(b);
    double me = hMC->GetBinError(b);
    double tv = sharedScale * theoryEval(th, x, useOS);

    if (tv <= thFloor) continue;
    if (me <= 0.0) continue;

    double pull = (mc - tv) / me;
    if (!std::isfinite(pull)) continue;

    out.chi2 += pull * pull;
    out.nPoints++;
  }

  return out;
}

TGraph* makeScaledTheoryGraph(const TheoryData& th, double scale, bool useOS, Color_t col) {
  const std::vector<double>& y = useOS ? th.os : th.ss;
  TGraph* g = new TGraph((int)th.qT.size());

  for (int i = 0; i < (int)th.qT.size(); ++i)
    g->SetPoint(i, th.qT[i], scale * y[i]);

  g->SetLineColor(col);
  g->SetLineWidth(2);
  g->SetLineStyle(2);
  return g;
}

TGraphAsymmErrors* makeBand(const TH1D* h, Color_t col, double alpha) {
  auto g = new TGraphAsymmErrors(h);
  g->SetFillColorAlpha(col, alpha);
  g->SetLineColor(col);
  g->SetLineWidth(1);
  g->SetMarkerSize(0.0);
  return g;
}

TGraphErrors* makeBlackPointErrors(const TH1D* h, int markerStyle = 20, double markerSize = 0.35) {
  auto g = new TGraphErrors(h);
  for (int i = 0; i < g->GetN(); ++i)
    g->SetPointError(i, 0.0, g->GetErrorY(i));

  g->SetLineColor(kBlack);
  g->SetLineWidth(1);
  g->SetMarkerColor(kBlack);
  g->SetMarkerStyle(markerStyle);
  g->SetMarkerSize(markerSize);
  return g;
}

TGraphErrors* makeRatioGraph(const TH1D* hMC,
                             const TheoryData& th,
                             double sharedScale,
                             bool useOS,
                             Color_t col,
                             double xPlotMax) {
  std::vector<double> xs, ys, exs, eys;

  for (int b = 1; b <= hMC->GetNbinsX(); ++b) {
    double x  = hMC->GetBinCenter(b);
    if (x > xPlotMax) continue;

    double mc = hMC->GetBinContent(b);
    double me = hMC->GetBinError(b);
    double tv = sharedScale * theoryEval(th, x, useOS);

    if (tv <= 0.0) continue;

    xs.push_back(x);
    ys.push_back(mc / tv);
    exs.push_back(0.0);
    eys.push_back(me / tv);
  }

  auto g = new TGraphErrors((int)xs.size());
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

void drawCell(TPad* cell,
              TH1D* hOS,
              TH1D* hSS,
              const TheoryData& th,
              int cut,
              double sharedScale,
              double xPlotMax,
              double xRatioAxisMax) {
  cell->cd();
  cell->SetMargin(0, 0, 0, 0);

  auto pTop = new TPad(Form("pTop_cut%d", cut), "", 0, 0.30, 1, 1);
  auto pBot = new TPad(Form("pBot_cut%d", cut), "", 0, 0.00, 1, 0.30);

  pTop->SetBottomMargin(0.02);
  pTop->SetLeftMargin(0.13);
  pTop->SetRightMargin(0.03);

  pBot->SetTopMargin(0.02);
  pBot->SetBottomMargin(0.35);
  pBot->SetLeftMargin(0.13);
  pBot->SetRightMargin(0.03);

  pTop->Draw();
  pBot->Draw();

  auto gOSBand = makeBand(hOS, kRed + 2, 0.28);
  auto gSSBand = makeBand(hSS, kBlue + 2, 0.28);
  auto gOSPts  = makeBlackPointErrors(hOS);
  auto gSSPts  = makeBlackPointErrors(hSS);

  auto gThOS = makeScaledTheoryGraph(th, sharedScale, true,  kRed + 1);
  auto gThSS = makeScaledTheoryGraph(th, sharedScale, false, kBlue + 1);

  pTop->cd();

  auto frameTop = (TH1D*)hOS->Clone(Form("frameTop_cut%d", cut));
  frameTop->Reset("ICES");
  frameTop->SetTitle(Form("Highest pair   cut %d%%", cut));
  frameTop->GetXaxis()->SetTitle("");
  frameTop->GetXaxis()->SetLabelSize(0.0);
  frameTop->GetXaxis()->SetTitleSize(0.0);
  frameTop->GetYaxis()->SetTitle("Events");
  frameTop->GetYaxis()->SetTitleSize(0.060);
  frameTop->GetYaxis()->SetLabelSize(0.050);
  frameTop->GetYaxis()->SetTitleOffset(0.95);
  frameTop->GetXaxis()->SetRangeUser(0.0, xPlotMax);
  frameTop->SetMinimum(0.0);

  double ymax = std::max({
    histMaxWithErrors(hOS, xPlotMax),
    histMaxWithErrors(hSS, xPlotMax),
    theoryMaxScaled(th, sharedScale, true,  xPlotMax),
    theoryMaxScaled(th, sharedScale, false, xPlotMax)
  });
  frameTop->SetMaximum(1.18 * ymax);
  frameTop->Draw();

  gOSBand->Draw("2 SAME");
  gSSBand->Draw("2 SAME");
  gThOS->Draw("L SAME");
  gThSS->Draw("L SAME");
  gOSPts->Draw("P E1 SAME");
  gSSPts->Draw("P E1 SAME");
  gPad->RedrawAxis();

  auto leg = new TLegend(0.78, 0.58, 0.995, 0.89);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.030);
  leg->AddEntry(gOSBand, "PYTHIA OS", "pf");
  leg->AddEntry(gThOS, Form("TMD OS #times %.4f", sharedScale), "l");
  leg->AddEntry(gSSBand, "PYTHIA SS", "pf");
  leg->AddEntry(gThSS, Form("TMD SS #times %.4f", sharedScale), "l");
  leg->Draw();

  pBot->cd();

  auto gROS = makeRatioGraph(hOS, th, sharedScale, true,  kRed + 1, xRatioAxisMax);
  auto gRSS = makeRatioGraph(hSS, th, sharedScale, false, kBlue + 1, xRatioAxisMax);

  auto frameBot = (TH1D*)hOS->Clone(Form("frameBot_cut%d", cut));
  frameBot->Reset("ICES");
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

  auto one = new TLine(0.0, 1.0, xRatioAxisMax, 1.0);
  one->SetLineColor(kBlack);
  one->SetLineStyle(2);
  one->SetLineWidth(1);
  one->Draw("SAME");

  gROS->Draw("P E1 SAME");
  gRSS->Draw("P E1 SAME");
  gPad->RedrawAxis();
}

void resultsfinal_smoothTheory_order_100M(const char* theoryFile = "/Users/shanesweetman/Desktop/TMD analysis week1/epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_LO.dat", const char* orderLabel = "LO") {
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(1);

  const char* pythiaFile = "/Users/shanesweetman/Desktop/TMD analysis week1/TMD-Analysis/output_100M.root";

  const double xPlotMax      = 10.0;
  const double xRatioAxisMax = 10.0;

  TheoryData th;
  if (!loadTheoryData(theoryFile, th)) return;

  std::cout << "Using theory file: " << theoryFile << "\n";
  std::cout << "Order label: " << orderLabel << "\n";

  TFile* f = TFile::Open(pythiaFile, "READ");
  if (!f || f->IsZombie()) {
    std::cerr << "Could not open " << pythiaFile << "\n";
    return;
  }

  std::vector<int> cuts = {0, 20, 40, 60};
  std::map<int,TH1D*> hRawOS, hRawSS;
  std::map<int,double> sharedScale;

  double thPeakOS = theoryPeakOS(th);
  if (thPeakOS <= 0.0) {
    std::cerr << "Theory OS peak is not positive.\n";
    return;
  }

  for (int c : cuts) {
    TString nameOS = Form("h_qT_highest_OS_pion_cut%d", c);
    TString nameSS = Form("h_qT_highest_SS_pion_cut%d", c);

    hRawOS[c] = dynamic_cast<TH1D*>(f->Get(nameOS));
    hRawSS[c] = dynamic_cast<TH1D*>(f->Get(nameSS));

    if (!hRawOS[c] || !hRawSS[c]) {
      std::cerr << "Could not find " << nameOS << " or " << nameSS << " in output_100M.root\n";
      return;
    }

    double pyPeakOS = pythiaPeakOS(hRawOS[c], xPlotMax);
    sharedScale[c] = (pyPeakOS > 0.0 ? pyPeakOS / thPeakOS : 1.0);

    std::cout << "cut " << c
              << "% : PYTHIA OS peak = " << pyPeakOS
              << " , theory OS peak = " << thPeakOS
              << " , shared scale K = " << sharedScale[c] << "\n";
  }

  auto cCombinedCounts = new TCanvas(Form("cCombinedCounts_%s_sharedPeak_100M", orderLabel),
                                     Form("combined counts shared peak scale 100M (%s)", orderLabel),
                                     1400, 900);
  cCombinedCounts->Divide(2, 2, 0.01, 0.01);

  for (int i = 0; i < 4; ++i) {
    int cut = cuts[i];
    cCombinedCounts->cd(i + 1);
    drawCell((TPad*)gPad,
             hRawOS[cut], hRawSS[cut], th, cut,
             sharedScale[cut],
             xPlotMax, xRatioAxisMax);
  }

  Chi2Result chiOS60 = computeChi2AllBins(hRawOS[60], th, sharedScale[60], true,  xPlotMax);
  Chi2Result chiSS60 = computeChi2AllBins(hRawSS[60], th, sharedScale[60], false, xPlotMax);

  const double redOS60 = (chiOS60.nPoints > 0 ? chiOS60.chi2 / chiOS60.nPoints : 0.0);
  const double redSS60 = (chiSS60.nPoints > 0 ? chiSS60.chi2 / chiSS60.nPoints : 0.0);

  std::cout << "\n60% cut chi-square summary (all valid bins)\n";
  std::cout << "  OS : chi2 = " << chiOS60.chi2
            << ", N = " << chiOS60.nPoints
            << ", chi2/N = " << redOS60 << "\n";
  std::cout << "  SS : chi2 = " << chiSS60.chi2
            << ", N = " << chiSS60.nPoints
            << ", chi2/N = " << redSS60 << "\n\n";

  TDatime now;
  TString tag = Form("%04d%02d%02d_%02d%02d%02d",
                     now.GetYear(), now.GetMonth(), now.GetDay(),
                     now.GetHour(), now.GetMinute(), now.GetSecond());

  TString rootName = Form("resultsfinal_smoothTheory_%s_100M_%s.root", orderLabel, tag.Data());
  TString pdfName  = Form("resultsfinal_smoothTheory_%s_100M_%s.pdf", orderLabel, tag.Data());
  TString pngName  = Form("resultsfinal_smoothTheory_%s_100M_%s.png", orderLabel, tag.Data());
  TString txtName  = Form("resultsfinal_smoothTheory_%s_100M_chi2_%s.txt", orderLabel, tag.Data());

  std::ofstream foutTxt(txtName.Data());
  foutTxt << "Order: " << orderLabel << "\n";
  foutTxt << "Theory file: " << theoryFile << "\n";
  foutTxt << "60% cut chi-square summary (all valid bins)\n";
  foutTxt << "OS : chi2 = " << chiOS60.chi2 << ", N = " << chiOS60.nPoints
          << ", chi2/N = " << redOS60 << "\n";
  foutTxt << "SS : chi2 = " << chiSS60.chi2 << ", N = " << chiSS60.nPoints
          << ", chi2/N = " << redSS60 << "\n";
  foutTxt.close();

  TFile fout(rootName, "RECREATE");
  cCombinedCounts->Write();
  fout.Close();

  cCombinedCounts->SaveAs(pdfName);
  cCombinedCounts->SaveAs(pngName);

  std::cout << "Saved:\n";
  std::cout << "  " << rootName << "\n";
  std::cout << "  " << pdfName  << "\n";
  std::cout << "  " << pngName  << "\n";
  std::cout << "  " << txtName  << "\n";
}
