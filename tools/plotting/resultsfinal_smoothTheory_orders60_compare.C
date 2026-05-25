#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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

Chi2Result computeChi2(const TH1D* hMC,
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

void drawOrderCell(TPad* cell,
                   TH1D* hOS,
                   TH1D* hSS,
                   const TheoryData& th,
                   const char* title,
                   double xPlotMax,
                   double xRatioAxisMax,
                   double& sharedScaleOut,
                   Chi2Result& chiOSOut,
                   Chi2Result& chiSSOut) {
  cell->cd();
  cell->SetMargin(0,0,0,0);

  auto pTop = new TPad(Form("pTop_%s", title), "", 0, 0.30, 1, 1);
  auto pBot = new TPad(Form("pBot_%s", title), "", 0, 0.00, 1, 0.30);

  pTop->SetBottomMargin(0.02);
  pTop->SetLeftMargin(0.13);
  pTop->SetRightMargin(0.03);

  pBot->SetTopMargin(0.02);
  pBot->SetBottomMargin(0.35);
  pBot->SetLeftMargin(0.13);
  pBot->SetRightMargin(0.03);

  pTop->Draw();
  pBot->Draw();

  const double sharedScale = (theoryPeakOS(th) > 0.0) ? pythiaPeakOS(hOS, xPlotMax) / theoryPeakOS(th) : 0.0;
  sharedScaleOut = sharedScale;

  chiOSOut = computeChi2(hOS, th, sharedScale, true, xPlotMax);
  chiSSOut = computeChi2(hSS, th, sharedScale, false, xPlotMax);

  auto gOSBand = makeBand(hOS, kRed + 2, 0.28);
  auto gSSBand = makeBand(hSS, kBlue + 2, 0.28);
  auto gOSPts  = makeBlackPointErrors(hOS);
  auto gSSPts  = makeBlackPointErrors(hSS);

  auto gThOS = makeScaledTheoryGraph(th, sharedScale, true,  kRed + 1);
  auto gThSS = makeScaledTheoryGraph(th, sharedScale, false, kBlue + 1);

  pTop->cd();

  auto frameTop = (TH1D*)hOS->Clone(Form("frameTop_%s", title));
  frameTop->Reset("ICES");
  frameTop->SetTitle(title);
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

  auto leg = new TLegend(0.54, 0.58, 0.93, 0.88);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.035);
  leg->AddEntry(gOSBand, "PYTHIA OS", "f");
  leg->AddEntry(gThOS, Form("TMD OS #times %.4f", sharedScale), "l");
  leg->AddEntry(gSSBand, "PYTHIA SS", "f");
  leg->AddEntry(gThSS, Form("TMD SS #times %.4f", sharedScale), "l");
  leg->Draw();

  pBot->cd();

  auto frameBot = new TH1D(Form("frameBot_%s", title), "", hOS->GetNbinsX(), hOS->GetXaxis()->GetXmin(), hOS->GetXaxis()->GetXmax());
  frameBot->SetTitle("");
  frameBot->GetXaxis()->SetTitle("q_{T} [GeV]");
  frameBot->GetYaxis()->SetTitle("Ratio");
  frameBot->GetXaxis()->SetTitleSize(0.12);
  frameBot->GetXaxis()->SetLabelSize(0.11);
  frameBot->GetYaxis()->SetTitleSize(0.11);
  frameBot->GetYaxis()->SetLabelSize(0.09);
  frameBot->GetYaxis()->SetTitleOffset(0.52);
  frameBot->GetYaxis()->CenterTitle();
  frameBot->GetXaxis()->SetRangeUser(0.0, xRatioAxisMax);
  frameBot->SetMinimum(0.5);
  frameBot->SetMaximum(1.5);
  frameBot->Draw();

  auto gRatioOS = makeRatioGraph(hOS, th, sharedScale, true,  kRed + 1, xPlotMax);
  auto gRatioSS = makeRatioGraph(hSS, th, sharedScale, false, kBlue + 1, xPlotMax);
  gRatioOS->Draw("P SAME");
  gRatioSS->Draw("P SAME");

  auto line1 = new TLine(0.0, 1.0, xRatioAxisMax, 1.0);
  line1->SetLineStyle(2);
  line1->SetLineWidth(2);
  line1->Draw();

  gPad->RedrawAxis();
}

void resultsfinal_smoothTheory_orders60_compare(
  const char* fileLO    = "/Users/shanesweetman/Desktop/TMD analysis week1/epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_LO.dat",
  const char* fileNLO   = "/Users/shanesweetman/Desktop/TMD analysis week1/epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_NLO.dat",
  const char* fileNNLO  = "/Users/shanesweetman/Desktop/TMD analysis week1/epemTMD-main-Final/theory_zscan_orders/theory_z_0p70_NNLO.dat",
  const char* fileN3LO  = "/Users/shanesweetman/Desktop/TMD analysis week1/epemTMD-main-Final/theory_zscan/theory_z_0p70.dat"
) {
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(1);

  const char* pythiaFile = "/Users/shanesweetman/Desktop/TMD analysis week1/TMD-Analysis/output_100M.root";
  const double xPlotMax = 10.0;
  const double xRatioAxisMax = 10.0;

  TFile* f = TFile::Open(pythiaFile, "READ");
  if (!f || f->IsZombie()) {
    std::cerr << "Could not open " << pythiaFile << "\n";
    return;
  }

  auto hRawOS = dynamic_cast<TH1D*>(f->Get("h_qT_highest_OS_pion_cut60"));
  auto hRawSS = dynamic_cast<TH1D*>(f->Get("h_qT_highest_SS_pion_cut60"));

  if (!hRawOS || !hRawSS) {
    std::cerr << "Could not find 60% histograms in output_100M.root\n";
    return;
  }

  auto hOS = (TH1D*)hRawOS->Clone("hOS60");
  auto hSS = (TH1D*)hRawSS->Clone("hSS60");
  hOS->SetDirectory(nullptr);
  hSS->SetDirectory(nullptr);

  TheoryData thLO, thNLO, thNNLO, thN3LO;
  if (!loadTheoryData(fileLO, thLO)) return;
  if (!loadTheoryData(fileNLO, thNLO)) return;
  if (!loadTheoryData(fileNNLO, thNNLO)) return;
  if (!loadTheoryData(fileN3LO, thN3LO)) return;

  auto c = new TCanvas("cOrders60", "cOrders60", 1600, 1000);
  c->Divide(2, 2, 0.01, 0.01);

  double Klo=0, Knlo=0, Knnlo=0, Kn3lo=0;
  Chi2Result chiOSlo, chiSSlo, chiOSnlo, chiSSnlo, chiOSnnlo, chiSSnnlo, chiOSn3lo, chiSSn3lo;

  c->cd(1);
  drawOrderCell((TPad*)gPad, hOS, hSS, thLO,
                "LO   z = 0.70   cut 60%",
                xPlotMax, xRatioAxisMax, Klo, chiOSlo, chiSSlo);

  c->cd(2);
  drawOrderCell((TPad*)gPad, hOS, hSS, thNLO,
                "NLO   z = 0.70   cut 60%",
                xPlotMax, xRatioAxisMax, Knlo, chiOSnlo, chiSSnlo);

  c->cd(3);
  drawOrderCell((TPad*)gPad, hOS, hSS, thNNLO,
                "NNLO   z = 0.70   cut 60%",
                xPlotMax, xRatioAxisMax, Knnlo, chiOSnnlo, chiSSnnlo);

  c->cd(4);
  drawOrderCell((TPad*)gPad, hOS, hSS, thN3LO,
                "N^{3}LO   z = 0.70   cut 60%",
                xPlotMax, xRatioAxisMax, Kn3lo, chiOSn3lo, chiSSn3lo);

  c->SaveAs("resultsfinal_smoothTheory_orders60_compare.pdf");
  c->SaveAs("resultsfinal_smoothTheory_orders60_compare.png");

  std::ofstream fout("resultsfinal_smoothTheory_orders60_compare_chi2.txt");
  fout << "60% cut comparison at z = 0.70\n\n";

  fout << "LO: shared scale K = " << Klo << "\n";
  fout << "  OS: chi2 = " << chiOSlo.chi2 << ", N = " << chiOSlo.nPoints
       << ", chi2/N = " << (chiOSlo.nPoints > 0 ? chiOSlo.chi2 / chiOSlo.nPoints : 0.0) << "\n";
  fout << "  SS: chi2 = " << chiSSlo.chi2 << ", N = " << chiSSlo.nPoints
       << ", chi2/N = " << (chiSSlo.nPoints > 0 ? chiSSlo.chi2 / chiSSlo.nPoints : 0.0) << "\n\n";

  fout << "NLO: shared scale K = " << Knlo << "\n";
  fout << "  OS: chi2 = " << chiOSnlo.chi2 << ", N = " << chiOSnlo.nPoints
       << ", chi2/N = " << (chiOSnlo.nPoints > 0 ? chiOSnlo.chi2 / chiOSnlo.nPoints : 0.0) << "\n";
  fout << "  SS: chi2 = " << chiSSnlo.chi2 << ", N = " << chiSSnlo.nPoints
       << ", chi2/N = " << (chiSSnlo.nPoints > 0 ? chiSSnlo.chi2 / chiSSnlo.nPoints : 0.0) << "\n\n";

  fout << "NNLO: shared scale K = " << Knnlo << "\n";
  fout << "  OS: chi2 = " << chiOSnnlo.chi2 << ", N = " << chiOSnnlo.nPoints
       << ", chi2/N = " << (chiOSnnlo.nPoints > 0 ? chiOSnnlo.chi2 / chiOSnnlo.nPoints : 0.0) << "\n";
  fout << "  SS: chi2 = " << chiSSnnlo.chi2 << ", N = " << chiSSnnlo.nPoints
       << ", chi2/N = " << (chiSSnnlo.nPoints > 0 ? chiSSnnlo.chi2 / chiSSnnlo.nPoints : 0.0) << "\n\n";

  fout << "N3LO/current: shared scale K = " << Kn3lo << "\n";
  fout << "  OS: chi2 = " << chiOSn3lo.chi2 << ", N = " << chiOSn3lo.nPoints
       << ", chi2/N = " << (chiOSn3lo.nPoints > 0 ? chiOSn3lo.chi2 / chiOSn3lo.nPoints : 0.0) << "\n";
  fout << "  SS: chi2 = " << chiSSn3lo.chi2 << ", N = " << chiSSn3lo.nPoints
       << ", chi2/N = " << (chiSSn3lo.nPoints > 0 ? chiSSn3lo.chi2 / chiSSn3lo.nPoints : 0.0) << "\n";

  fout.close();

  std::cout << "Saved resultsfinal_smoothTheory_orders60_compare.pdf/.png\n";
  std::cout << "Saved resultsfinal_smoothTheory_orders60_compare_chi2.txt\n";
}
