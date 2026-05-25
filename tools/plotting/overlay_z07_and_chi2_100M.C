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
#include "TPaveText.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <utility>

struct Chi2Result {
  double chi2 = 0.0;
  int nUsed = 0;
};

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
  double ymin = 1e99;
  double ymax = -1e99;

  for (int b = 1; b <= hOS->GetNbinsX(); ++b) {
    double x = hOS->GetBinCenter(b);
    if (x > xMax) continue;

    double os = hOS->GetBinContent(b);
    double ss = hSS->GetBinContent(b);
    if (os <= 0.0 || ss <= 0.0) continue;

    double r = os / ss;
    double e = r * std::sqrt(std::pow(hOS->GetBinError(b)/os, 2) +
                             std::pow(hSS->GetBinError(b)/ss, 2));

    ymin = std::min(ymin, r - e);
    ymax = std::max(ymax, r + e);
  }

  if (!(ymin < ymax)) return {0.5, 1.5};

  ymin = std::min(ymin, 1.0);
  ymax = std::max(ymax, 1.0);

  double pad = 0.15 * (ymax - ymin);
  ymin = std::max(0.0, ymin - pad);
  ymax += pad;

  return {ymin, ymax};
}

static TH1D* makeTheoryOnDataBinning(const TH1D* hTheory, const TH1D* hData, const char* newname) {
  auto h = (TH1D*)hData->Clone(newname);
  h->Reset("ICES");
  h->SetDirectory(nullptr);
  h->Sumw2();

  for (int b = 1; b <= h->GetNbinsX(); ++b) {
    double x = h->GetBinCenter(b);
    double y = 0.0;

    if (x >= hTheory->GetXaxis()->GetXmin() && x <= hTheory->GetXaxis()->GetXmax())
      y = hTheory->Interpolate(x);

    h->SetBinContent(b, y);
    h->SetBinError(b, 0.0);
  }
  return h;
}

static double sharedScale(const TH1D* hOSData,
                          const TH1D* hSSData,
                          const TH1D* hOSTheory,
                          const TH1D* hSSTheory,
                          double xMin,
                          double xMax) {
  double num = 0.0;
  double den = 0.0;

  auto accumulate = [&](const TH1D* hD, const TH1D* hT) {
    for (int b = 1; b <= hD->GetNbinsX(); ++b) {
      double x = hD->GetBinCenter(b);
      if (x < xMin || x > xMax) continue;

      double d = hD->GetBinContent(b);
      double t = hT->GetBinContent(b);
      double e = hD->GetBinError(b);

      if (e <= 0.0) continue;

      num += d * t / (e * e);
      den += t * t / (e * e);
    }
  };

  accumulate(hOSData, hOSTheory);
  accumulate(hSSData, hSSTheory);

  if (den <= 0.0) return 0.0;
  return num / den;
}

static Chi2Result chi2AgainstTheory(const TH1D* hData,
                                    const TH1D* hTheory,
                                    double scale,
                                    double xMin,
                                    double xMax) {
  Chi2Result out;

  for (int b = 1; b <= hData->GetNbinsX(); ++b) {
    double x = hData->GetBinCenter(b);
    if (x < xMin || x > xMax) continue;

    double d = hData->GetBinContent(b);
    double t = scale * hTheory->GetBinContent(b);
    double e = hData->GetBinError(b);

    if (e <= 0.0) continue;

    out.chi2 += std::pow((d - t) / e, 2);
    out.nUsed++;
  }

  return out;
}

static void drawCellOSSSTheory(TPad* cell,
                               TH1D* hOSData,
                               TH1D* hSSData,
                               TH1D* hOSTheory,
                               TH1D* hSSTheory,
                               double sharedScaleVal,
                               const char* tag,
                               const char* title,
                               double xMax) {
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

  pTop->cd();

  auto frameTop = (TH1D*)hOSData->Clone(Form("frameTop_%s", tag));
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

  double maxData = std::max(histMaxWithErrors(hOSData), histMaxWithErrors(hSSData));
  double maxTheory = 0.0;
  for (int b = 1; b <= hOSTheory->GetNbinsX(); ++b) {
    maxTheory = std::max(maxTheory, sharedScaleVal * hOSTheory->GetBinContent(b));
    maxTheory = std::max(maxTheory, sharedScaleVal * hSSTheory->GetBinContent(b));
  }
  frameTop->SetMaximum(1.20 * std::max(maxData, maxTheory));
  frameTop->Draw();

  auto gOSBand = makeBand(hOSData, kRed+1, 0.40);
  auto gSSBand = makeBand(hSSData, kBlue+1, 0.40);
  auto gOSPts  = makePointErrors(hOSData, kRed+1, 20, 0.30);
  auto gSSPts  = makePointErrors(hSSData, kBlue+1, 24, 0.30);

  gOSBand->Draw("2 SAME");
  gSSBand->Draw("2 SAME");
  gOSPts->Draw("P E1 SAME");
  gSSPts->Draw("P E1 SAME");

  auto hOSTheoryScaled = (TH1D*)hOSTheory->Clone(Form("hOSTheoryScaled_%s", tag));
  auto hSSTheoryScaled = (TH1D*)hSSTheory->Clone(Form("hSSTheoryScaled_%s", tag));

  hOSTheoryScaled->Scale(sharedScaleVal);
  hSSTheoryScaled->Scale(sharedScaleVal);

  hOSTheoryScaled->SetLineColor(kRed+1);
  hOSTheoryScaled->SetLineStyle(2);
  hOSTheoryScaled->SetLineWidth(2);
  hOSTheoryScaled->SetMarkerSize(0);

  hSSTheoryScaled->SetLineColor(kBlue+1);
  hSSTheoryScaled->SetLineStyle(2);
  hSSTheoryScaled->SetLineWidth(2);
  hSSTheoryScaled->SetMarkerSize(0);

  hOSTheoryScaled->Draw("HIST SAME");
  hSSTheoryScaled->Draw("HIST SAME");

  auto leg = new TLegend(0.52, 0.68, 0.92, 0.90);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(gOSBand, "PYTHIA OS", "pf");
  leg->AddEntry(gSSBand, "PYTHIA SS", "pf");
  leg->AddEntry(hOSTheoryScaled, "TMD OS  z=0.7", "l");
  leg->AddEntry(hSSTheoryScaled, "TMD SS  z=0.7", "l");
  leg->Draw();

  pBot->cd();

  auto hR = (TH1D*)hOSData->Clone(Form("hRatio_%s", tag));
  hR->Sumw2();
  hR->Divide(hSSData);

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

  auto yr = findRatioRange(hOSData, hSSData, xMax);
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

void overlay_z07_and_chi2_100M() {
  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(5);

  // ------------------------------------------------------------
  // EDIT THESE 4 LINES TO MATCH YOUR VALERIO z=0.7 ROOT OUTPUT
  // ------------------------------------------------------------
  const char* THEORY_OS_FILE = "theory_z07_OS.root";
  const char* THEORY_SS_FILE = "theory_z07_SS.root";
  const char* THEORY_OS_HIST = "h_theory_OS";
  const char* THEORY_SS_HIST = "h_theory_SS";

  const char* PYTHIA_FILE = "output_100M.root";

  TFile *fP = TFile::Open(PYTHIA_FILE, "UPDATE");
  if (!fP || fP->IsZombie()) {
    std::cout << "Could not open " << PYTHIA_FILE << "\n";
    return;
  }

  TFile *fOS = TFile::Open(THEORY_OS_FILE, "READ");
  TFile *fSS = TFile::Open(THEORY_SS_FILE, "READ");
  if (!fOS || fOS->IsZombie() || !fSS || fSS->IsZombie()) {
    std::cout << "Could not open one of the theory files.\n";
    return;
  }

  auto hTheoryOSRaw = (TH1D*)fOS->Get(THEORY_OS_HIST);
  auto hTheorySSRaw = (TH1D*)fSS->Get(THEORY_SS_HIST);
  if (!hTheoryOSRaw || !hTheorySSRaw) {
    std::cout << "Could not find one of the theory histograms.\n";
    return;
  }

  int cuts[4] = {0,20,40,60};
  TH1D* hOSData[4];
  TH1D* hSSData[4];
  TH1D* hOSTheory[4];
  TH1D* hSSTheory[4];
  double scale[4];

  for (int i = 0; i < 4; ++i) {
    hOSData[i] = (TH1D*)fP->Get(Form("h_qT_highest_OS_pion_cut%d", cuts[i]));
    hSSData[i] = (TH1D*)fP->Get(Form("h_qT_highest_SS_pion_cut%d", cuts[i]));
    if (!hOSData[i] || !hSSData[i]) {
      std::cout << "Missing PYTHIA histogram for cut " << cuts[i] << "%\n";
      return;
    }

    hOSTheory[i] = makeTheoryOnDataBinning(hTheoryOSRaw, hOSData[i], Form("hOSTheory_cut%d", cuts[i]));
    hSSTheory[i] = makeTheoryOnDataBinning(hTheorySSRaw, hSSData[i], Form("hSSTheory_cut%d", cuts[i]));

    scale[i] = sharedScale(hOSData[i], hSSData[i], hOSTheory[i], hSSTheory[i], 0.0, 10.0);

    std::cout << "cut " << cuts[i] << "% : shared scale = " << scale[i] << "\n";
  }

  // 2x2 counts overlay
  auto c2x2 = new TCanvas("c_qT_OSSS_4cuts_pion_counts_100M_z07",
                          "100M counts with z=0.7 TMD overlay", 1400, 900);
  c2x2->Divide(2,2,0.01,0.01);

  for (int i = 0; i < 4; ++i) {
    c2x2->cd(i+1);
    drawCellOSSSTheory((TPad*)gPad,
                       hOSData[i], hSSData[i],
                       hOSTheory[i], hSSTheory[i],
                       scale[i],
                       Form("cut%d_100M_z07", cuts[i]),
                       Form("Highest pair   cut %d%%", cuts[i]),
                       10.0);
  }

  // Single 60% cut overlay
  int i60 = 3;
  auto c60 = new TCanvas("c_qT_OSSS_pion_cut60_counts_100M_z07",
                         "60% cut 100M with z=0.7 TMD overlay", 900, 700);
  drawCellOSSSTheory((TPad*)c60,
                     hOSData[i60], hSSData[i60],
                     hOSTheory[i60], hSSTheory[i60],
                     scale[i60],
                     "cut60_100M_z07_single",
                     "Highest pair   cut 60%",
                     10.0);

  // Chi-square on 60% cut using all bins
  Chi2Result chiOS = chi2AgainstTheory(hOSData[i60], hOSTheory[i60], scale[i60], 0.0, 10.0);
  Chi2Result chiSS = chi2AgainstTheory(hSSData[i60], hSSTheory[i60], scale[i60], 0.0, 10.0);

  double chi2Combined = chiOS.chi2 + chiSS.chi2;
  int ndfCombined = chiOS.nUsed + chiSS.nUsed - 1; // one shared fitted scale
  double redChi2Combined = (ndfCombined > 0) ? chi2Combined / ndfCombined : 0.0;

  std::cout << "\n===== 60% cut chi-square (all bins) =====\n";
  std::cout << "shared scale         = " << scale[i60] << "\n";
  std::cout << "OS chi2              = " << chiOS.chi2 << "   using " << chiOS.nUsed << " bins\n";
  std::cout << "SS chi2              = " << chiSS.chi2 << "   using " << chiSS.nUsed << " bins\n";
  std::cout << "combined chi2        = " << chi2Combined << "\n";
  std::cout << "combined ndf         = " << ndfCombined << "\n";
  std::cout << "reduced chi2 (chi2/ndf) = " << redChi2Combined << "\n";
  std::cout << "=========================================\n\n";

  // Put chi2 info onto the 60% plot
  TPad* pTop60 = (TPad*)c60->GetPrimitive("pTop_cut60_100M_z07_single");
  if (pTop60) {
    pTop60->cd();
    auto box = new TPaveText(0.16, 0.62, 0.48, 0.88, "NDC");
    box->SetFillStyle(0);
    box->SetBorderSize(0);
    box->SetTextAlign(12);
    box->SetTextSize(0.035);
    box->AddText(Form("shared scale = %.6g", scale[i60]));
    box->AddText(Form("#chi^{2}_{OS} = %.2f", chiOS.chi2));
    box->AddText(Form("#chi^{2}_{SS} = %.2f", chiSS.chi2));
    box->AddText(Form("#chi^{2}_{tot} = %.2f", chi2Combined));
    box->AddText(Form("#chi^{2}/ndf = %.2f", redChi2Combined));
    box->Draw();
  }

  // Save
  fP->cd();
  c2x2->Write("", TObject::kOverwrite);
  c60->Write("", TObject::kOverwrite);

  c2x2->SaveAs("c_qT_OSSS_4cuts_pion_counts_100M_z07.pdf");
  c2x2->SaveAs("c_qT_OSSS_4cuts_pion_counts_100M_z07.png");
  c60->SaveAs("c_qT_OSSS_pion_cut60_counts_100M_z07.pdf");
  c60->SaveAs("c_qT_OSSS_pion_cut60_counts_100M_z07.png");

  fP->Close();
  fOS->Close();
  fSS->Close();
}