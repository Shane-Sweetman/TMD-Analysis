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

  const int nb = hOS->GetNbinsX();
  for (int b = 1; b <= nb; ++b) {
    double x = hOS->GetBinCenter(b);
    if (x > xMax) continue;

    double os = hOS->GetBinContent(b);
    double ss = hSS->GetBinContent(b);
    if (os <= 0.0 || ss <= 0.0) continue;

    double r = os / ss;
    double e = r * std::sqrt(std::pow(hOS->GetBinError(b)/os,2) + std::pow(hSS->GetBinError(b)/ss,2));

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

void make_100M_canvases(const char* inputFile = "output_100M.root",
                        const char* outputTag = "100M") {
  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(5);

  TFile *f = TFile::Open(inputFile, "UPDATE");
  if (!f || f->IsZombie()) {
    std::cout << "Could not open " << inputFile << "\n";
    return;
  }

  int cuts[4] = {0,20,40,60};
  TH1D* hOS[4];
  TH1D* hSS[4];

  for (int i = 0; i < 4; ++i) {
    hOS[i] = (TH1D*)f->Get(Form("h_qT_highest_OS_pion_cut%d", cuts[i]));
    hSS[i] = (TH1D*)f->Get(Form("h_qT_highest_SS_pion_cut%d", cuts[i]));
    if (!hOS[i] || !hSS[i]) {
      std::cout << "Missing histogram for cut " << cuts[i] << "%\n";
      f->Close();
      return;
    }
  }

  TString tag(outputTag);

  auto cCounts = new TCanvas(Form("c_qT_OSSS_4cuts_pion_counts_%s", tag.Data()),
                             Form("Pion OS vs SS counts %s", tag.Data()),
                             1400, 900);
  cCounts->Divide(2,2,0.01,0.01);
  for (int i = 0; i < 4; ++i) {
    cCounts->cd(i+1);
    drawCellOSSS((TPad*)gPad, hOS[i], hSS[i],
                 Form("pion_cut%d_counts_%s", cuts[i], tag.Data()),
                 Form("Highest pair   cut %d%%", cuts[i]),
                 10.0,
                 false);
  }

  auto cNorm = new TCanvas(Form("c_qT_OSSS_4cuts_pion_norm_%s", tag.Data()),
                           Form("Pion OS vs SS normalised %s", tag.Data()),
                           1400, 900);
  cNorm->Divide(2,2,0.01,0.01);
  for (int i = 0; i < 4; ++i) {
    cNorm->cd(i+1);
    drawCellOSSS((TPad*)gPad, hOS[i], hSS[i],
                 Form("pion_cut%d_norm_%s", cuts[i], tag.Data()),
                 Form("Highest pair   cut %d%%", cuts[i]),
                 10.0,
                 true);
  }

  cCounts->Write("", TObject::kOverwrite);
  cNorm->Write("", TObject::kOverwrite);

  cCounts->SaveAs(Form("c_qT_OSSS_4cuts_pion_counts_%s.pdf", tag.Data()));
  cCounts->SaveAs(Form("c_qT_OSSS_4cuts_pion_counts_%s.png", tag.Data()));
  cNorm->SaveAs(Form("c_qT_OSSS_4cuts_pion_norm_%s.pdf", tag.Data()));
  cNorm->SaveAs(Form("c_qT_OSSS_4cuts_pion_norm_%s.png", tag.Data()));

  f->Close();
}
