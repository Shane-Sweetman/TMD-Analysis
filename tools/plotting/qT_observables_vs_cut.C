#include <iostream>
#include <vector>
#include <map>

#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TAxis.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TString.h"

struct ObsPoint {
  double x = 0.0;
  double y = 0.0;
  double ex = 0.0;
  double ey = 0.0;
};

ObsPoint meanPoint(const TH1D* h, double cut) {
  ObsPoint p;
  p.x  = cut;
  p.ex = 0.0;
  p.y  = h->GetMean();
  p.ey = h->GetMeanError();
  return p;
}

ObsPoint peakPoint(const TH1D* h, double cut) {
  ObsPoint p;
  p.x  = cut;
  p.ex = 0.0;

  const int bmax = h->GetMaximumBin();
  p.y  = h->GetBinCenter(bmax);
  p.ey = 0.5 * h->GetBinWidth(bmax); // simple binning uncertainty

  return p;
}

TGraphErrors* makeGraph(const std::vector<ObsPoint>& pts,
                        Color_t col,
                        int mstyle,
                        const char* name) {
  auto* g = new TGraphErrors((int)pts.size());
  g->SetName(name);

  for (int i = 0; i < (int)pts.size(); ++i) {
    g->SetPoint(i, pts[i].x, pts[i].y);
    g->SetPointError(i, pts[i].ex, pts[i].ey);
  }

  g->SetLineColor(col);
  g->SetMarkerColor(col);
  g->SetLineWidth(2);
  g->SetMarkerStyle(mstyle);
  g->SetMarkerSize(1.2);
  return g;
}

void styleAxes(TGraphErrors* g,
               const char* xtitle,
               const char* ytitle,
               double xmin, double xmax,
               double ymin, double ymax) {
  g->GetXaxis()->SetTitle(xtitle);
  g->GetYaxis()->SetTitle(ytitle);
  g->GetXaxis()->SetLimits(xmin, xmax);
  g->SetMinimum(ymin);
  g->SetMaximum(ymax);

  g->GetXaxis()->SetTitleSize(0.050);
  g->GetYaxis()->SetTitleSize(0.050);
  g->GetXaxis()->SetLabelSize(0.045);
  g->GetYaxis()->SetLabelSize(0.045);
  g->GetYaxis()->SetTitleOffset(1.05);
}

void qT_observables_vs_cut() {
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);

  const char* infile = "/Users/shanesweetman/Desktop/TMD analysis week1/TMD-Analysis/output.root";

  TFile* f = TFile::Open(infile, "READ");
  if (!f || f->IsZombie()) {
    std::cerr << "Could not open " << infile << "\n";
    return;
  }

  const std::vector<int> cuts = {0, 20, 40, 70};

  std::vector<ObsPoint> meanOS, meanSS;
  std::vector<ObsPoint> peakOS, peakSS;

  for (int c : cuts) {
    TString nameOS = Form("h_qT_highest_OS_pion_cut%d", c);
    TString nameSS = Form("h_qT_highest_SS_pion_cut%d", c);

    TH1D* hOS = dynamic_cast<TH1D*>(f->Get(nameOS));
    TH1D* hSS = dynamic_cast<TH1D*>(f->Get(nameSS));

    if (!hOS || !hSS) {
      std::cerr << "Missing histogram(s): " << nameOS << " and/or " << nameSS << "\n";
      return;
    }

    meanOS.push_back(meanPoint(hOS, c));
    meanSS.push_back(meanPoint(hSS, c));

    peakOS.push_back(peakPoint(hOS, c));
    peakSS.push_back(peakPoint(hSS, c));

    std::cout << "cut " << c
              << "% : <qT>_OS = " << meanOS.back().y << " +- " << meanOS.back().ey
              << " , <qT>_SS = " << meanSS.back().y << " +- " << meanSS.back().ey
              << " , qTmax_OS = " << peakOS.back().y << " +- " << peakOS.back().ey
              << " , qTmax_SS = " << peakSS.back().y << " +- " << peakSS.back().ey
              << "\n";
  }

  auto* gMeanOS = makeGraph(meanOS, kRed + 1, 20, "gMeanOS");
  auto* gMeanSS = makeGraph(meanSS, kBlue + 1, 24, "gMeanSS");

  auto* gPeakOS = makeGraph(peakOS, kRed + 1, 20, "gPeakOS");
  auto* gPeakSS = makeGraph(peakSS, kBlue + 1, 24, "gPeakSS");

  // ---------- <qT> vs cut ----------
  double meanYmin = 1e9, meanYmax = -1e9;
  for (const auto& p : meanOS) {
    meanYmin = std::min(meanYmin, p.y - p.ey);
    meanYmax = std::max(meanYmax, p.y + p.ey);
  }
  for (const auto& p : meanSS) {
    meanYmin = std::min(meanYmin, p.y - p.ey);
    meanYmax = std::max(meanYmax, p.y + p.ey);
  }
  double meanPad = 0.15 * (meanYmax - meanYmin);
  meanYmin -= meanPad;
  meanYmax += meanPad;

  auto* cMean = new TCanvas("cMean_qT_vs_cut", "Mean qT vs cut", 900, 700);
  gMeanOS->Draw("AP");
  styleAxes(gMeanOS,
            "cut [%]",
            "<q_{T}> [GeV]",
            -2.0, 72.0,
            meanYmin, meanYmax);

  gMeanOS->SetTitle("<q_{T}> vs pion momentum fraction cut");
  gMeanOS->Draw("AP");
  gMeanSS->Draw("P SAME");

  auto* legMean = new TLegend(0.66, 0.76, 0.92, 0.90);
  legMean->SetBorderSize(0);
  legMean->SetFillStyle(0);
  legMean->SetTextSize(0.036);
  legMean->AddEntry(gMeanOS, "OS", "lep");
  legMean->AddEntry(gMeanSS, "SS", "lep");
  legMean->Draw();

  cMean->SaveAs("mean_qT_vs_cut.pdf");
  cMean->SaveAs("mean_qT_vs_cut.png");

  // ---------- qTmax vs cut ----------
  double peakYmin = 1e9, peakYmax = -1e9;
  for (const auto& p : peakOS) {
    peakYmin = std::min(peakYmin, p.y - p.ey);
    peakYmax = std::max(peakYmax, p.y + p.ey);
  }
  for (const auto& p : peakSS) {
    peakYmin = std::min(peakYmin, p.y - p.ey);
    peakYmax = std::max(peakYmax, p.y + p.ey);
  }
  double peakPad = 0.15 * (peakYmax - peakYmin);
  peakYmin -= peakPad;
  peakYmax += peakPad;

  auto* cPeak = new TCanvas("cPeak_qT_vs_cut", "qT peak vs cut", 900, 700);
  gPeakOS->Draw("AP");
  styleAxes(gPeakOS,
            "cut [%]",
            "q_{T}^{max} [GeV]",
            -2.0, 72.0,
            peakYmin, peakYmax);

  gPeakOS->SetTitle("q_{T}^{max} vs pion momentum fraction cut");
  gPeakOS->Draw("AP");
  gPeakSS->Draw("P SAME");

  auto* legPeak = new TLegend(0.66, 0.76, 0.92, 0.90);
  legPeak->SetBorderSize(0);
  legPeak->SetFillStyle(0);
  legPeak->SetTextSize(0.036);
  legPeak->AddEntry(gPeakOS, "OS", "lep");
  legPeak->AddEntry(gPeakSS, "SS", "lep");
  legPeak->Draw();

  cPeak->SaveAs("qTmax_vs_cut.pdf");
  cPeak->SaveAs("qTmax_vs_cut.png");

  // optional ROOT output
  TFile fout("qT_observables_vs_cut.root", "RECREATE");
  gMeanOS->Write();
  gMeanSS->Write();
  gPeakOS->Write();
  gPeakSS->Write();
  cMean->Write();
  cPeak->Write();
  fout.Close();

  std::cout << "Saved mean_qT_vs_cut.pdf/png\n";
  std::cout << "Saved qTmax_vs_cut.pdf/png\n";
  std::cout << "Saved qT_observables_vs_cut.root\n";
}