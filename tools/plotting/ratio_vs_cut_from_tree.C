#include <vector>
#include <iostream>
#include <cmath>

#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TStyle.h"
#include "TString.h"

void ratio_vs_cut_from_tree(const char* inputFile = "output_100M.root",
                            const char* outputTag = "100M") {
  gStyle->SetOptStat(0);

  TFile* f = TFile::Open(inputFile, "READ");
  if (!f || f->IsZombie()) {
    std::cerr << "Could not open " << inputFile << "\n";
    return;
  }

  TTree* t = (TTree*)f->Get("tPionPairs");
  if (!t) {
    std::cerr << "Could not find tPionPairs\n";
    return;
  }

  std::vector<double> cuts;
  for (int c = 0; c <= 80; c += 5) cuts.push_back((double)c);

  std::vector<double> nOS(cuts.size(), 0.0);
  std::vector<double> nSS(cuts.size(), 0.0);

  double minFrac = 0.0;
  int isOS = 0;

  t->SetBranchAddress("minFrac", &minFrac);
  t->SetBranchAddress("isOS", &isOS);

  const Long64_t nEntries = t->GetEntries();

  for (Long64_t i = 0; i < nEntries; ++i) {
    t->GetEntry(i);

    for (size_t j = 0; j < cuts.size(); ++j) {
      double thr = cuts[j] / 100.0;
      if (minFrac >= thr) {
        if (isOS) nOS[j] += 1.0;
        else      nSS[j] += 1.0;
      } else {
        break;
      }
    }
  }

  std::vector<double> x(cuts.size()), y(cuts.size()), ex(cuts.size(), 0.0), ey(cuts.size(), 0.0);

  for (size_t j = 0; j < cuts.size(); ++j) {
    x[j] = cuts[j];

    if (nOS[j] > 0.0 && nSS[j] > 0.0) {
      y[j]  = nOS[j] / nSS[j];
      ey[j] = y[j] * std::sqrt(1.0 / nOS[j] + 1.0 / nSS[j]);
    } else {
      y[j] = 0.0;
      ey[j] = 0.0;
    }

    std::cout << "cut " << cuts[j]
              << "% : NOS = " << nOS[j]
              << " , NSS = " << nSS[j]
              << " , ratio = " << y[j]
              << " +- " << ey[j] << "\n";
  }

  TCanvas* c = new TCanvas("c_ratio_vs_cut_tree", "OS/SS ratio vs cut", 900, 700);

  TGraphErrors* g = new TGraphErrors((int)cuts.size(), x.data(), y.data(), ex.data(), ey.data());
  g->SetTitle("OS/SS ratio vs pion momentum fraction cut;cut [%];N_{OS}/N_{SS}");
  g->SetMarkerStyle(20);
  g->SetMarkerSize(0.9);
  g->SetLineWidth(1);
  g->Draw("AP");

  g->GetXaxis()->SetLimits(0, 85);
  g->GetYaxis()->SetRangeUser(0.5, 7.5);

  TString tag(outputTag);
  TString rootName = Form("ratio_vs_cut_from_tree_%s.root", tag.Data());
  TString pdfName = Form("ratio_vs_cut_from_tree_%s.pdf", tag.Data());
  TString pngName = Form("ratio_vs_cut_from_tree_%s.png", tag.Data());

  TFile fout(rootName, "RECREATE");
  g->Write("g_ratio_vs_cut");
  c->Write();
  fout.Close();

  c->SaveAs(pdfName);
  c->SaveAs(pngName);

  std::cout << "Saved " << rootName << ", " << pdfName << ", " << pngName << "\n";
}
