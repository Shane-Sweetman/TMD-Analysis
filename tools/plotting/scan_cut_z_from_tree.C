
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
#include "TTree.h"
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

struct ComboResult {
  int cut = -1;
  double z = -1.0;
  std::string ztag;
  double scale = 0.0;
  double chi2OS = 0.0;
  double chi2SS = 0.0;
  int nOS = 0;
  int nSS = 0;
  double chi2PerNOS = 1e300;
  double chi2PerNSS = 1e300;
  double score = 1e300; // ranking among survivors
  bool beatsBoth = false;
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
                       double scale,
                       bool useOS,
                       double xMax,
                       double theoryFloorFrac = 1e-3) {
  Chi2Result out;

  double thPeak = theoryMaxScaled(th, scale, useOS, xMax);
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

void drawBestCell(TPad* cell,
                  TH1D* hOS,
                  TH1D* hSS,
                  const TheoryData& th,
                  int cut,
                  double z,
                  double sharedScale,
                  double xPlotMax,
                  double xRatioAxisMax) {
  cell->cd();
  cell->SetMargin(0, 0, 0, 0);

  auto pTop = new TPad("pTop_best", "", 0, 0.30, 1, 1);
  auto pBot = new TPad("pBot_best", "", 0, 0.00, 1, 0.30);

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

  auto frameTop = (TH1D*)hOS->Clone("frameTop_best");
  frameTop->Reset("ICES");
  frameTop->SetTitle(Form("Best combo   cut %d%%   z = %.2f", cut, z));
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

  auto frameBot = (TH1D*)hOS->Clone("frameBot_best");
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

std::string zToTag(double z) {
  int iz = (int)std::round(100 * z);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0p%02d", iz);
  return std::string(buf);
}

void scan_cut_z_from_tree() {
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);
  gStyle->SetEndErrorSize(1);

  const char* pythiaFile = "output_100M.root";
  const char* theoryDir  = "../epemTMD-main-Final/theory_zscan";
  const double xPlotMax = 10.0;
  const double xRatioAxisMax = 10.0;

  const int NBINS = 40;
  const double XMIN = 0.0;
  const double XMAX = 10.0;

  std::vector<int> cuts;
  for (int c = 5; c <= 90; c += 5) cuts.push_back(c);

  TFile* f = TFile::Open(pythiaFile, "READ");
  if (!f || f->IsZombie()) {
    std::cerr << "Could not open " << pythiaFile << "\n";
    return;
  }

  TTree* t = dynamic_cast<TTree*>(f->Get("tPionPairs"));
  if (!t) {
    std::cerr << "Could not find tPionPairs in " << pythiaFile << "\n";
    return;
  }

  double qT = 0.0;
  double minFrac = 0.0;
  int isOS = 0;

  t->SetBranchAddress("qT", &qT);
  t->SetBranchAddress("minFrac", &minFrac);
  t->SetBranchAddress("isOS", &isOS);

  std::map<int,TH1D*> hOS;
  std::map<int,TH1D*> hSS;

  for (int c : cuts) {
    hOS[c] = new TH1D(Form("hOS_cut%d_scan", c), "", NBINS, XMIN, XMAX);
    hSS[c] = new TH1D(Form("hSS_cut%d_scan", c), "", NBINS, XMIN, XMAX);
    hOS[c]->Sumw2();
    hSS[c]->Sumw2();
  }

  Long64_t nEntries = t->GetEntries();
  for (Long64_t i = 0; i < nEntries; ++i) {
    t->GetEntry(i);

    for (int c : cuts) {
      double thr = c / 100.0;
      if (minFrac >= thr) {
        if (isOS) hOS[c]->Fill(qT);
        else      hSS[c]->Fill(qT);
      }
    }
  }

  // Benchmark = 60% and z = 0.70
  TheoryData thBench;
  const std::string benchTag = "0p70";
  const std::string benchFile = std::string(theoryDir) + "/theory_z_" + benchTag + ".dat";
  if (!loadTheoryData(benchFile, thBench)) return;

  double thPeakBench = theoryPeakOS(thBench);
  if (thPeakBench <= 0.0) {
    std::cerr << "Benchmark theory OS peak is not positive.\n";
    return;
  }

  double benchScale = pythiaPeakOS(hOS[60], xPlotMax) / thPeakBench;
  Chi2Result benchOS = computeChi2(hOS[60], thBench, benchScale, true,  xPlotMax);
  Chi2Result benchSS = computeChi2(hSS[60], thBench, benchScale, false, xPlotMax);

  double benchOSPerN = (benchOS.nPoints > 0 ? benchOS.chi2 / benchOS.nPoints : 1e300);
  double benchSSPerN = (benchSS.nPoints > 0 ? benchSS.chi2 / benchSS.nPoints : 1e300);

  std::vector<ComboResult> survivors;
  ComboResult bestOverall;

  for (int zi = 5; zi <= 95; zi += 5) {
    double z = zi / 100.0;
    std::string ztag = zToTag(z);
    std::string theoryFile = std::string(theoryDir) + "/theory_z_" + ztag + ".dat";

    TheoryData th;
    if (!loadTheoryData(theoryFile, th)) {
      std::cerr << "Skipping missing theory file: " << theoryFile << "\n";
      continue;
    }

    double thPeakOS = theoryPeakOS(th);
    if (thPeakOS <= 0.0) continue;

    for (int c : cuts) {
      ComboResult r;
      r.cut = c;
      r.z = z;
      r.ztag = ztag;
      r.scale = pythiaPeakOS(hOS[c], xPlotMax) / thPeakOS;

      Chi2Result osRes = computeChi2(hOS[c], th, r.scale, true,  xPlotMax);
      Chi2Result ssRes = computeChi2(hSS[c], th, r.scale, false, xPlotMax);

      r.chi2OS = osRes.chi2;
      r.chi2SS = ssRes.chi2;
      r.nOS = osRes.nPoints;
      r.nSS = ssRes.nPoints;
      r.chi2PerNOS = (r.nOS > 0 ? r.chi2OS / r.nOS : 1e300);
      r.chi2PerNSS = (r.nSS > 0 ? r.chi2SS / r.nSS : 1e300);
      r.score = r.chi2PerNOS + r.chi2PerNSS;

      r.beatsBoth = (r.chi2PerNOS < benchOSPerN && r.chi2PerNSS < benchSSPerN);

      if (bestOverall.score > r.score) bestOverall = r;
      if (r.beatsBoth) survivors.push_back(r);
    }
  }

  std::sort(survivors.begin(), survivors.end(),
            [](const ComboResult& a, const ComboResult& b) {
              if (a.score != b.score) return a.score < b.score;
              if (a.cut != b.cut) return a.cut < b.cut;
              return a.z < b.z;
            });

  TDatime now;
  TString tag = Form("%04d%02d%02d_%02d%02d%02d",
                     now.GetYear(), now.GetMonth(), now.GetDay(),
                     now.GetHour(), now.GetMinute(), now.GetSecond());

  TString txtName = Form("scan_cut_z_better_than_60_0p70_%s.txt", tag.Data());
  std::ofstream foutTxt(txtName.Data());

  foutTxt << "Benchmark combo: cut 60%, z = 0.70\n";
  foutTxt << "Shared peak scale benchmark\n";
  foutTxt << "OS : chi2 = " << benchOS.chi2 << ", N = " << benchOS.nPoints
          << ", chi2/N = " << benchOSPerN << "\n";
  foutTxt << "SS : chi2 = " << benchSS.chi2 << ", N = " << benchSS.nPoints
          << ", chi2/N = " << benchSSPerN << "\n\n";

  if (survivors.empty()) {
    foutTxt << "No combinations beat the benchmark for both OS and SS.\n";
  } else {
    foutTxt << "Only combinations with BOTH\n";
    foutTxt << "  chi2/N(OS) < " << benchOSPerN << "\n";
    foutTxt << "and\n";
    foutTxt << "  chi2/N(SS) < " << benchSSPerN << "\n";
    foutTxt << "are listed below.\n\n";

    foutTxt << "cut(%)   z      scale        chi2_OS      N_OS   chi2/N_OS    chi2_SS      N_SS   chi2/N_SS    score\n";
    foutTxt << "-----------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : survivors) {
      foutTxt << Form("%3d    %4.2f   %11.5f   %11.4f   %3d   %10.4f   %11.4f   %3d   %10.4f   %10.4f\n",
                      r.cut, r.z, r.scale, r.chi2OS, r.nOS, r.chi2PerNOS,
                      r.chi2SS, r.nSS, r.chi2PerNSS, r.score);
    }
  }
  foutTxt.close();

  ComboResult chosen = survivors.empty() ? bestOverall : survivors.front();
  std::string chosenTheoryFile = std::string(theoryDir) + "/theory_z_" + chosen.ztag + ".dat";
  TheoryData chosenTheory;
  if (!loadTheoryData(chosenTheoryFile, chosenTheory)) {
    std::cerr << "Could not load chosen theory file: " << chosenTheoryFile << "\n";
    return;
  }

  TString baseName;
  if (survivors.empty())
    baseName = Form("best_overall_cut%d_z%s_%s", chosen.cut, chosen.ztag.c_str(), tag.Data());
  else
    baseName = Form("best_survivor_cut%d_z%s_%s", chosen.cut, chosen.ztag.c_str(), tag.Data());

  auto cBest = new TCanvas("cBestCombo", "Best combo", 900, 700);
  drawBestCell((TPad*)cBest, hOS[chosen.cut], hSS[chosen.cut], chosenTheory,
               chosen.cut, chosen.z, chosen.scale, xPlotMax, xRatioAxisMax);

  TString rootName = baseName + ".root";
  TString pdfName  = baseName + ".pdf";
  TString pngName  = baseName + ".png";

  TFile fout(rootName, "RECREATE");
  cBest->Write();
  fout.Close();

  cBest->SaveAs(pdfName);
  cBest->SaveAs(pngName);

  std::cout << "Benchmark 60% / z=0.70\n";
  std::cout << "  OS : chi2 = " << benchOS.chi2 << ", N = " << benchOS.nPoints
            << ", chi2/N = " << benchOSPerN << "\n";
  std::cout << "  SS : chi2 = " << benchSS.chi2 << ", N = " << benchSS.nPoints
            << ", chi2/N = " << benchSSPerN << "\n\n";

  if (survivors.empty()) {
    std::cout << "No survivor beat the benchmark for both OS and SS.\n";
    std::cout << "Best overall score used for plotting instead:\n";
  } else {
    std::cout << "Found " << survivors.size() << " survivor combinations.\n";
    std::cout << "Best survivor used for plotting:\n";
  }

  std::cout << "  cut = " << chosen.cut << "%, z = " << chosen.z
            << ", scale = " << chosen.scale << "\n";
  std::cout << "  OS : chi2 = " << chosen.chi2OS << ", N = " << chosen.nOS
            << ", chi2/N = " << chosen.chi2PerNOS << "\n";
  std::cout << "  SS : chi2 = " << chosen.chi2SS << ", N = " << chosen.nSS
            << ", chi2/N = " << chosen.chi2PerNSS << "\n";
  std::cout << "Saved:\n";
  std::cout << "  " << txtName << "\n";
  std::cout << "  " << rootName << "\n";
  std::cout << "  " << pdfName << "\n";
  std::cout << "  " << pngName << "\n";
}
