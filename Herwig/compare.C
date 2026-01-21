// compare.C  (ROOT macro)

TFile *fp = nullptr;
TFile *fh = nullptr;

int gCanvasCounter = 0;

TCanvas* MakeCanvas(const char* base, const char* title, int w=900, int h=600) {
  TString name = TString::Format("%s_%d", base, ++gCanvasCounter);
  return new TCanvas(name, title, w, h);
}

TH1* GetHistSafe(TFile* f, const char* hname, const char* tag) {
  if (!f) { std::cout << "ERROR: " << tag << " file pointer is null\n"; return nullptr; }
  auto h = (TH1*) f->Get(hname);
  if (!h) std::cout << "ERROR: missing " << tag << " histogram: " << hname << "\n";
  return h;
}

void compare(const char* pythia="pythia1.root",
             const char* herwig="Herwig/herwig1.root") {

  fp = TFile::Open(pythia);
  fh = TFile::Open(herwig);

  if (!fp || fp->IsZombie()) { std::cout << "Failed to open Pythia file: " << pythia << "\n"; return; }
  if (!fh || fh->IsZombie()) { std::cout << "Failed to open Herwig file: " << herwig << "\n"; return; }

  // histogram object names (must match what you wrote in both codes)
  const char* hC_OS = "h_combined_pT_closestToQuark_OS";
  const char* hC_SS = "h_combined_pT_closestToQuark_SS";
  const char* hH_OS = "h_combined_pT_highestPt_OS";
  const char* hH_SS = "h_combined_pT_highestPt_SS";

  new TBrowser();

  auto bar = new TControlBar("vertical","Pythia vs Herwig (4 hist)");

  // -------------------------
  // Individual plots
  // -------------------------
  bar->AddButton("Pythia: closest OS",
    "auto c=MakeCanvas(\"cP_cOS\",\"Pythia closest OS\");"
    "auto h=GetHistSafe(fp,\"h_combined_pT_closestToQuark_OS\",\"Pythia\");"
    "if(h){h->Draw(\"hist\"); gPad->Update();}");

  bar->AddButton("Pythia: closest SS",
    "auto c=MakeCanvas(\"cP_cSS\",\"Pythia closest SS\");"
    "auto h=GetHistSafe(fp,\"h_combined_pT_closestToQuark_SS\",\"Pythia\");"
    "if(h){h->Draw(\"hist\"); gPad->Update();}");

  bar->AddButton("Pythia: highest OS",
    "auto c=MakeCanvas(\"cP_hOS\",\"Pythia highest OS\");"
    "auto h=GetHistSafe(fp,\"h_combined_pT_highestPt_OS\",\"Pythia\");"
    "if(h){h->Draw(\"hist\"); gPad->Update();}");

  bar->AddButton("Pythia: highest SS",
    "auto c=MakeCanvas(\"cP_hSS\",\"Pythia highest SS\");"
    "auto h=GetHistSafe(fp,\"h_combined_pT_highestPt_SS\",\"Pythia\");"
    "if(h){h->Draw(\"hist\"); gPad->Update();}");

  bar->AddButton("Herwig: closest OS",
    "auto c=MakeCanvas(\"cH_cOS\",\"Herwig closest OS\");"
    "auto h=GetHistSafe(fh,\"h_combined_pT_closestToQuark_OS\",\"Herwig\");"
    "if(h){h->Draw(\"hist\"); gPad->Update();}");

  bar->AddButton("Herwig: closest SS",
    "auto c=MakeCanvas(\"cH_cSS\",\"Herwig closest SS\");"
    "auto h=GetHistSafe(fh,\"h_combined_pT_closestToQuark_SS\",\"Herwig\");"
    "if(h){h->Draw(\"hist\"); gPad->Update();}");

  bar->AddButton("Herwig: highest OS",
    "auto c=MakeCanvas(\"cH_hOS\",\"Herwig highest OS\");"
    "auto h=GetHistSafe(fh,\"h_combined_pT_highestPt_OS\",\"Herwig\");"
    "if(h){h->Draw(\"hist\"); gPad->Update();}");

  bar->AddButton("Herwig: highest SS",
    "auto c=MakeCanvas(\"cH_hSS\",\"Herwig highest SS\");"
    "auto h=GetHistSafe(fh,\"h_combined_pT_highestPt_SS\",\"Herwig\");"
    "if(h){h->Draw(\"hist\"); gPad->Update();}");

  // -------------------------
  // Overlay plots (4)
  // -------------------------
  bar->AddButton("Overlay: closest OS",
    "auto c=MakeCanvas(\"cO_cOS\",\"Overlay closest OS\");"
    "auto hp=GetHistSafe(fp,\"h_combined_pT_closestToQuark_OS\",\"Pythia\");"
    "auto hh=GetHistSafe(fh,\"h_combined_pT_closestToQuark_OS\",\"Herwig\");"
    "if(hp && hh){hp->SetStats(0);hh->SetStats(0);hp->SetLineWidth(2);hh->SetLineWidth(2);"
    "hp->SetLineColor(kBlue);hh->SetLineColor(kRed);"
    "hp->Draw(\"hist\");hh->Draw(\"hist same\");"
    "auto leg=new TLegend(0.65,0.75,0.88,0.88);"
    "leg->AddEntry(hp,\"Pythia\",\"l\");leg->AddEntry(hh,\"Herwig\",\"l\");leg->Draw(); gPad->Update();}");

  bar->AddButton("Overlay: closest SS",
    "auto c=MakeCanvas(\"cO_cSS\",\"Overlay closest SS\");"
    "auto hp=GetHistSafe(fp,\"h_combined_pT_closestToQuark_SS\",\"Pythia\");"
    "auto hh=GetHistSafe(fh,\"h_combined_pT_closestToQuark_SS\",\"Herwig\");"
    "if(hp && hh){hp->SetStats(0);hh->SetStats(0);hp->SetLineWidth(2);hh->SetLineWidth(2);"
    "hp->SetLineColor(kBlue);hh->SetLineColor(kRed);"
    "hp->Draw(\"hist\");hh->Draw(\"hist same\");"
    "auto leg=new TLegend(0.65,0.75,0.88,0.88);"
    "leg->AddEntry(hp,\"Pythia\",\"l\");leg->AddEntry(hh,\"Herwig\",\"l\");leg->Draw(); gPad->Update();}");

  bar->AddButton("Overlay: highest OS",
    "auto c=MakeCanvas(\"cO_hOS\",\"Overlay highest OS\");"
    "auto hp=GetHistSafe(fp,\"h_combined_pT_highestPt_OS\",\"Pythia\");"
    "auto hh=GetHistSafe(fh,\"h_combined_pT_highestPt_OS\",\"Herwig\");"
    "if(hp && hh){hp->SetStats(0);hh->SetStats(0);hp->SetLineWidth(2);hh->SetLineWidth(2);"
    "hp->SetLineColor(kBlue);hh->SetLineColor(kRed);"
    "hp->Draw(\"hist\");hh->Draw(\"hist same\");"
    "auto leg=new TLegend(0.65,0.75,0.88,0.88);"
    "leg->AddEntry(hp,\"Pythia\",\"l\");leg->AddEntry(hh,\"Herwig\",\"l\");leg->Draw(); gPad->Update();}");

  bar->AddButton("Overlay: highest SS",
    "auto c=MakeCanvas(\"cO_hSS\",\"Overlay highest SS\");"
    "auto hp=GetHistSafe(fp,\"h_combined_pT_highestPt_SS\",\"Pythia\");"
    "auto hh=GetHistSafe(fh,\"h_combined_pT_highestPt_SS\",\"Herwig\");"
    "if(hp && hh){hp->SetStats(0);hh->SetStats(0);hp->SetLineWidth(2);hh->SetLineWidth(2);"
    "hp->SetLineColor(kBlue);hh->SetLineColor(kRed);"
    "hp->Draw(\"hist\");hh->Draw(\"hist same\");"
    "auto leg=new TLegend(0.65,0.75,0.88,0.88);"
    "leg->AddEntry(hp,\"Pythia\",\"l\");leg->AddEntry(hh,\"Herwig\",\"l\");leg->Draw(); gPad->Update();}");

  // -------------------------
  // Side-by-side (4)
  // -------------------------
  bar->AddButton("Side-by-side: closest OS",
    "auto c=MakeCanvas(\"cS_cOS\",\"Side-by-side closest OS\",1100,450);"
    "c->Divide(2,1);"
    "c->cd(1); auto hp=GetHistSafe(fp,\"h_combined_pT_closestToQuark_OS\",\"Pythia\"); if(hp) hp->Draw(\"hist\");"
    "c->cd(2); auto hh=GetHistSafe(fh,\"h_combined_pT_closestToQuark_OS\",\"Herwig\"); if(hh) hh->Draw(\"hist\");"
    "gPad->Update();");

  bar->AddButton("Side-by-side: closest SS",
    "auto c=MakeCanvas(\"cS_cSS\",\"Side-by-side closest SS\",1100,450);"
    "c->Divide(2,1);"
    "c->cd(1); auto hp=GetHistSafe(fp,\"h_combined_pT_closestToQuark_SS\",\"Pythia\"); if(hp) hp->Draw(\"hist\");"
    "c->cd(2); auto hh=GetHistSafe(fh,\"h_combined_pT_closestToQuark_SS\",\"Herwig\"); if(hh) hh->Draw(\"hist\");"
    "gPad->Update();");

  bar->AddButton("Side-by-side: highest OS",
    "auto c=MakeCanvas(\"cS_hOS\",\"Side-by-side highest OS\",1100,450);"
    "c->Divide(2,1);"
    "c->cd(1); auto hp=GetHistSafe(fp,\"h_combined_pT_highestPt_OS\",\"Pythia\"); if(hp) hp->Draw(\"hist\");"
    "c->cd(2); auto hh=GetHistSafe(fh,\"h_combined_pT_highestPt_OS\",\"Herwig\"); if(hh) hh->Draw(\"hist\");"
    "gPad->Update();");

  bar->AddButton("Side-by-side: highest SS",
    "auto c=MakeCanvas(\"cS_hSS\",\"Side-by-side highest SS\",1100,450);"
    "c->Divide(2,1);"
    "c->cd(1); auto hp=GetHistSafe(fp,\"h_combined_pT_highestPt_SS\",\"Pythia\"); if(hp) hp->Draw(\"hist\");"
    "c->cd(2); auto hh=GetHistSafe(fh,\"h_combined_pT_highestPt_SS\",\"Herwig\"); if(hh) hh->Draw(\"hist\");"
    "gPad->Update();");

  bar->Show();
}
