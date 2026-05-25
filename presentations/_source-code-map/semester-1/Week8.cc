// TMD.cc
// e+ e- -> qqbar, hadron-pair & jet-based TMD-like observables with FastJet (anti-kt).
// Extended with: jet observables + leading hadron ancestry tracing + improved jet diagnostics

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
#include <set>

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"

// Pythia
#include "Pythia8/Pythia.h"

// FastJet
#include "fastjet/ClusterSequence.hh"

using namespace Pythia8;
using namespace fastjet;

// ---------- Ancestry Tracing ----------
struct AncestryInfo {
    int motherQuarkFlavor;  // 1-5 for d,u,s,c,b; 0=unknown
    int resonanceID;         // PDG ID of intermediate resonance (0 if none)
    bool fromPrimaryQuark;   // true if traceable to hard process quark
    std::vector<int> chain;  // full decay chain indices
};

// Trace hadron back to find mother quark and resonances
AncestryInfo traceAncestry(const Event &event, int hadronIdx) { 
    AncestryInfo info;
    info.motherQuarkFlavor = 0;
    info.resonanceID = 0;
    info.fromPrimaryQuark = false;
    
    int current = hadronIdx;
    std::set<int> visited;
    
    while (current > 0 && visited.find(current) == visited.end()) {
        visited.insert(current);
        info.chain.push_back(current);
        
        int pdg = std::abs(event[current].id());
        int status = event[current].status();
        
        if (pdg >= 1 && pdg <= 5) {
            info.motherQuarkFlavor = pdg;
            if (status == 23 || status == 21 || status == 22) {
                info.fromPrimaryQuark = true;
            }
            break;
        }
        
        if ((pdg >= 100 && pdg < 1000) || (pdg >= 1000 && pdg < 10000)) {
            if (info.resonanceID == 0) {
                info.resonanceID = event[current].id();
            }
        }
        
        int mother1 = event[current].mother1();
        if (mother1 > 0 && mother1 < event.size()) {
            current = mother1;
        } else {
            break;
        }
    }
    
    return info;
}

// ---------- Event Shape Calculations ----------
struct EventShapes {
    double thrust; 
    double sphericity;
    double circularity;
};

double calculateThrust(const std::vector<PseudoJet> &particles) {
    if (particles.empty()) return 0.0;
    
    double totalP = 0.0;
    for (const auto &p : particles) {
        totalP += std::sqrt(p.px()*p.px() + p.py()*p.py() + p.pz()*p.pz());
    }
    if (totalP <= 0.0) return 0.0;
    
    double maxThrust = 0.0;
    int nSamples = 100;
    for (int i = 0; i < nSamples; ++i) {
        double theta = M_PI * i / nSamples; 
        for (int j = 0; j < nSamples; ++j) {
            double phi = 2.0 * M_PI * j / nSamples; 
            double nx = std::sin(theta) * std::cos(phi);
            double ny = std::sin(theta) * std::sin(phi);
            double nz = std::cos(theta);
            
            double sum = 0.0;
            for (const auto &p : particles) {
                double dot = std::abs(p.px()*nx + p.py()*ny + p.pz()*nz);
                sum += dot;
            }
            double thrust = sum / totalP;
            if (thrust > maxThrust) maxThrust = thrust;
        }
    }
    return maxThrust;
}

EventShapes calculateEventShapes(const std::vector<PseudoJet> &particles) {
    EventShapes shapes = {0.0, 0.0, 0.0};
    if (particles.empty()) return shapes;
    
    double S[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double totalP2 = 0.0;
    
    for (const auto &p : particles) {
        double px = p.px(), py = p.py(), pz = p.pz();
        double p2 = px*px + py*py + pz*pz;
        totalP2 += p2;
        
        S[0][0] += px*px; 
        S[0][1] += px*py; S[1][0] += px*py;
        S[0][2] += px*pz; S[2][0] += px*pz;
        S[1][1] += py*py;
        S[1][2] += py*pz; S[2][1] += py*pz;
        S[2][2] += pz*pz;
    }
    
    if (totalP2 <= 0.0) return shapes;
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S[i][j] /= totalP2;
        }
    }
    
    TMatrixDSym matrix(3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix(i,j) = S[i][j];
        }
    }
    
    TMatrixDSymEigen eigen(matrix);
    TVectorD eigenvalues = eigen.GetEigenValues();
    
    std::vector<double> eigs = {eigenvalues[0], eigenvalues[1], eigenvalues[2]};
    std::sort(eigs.begin(), eigs.end(), std::greater<double>());
    
    double lambda2 = eigs[1];
    double lambda3 = eigs[2];
    
    shapes.sphericity = 1.5 * (lambda2 + lambda3);
    shapes.thrust = calculateThrust(particles);
    
    // Circularity (2D transverse)
    double S2D[2][2] = {{0,0},{0,0}};
    double totalPT2 = 0.0;
    for (const auto &p : particles) {
        double px = p.px(), py = p.py();
        double pt2 = px*px + py*py;
        totalPT2 += pt2;
        S2D[0][0] += px*px;
        S2D[0][1] += px*py; S2D[1][0] += px*py;
        S2D[1][1] += py*py;
    }
    
    if (totalPT2 > 0.0) {
        S2D[0][0] /= totalPT2; 
        S2D[0][1] /= totalPT2; 
        S2D[1][0] /= totalPT2;
        S2D[1][1] /= totalPT2;
        
        double trace = S2D[0][0] + S2D[1][1];
        double det = S2D[0][0]*S2D[1][1] - S2D[0][1]*S2D[1][0];
        double discriminant = trace*trace - 4*det;
        if (discriminant >= 0) {
            double sqrtDisc = std::sqrt(discriminant);
            double eig1 = (trace + sqrtDisc) / 2.0;
            double eig2 = (trace - sqrtDisc) / 2.0;
            double minEig = std::min(eig1, eig2);
            double sumEig = eig1 + eig2;
            if (sumEig > 0) {
                shapes.circularity = 2.0 * minEig / sumEig;
            }
        }
    }
    
    return shapes;
}

// ----------- histogram map ----------
static std::map<std::string, TH1D*> hmap;

TH1D* getHist1D(const std::string &name, const std::string &title,
                int nbins = 64, double xmin = -3.2, double xmax = 3.2) {
    auto it = hmap.find(name);
    if (it != hmap.end()) return it->second;
    TH1D *h = new TH1D(name.c_str(), title.c_str(), nbins, xmin, xmax);
    h->SetDirectory(nullptr);
    hmap[name] = h;
    return h;
}

#include <iomanip>
#include <limits>

// --- Pretty, wide, aligned event dump ---
static void printEventTableWide(const Pythia8::Event& ev, int ievt,
                                int startRow, int rowsPerPage) {
    using std::cout; using std::left; using std::right; using std::setw; using std::setprecision; using std::fixed;

    const int N = ev.size();
    const int endRow = std::min(N - 1, startRow + rowsPerPage - 1);

    const int W_ROW   = 6;
    const int W_EVT   = 7;
    const int W_SIZE  = 6;
    const int W_NO    = 6;
    const int W_ID    = 8;
    const int W_NAME  = 14;
    const int W_ST    = 6;
    const int W_M1    = 6;
    const int W_M2    = 6;
    const int W_D1    = 6;
    const int W_D2    = 6;
    const int W_P     = 13;

    auto sep = [&](){
        cout
          << std::string(W_ROW,   '-') << "+"
          << std::string(W_EVT,   '-') << "+"
          << std::string(W_SIZE,  '-') << "+"
          << std::string(W_NO,    '-') << "+"
          << std::string(W_ID,    '-') << "+"
          << std::string(W_NAME,  '-') << "+"
          << std::string(W_ST,    '-') << "+"
          << std::string(W_M1+1+W_M2, '-') << "+"
          << std::string(W_D1+1+W_D2, '-') << "+"
          << std::string(W_P,     '-') << "+"
          << std::string(W_P,     '-') << "+"
          << std::string(W_P,     '-') << "+"
          << std::string(W_P,     '-') << "\n";
    };

    cout << "\n=== EVENT " << ievt << " ===\n\n";
    cout << left
         << setw(W_ROW)  << "row"  << "|"
         << setw(W_EVT)  << "event"<< "|"
         << setw(W_SIZE) << "size" << "|"
         << setw(W_NO)   << "no"   << "|"
         << setw(W_ID)   << "id"   << "|"
         << setw(W_NAME) << "name" << "|"
         << setw(W_ST)   << "st"   << "|"
         << setw(W_M1)   << "m1"   << " " << setw(W_M2) << "m2" << "|"
         << setw(W_D1)   << "d1"   << " " << setw(W_D2) << "d2" << "|"
         << setw(W_P)    << "px"   << "|"
         << setw(W_P)    << "py"   << "|"
         << setw(W_P)    << "pz"   << "|"
         << setw(W_P)    << "E"    << "\n";
    sep();

    cout << fixed << setprecision(3);

    for (int i = startRow; i <= endRow; ++i) {
        const auto& p = ev[i];
        const int pid = p.id();
        std::string pname;
        switch (pid) {
            case 90: pname = "system"; break;
            case 22: pname = "gamma"; break;
            case 23: pname = "Z0"; break;
            case 11: pname = "e-"; break;
            case -11: pname = "e+"; break;
            case 13: pname = "mu-"; break;
            case -13: pname = "mu+"; break;
            case 14: pname = "nu_mu"; break;
            case -14: pname = "nu_mubar"; break;
            case 21: pname = "g"; break;
            case 1: pname = "d"; break;
            case -1: pname = "dbar"; break;
            case 111: pname = "pi0"; break;
            case 211: pname = "pi+"; break;
            case -211: pname = "pi-"; break;
            case 321: pname = "K+"; break;
            case -321: pname = "K-"; break;
            case 113: pname = "rho0"; break;
            case -213: pname = "rho-"; break;
            case 213: pname = "rho+"; break;
            case 2212: pname = "p+"; break;
            case -2212: pname = "pbar"; break;
            default: {
                std::ostringstream os;
                os << "id" << pid;
                pname = os.str();
            }
        }

        int d1 = p.daughter1();
        int d2 = p.daughter2();

        cout << left
             << setw(W_ROW)  << i         << "|"
             << setw(W_EVT)  << ievt      << "|"
             << setw(W_SIZE) << N         << "|"
             << setw(W_NO)   << p.index() << "|"
             << setw(W_ID)   << pid       << "|"
             << setw(W_NAME) << pname     << "|"
             << setw(W_ST)   << p.status()<< "|"
             << setw(W_M1)   << p.mother1()<< " " << setw(W_M2) << p.mother2() << "|"
             << setw(W_D1)   << d1        << " " << setw(W_D2) << d2           << "|"
             << right
             << setw(W_P)    << p.px()    << "|"
             << setw(W_P)    << p.py()    << "|"
             << setw(W_P)    << p.pz()    << "|"
             << setw(W_P)    << p.e()     << "\n";
    }

    std::cout << "(showing rows " << startRow << " to " << endRow << " of " << N << " total)\n";
}

// Simple pager
char promptPager(const char* msg = "More") {
    std::cout << msg << " [Enter=+10, n=next event, q=quit]: " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) return 'q';
    if (line == "q" || line == "Q") return 'q';
    if (line == "n" || line == "N") return 'n';
    return 'c';
}

int main(int argc, char* argv[]) {
    
    // ========== INTERACTIVE MODE TOGGLE ==========
    bool INTERACTIVE_MODE = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--interactive" || arg == "-i") {
            INTERACTIVE_MODE = true;
            std::cout << "\n*** INTERACTIVE MODE ENABLED ***\n";
            std::cout << "You can browse event tables interactively.\n\n";
        }
    }
    
    TFile *fout = new TFile("ee_hadron_corr.root", "RECREATE");

    // Jet-based observables
    TH1D *h_jetMult     = new TH1D("h_jetMult", "Jet multiplicity per event;N_{jets};Entries", 20, 0, 20);
    TH1D *h_jetPt       = new TH1D("h_jetPt", "Jet pT spectrum;p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_jetEta      = new TH1D("h_jetEta", "Jet pseudorapidity;#eta;Entries", 100, -5, 5);
    TH1D *h_jetRapidity = new TH1D("h_jetRapidity", "Jet rapidity;y;Entries", 100, -5, 5);
    TH1D *h_jetDeltaPhi = new TH1D("h_jetDeltaPhi", "Delta phi between leading jets;#Delta#phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_jetConstMult = new TH1D("h_jetConstMult", "Jet constituent multiplicity;N_{constituents};Entries", 100, 0, 100);
    
    // Event shapes
    TH1D *h_thrust      = new TH1D("h_thrust", "Event thrust;T;Entries", 100, 0, 1);
    TH1D *h_sphericity  = new TH1D("h_sphericity", "Event sphericity;S;Entries", 100, 0, 1);
    TH1D *h_circularity = new TH1D("h_circularity", "Event circularity;C;Entries", 100, 0, 1);

    // Leading hadron observables 
    TH1D *h_leadPi_pT = new TH1D("h_leadPi_pT", "Leading pion pT;p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_leadPi_eta = new TH1D("h_leadPi_eta", "Leading pion eta;#eta;Entries", 100, -5, 5);
    TH1D *h_leadPi_deltaPhi = new TH1D("h_leadPi_deltaPhi", "Delta phi between leading pions;#Delta#phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_leadK_pT = new TH1D("h_leadK_pT", "Leading kaon pT;p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_leadK_deltaPhi = new TH1D("h_leadK_deltaPhi", "Delta phi between leading kaons;#Delta#phi [rad];Entries", 64, -3.2, 3.2);

    // JET-ONLY HISTOGRAMS
    TH1D *h_leadPi_pT_inJet = new TH1D("h_leadPi_pT_inJet", "Leading pion pT (in jets);p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_leadPi_eta_inJet = new TH1D("h_leadPi_eta_inJet", "Leading pion eta (in jets);#eta;Entries", 100, -5, 5);
    TH1D *h_leadPi_deltaPhi_inJet = new TH1D("h_leadPi_deltaPhi_inJet", "Delta phi between leading pions (in jets);#Delta#phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_leadK_pT_inJet = new TH1D("h_leadK_pT_inJet", "Leading kaon pT (in jets);p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_leadK_deltaPhi_inJet = new TH1D("h_leadK_deltaPhi_inJet", "Delta phi between leading kaons (in jets);#Delta#phi [rad];Entries", 64, -3.2, 3.2);

    // DIAGNOSTIC HISTOGRAMS
    TH1D *h_jet_nJets_dist = new TH1D("h_jet_nJets_dist", "Detailed N_{jets} distribution;N_{jets};Events", 20, 0, 20);
    TH1D *h_jet_leadingPt = new TH1D("h_jet_leadingPt", "Leading jet p_{T};p_{T} [GeV];Events", 100, 0, 50);
    TH1D *h_jet_subleadingPt = new TH1D("h_jet_subleadingPt", "Subleading jet p_{T};p_{T} [GeV];Events", 100, 0, 50);
    TH1D *h_jet_deltaPhi_2jetOnly = new TH1D("h_jet_deltaPhi_2jetOnly", "#Delta#phi (exactly 2 jets);#Delta#phi [rad];Events", 64, -3.2, 3.2);
    TH1D *h_jet_deltaPhi_multijet = new TH1D("h_jet_deltaPhi_multijet", "#Delta#phi (3+ jets);#Delta#phi [rad];Events", 64, -3.2, 3.2);
    TH2D *h2_jet_phi1_vs_phi2 = new TH2D("h2_jet_phi1_vs_phi2", "Leading vs subleading jet #phi;#phi_{1} [rad];#phi_{2} [rad]", 64, -3.2, 3.2, 64, -3.2, 3.2);

    h_leadPi_pT_inJet->SetDirectory(nullptr);
    h_leadPi_eta_inJet->SetDirectory(nullptr);
    h_leadPi_deltaPhi_inJet->SetDirectory(nullptr);
    h_leadK_pT_inJet->SetDirectory(nullptr);
    h_leadK_deltaPhi_inJet->SetDirectory(nullptr);
    h_jet_nJets_dist->SetDirectory(nullptr);
    h_jet_leadingPt->SetDirectory(nullptr);
    h_jet_subleadingPt->SetDirectory(nullptr);
    h_jet_deltaPhi_2jetOnly->SetDirectory(nullptr);
    h_jet_deltaPhi_multijet->SetDirectory(nullptr);
    h2_jet_phi1_vs_phi2->SetDirectory(nullptr);

    h_jetMult->SetDirectory(nullptr); 
    h_jetPt->SetDirectory(nullptr);
    h_jetEta->SetDirectory(nullptr);
    h_jetRapidity->SetDirectory(nullptr);
    h_jetDeltaPhi->SetDirectory(nullptr);
    h_jetConstMult->SetDirectory(nullptr);
    h_thrust->SetDirectory(nullptr);
    h_sphericity->SetDirectory(nullptr);
    h_circularity->SetDirectory(nullptr);
    h_leadPi_pT->SetDirectory(nullptr);
    h_leadPi_eta->SetDirectory(nullptr);
    h_leadPi_deltaPhi->SetDirectory(nullptr);
    h_leadK_pT->SetDirectory(nullptr);
    h_leadK_deltaPhi->SetDirectory(nullptr);

    // NEW FEATURE 1: Thrust vs Jet Radius R
    static const double Rgrid[] = {0.2, 0.4, 0.6, 0.8, 1.0};
    static const int NR = sizeof(Rgrid)/sizeof(Rgrid[0]);
    const double jetPtMin_for_Rscan = 1.0;

    std::vector<TH1D*> h_thrust_R;
    for (int iR = 0; iR < NR; ++iR) {
        std::ostringstream oss;
        oss << "h_thrust_R" << std::fixed << std::setprecision(1) << Rgrid[iR];
        std::string hname = oss.str();
        std::ostringstream title_oss;
        title_oss << "Thrust (R=" << std::fixed << std::setprecision(1) << Rgrid[iR] << ");T;Entries";
        TH1D *h = new TH1D(hname.c_str(), title_oss.str().c_str(), 100, 0, 1);
        h->SetDirectory(nullptr);
        h_thrust_R.push_back(h);
    }
    std::vector<double> sumT_R(NR, 0.0);
    std::vector<double> cntT_R(NR, 0.0);

    // NEW FEATURE 2: 2D Njets vs Thrust
    TH2D* h2_Njets_vs_Thrust = new TH2D("h2_Njets_vs_Thrust",
        "N_{jets} vs Thrust;Thrust;N_{jets}", 100, 0, 1, 21, -0.5, 20.5);
    h2_Njets_vs_Thrust->SetDirectory(nullptr);

    // NEW FEATURE 3: Delta pT (jet - pion)
    TH1D* h_dpt_jet_minus_pion = new TH1D("h_dpt_jet_minus_pion",
        "#Delta p_{T} (jet - pion);#Delta p_{T} [GeV];Entries", 100, -10, 50);
    h_dpt_jet_minus_pion->SetDirectory(nullptr);

    TH2D* h2_dptJetPi_vs_pionPt = new TH2D("h2_dptJetPi_vs_pionPt",
        "#Delta p_{T} (jet-pion) vs pion p_{T};pion p_{T} [GeV];#Delta p_{T} [GeV]",
        100, 0, 50, 100, -10, 50);
    h2_dptJetPi_vs_pionPt->SetDirectory(nullptr);

    // NEW FEATURE 4: Delta pT (pion - parent)
    TH1D* h_dpt_pion_minus_parent = new TH1D("h_dpt_pion_minus_parent",
        "#Delta p_{T} (pion - parent);#Delta p_{T} [GeV];Entries", 100, -20, 20);
    h_dpt_pion_minus_parent->SetDirectory(nullptr);

    TH1D* h_dpt_pion_minus_parent_rhoOmegaPhiKst = new TH1D("h_dpt_pion_minus_parent_rhoOmegaPhiKst",
        "#Delta p_{T} (pion - parent) [#rho,#omega,#phi,K*];#Delta p_{T} [GeV];Entries", 100, -20, 20);
    h_dpt_pion_minus_parent_rhoOmegaPhiKst->SetDirectory(nullptr);

    TH1D* h_dpt_pion_minus_parent_other = new TH1D("h_dpt_pion_minus_parent_other",
        "#Delta p_{T} (pion - parent) [other];#Delta p_{T} [GeV];Entries", 100, -20, 20);
    h_dpt_pion_minus_parent_other->SetDirectory(nullptr);

    TH2D* h2_ratioPiOverParent_vs_pionPt = new TH2D("h2_ratioPiOverParent_vs_pionPt",
        "p_{T}(pion)/p_{T}(parent) vs pion p_{T};pion p_{T} [GeV];p_{T} ratio",
        100, 0, 50, 100, 0, 2);
    h2_ratioPiOverParent_vs_pionPt->SetDirectory(nullptr);

    TH2D* h2_dR_PiParent_vs_pionPt = new TH2D("h2_dR_PiParent_vs_pionPt",
        "#DeltaR(pion,parent) vs pion p_{T};pion p_{T} [GeV];#DeltaR",
        100, 0, 50, 100, 0, 5);
    h2_dR_PiParent_vs_pionPt->SetDirectory(nullptr);

    // Pythia setup
    Pythia pythia;
    pythia.readString("Beams:idA = -11");
    pythia.readString("Beams:idB = 11");
    pythia.readString("Beams:eCM = 91.2");
    pythia.readString("PDF:lepton = off");
    pythia.readString("HadronLevel:all = on");
    pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = 123456788");

    if (!pythia.init()) {
        std::cerr << "Pythia initialization failed\n";
        return 1;
    }

    const int nEvents = 20000;
    int eventsWithLeadingPions = 0;
    int eventsWithLeadingKaons = 0;

    // FastJet parameters - IMPROVED VALUES
    double R = 0.4;  // Reduced from 0.6
    double jetPtMin = 5.0;  // Increased from 1.0

    bool quitAll = false;

    // Event loop
    for (int ievt = 0; ievt < nEvents; ++ievt) {
        if (!pythia.next()) continue;

        // INTERACTIVE EVENT TABLE
        if (INTERACTIVE_MODE) {
            const int PAGE = 10;
            int startRow = 0;

            printEventTableWide(pythia.event, ievt, startRow, PAGE);
            startRow += PAGE;

            while (true) {
                char cmd = promptPager();
                if (cmd == 'q') {
                    quitAll = true;
                    break;
                } else if (cmd == 'n') {
                    break;
                } else {
                    printEventTableWide(pythia.event, ievt, startRow, PAGE);
                    startRow += PAGE;
                    if (startRow >= pythia.event.size()) {
                        char atEnd = promptPager("(end of event) Next");
                        if (atEnd == 'q') quitAll = true;
                        break;
                    }
                }
            }
            if (quitAll) break;
        }

        // Progress reporting
        if (!INTERACTIVE_MODE && (ievt + 1) % 1000 == 0) {
            std::cout << "Processed " << (ievt + 1) << " events..." << std::endl;
        }

        // Collect final-state charged hadrons
        std::vector<int> hadrons;
        hadrons.reserve(64);
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            if (!pythia.event[i].isCharged()) continue;
            hadrons.push_back(i);
        }

        // LEADING HADRON ANALYSIS
        std::vector<std::pair<double, int>> pions;
        for (int idx : hadrons) {
            int pdg = std::abs(pythia.event[idx].id());
            if (pdg == 211) {
                double p = pythia.event[idx].pAbs();
                pions.push_back({p, idx});
            }
        }
        std::sort(pions.begin(), pions.end(), std::greater<std::pair<double,int>>());
        
        std::vector<std::pair<double, int>> kaons;
        for (int idx : hadrons) {
            int pdg = std::abs(pythia.event[idx].id());
            if (pdg == 321) {
                double p = pythia.event[idx].pAbs();
                kaons.push_back({p, idx});
            }
        }
        std::sort(kaons.begin(), kaons.end(), std::greater<std::pair<double,int>>());

        // Analyze leading two pions
        if (pions.size() >= 2) {
            ++eventsWithLeadingPions;
            int pi1_idx = pions[0].second;
            int pi2_idx = pions[1].second;
            
            AncestryInfo anc1 = traceAncestry(pythia.event, pi1_idx);
            AncestryInfo anc2 = traceAncestry(pythia.event, pi2_idx);
            
            double pT1 = pythia.event[pi1_idx].pT();
            double pT2 = pythia.event[pi2_idx].pT();
            double eta1 = pythia.event[pi1_idx].eta();
            double eta2 = pythia.event[pi2_idx].eta();
            double phi1 = pythia.event[pi1_idx].phi();
            double phi2 = pythia.event[pi2_idx].phi();
            
            h_leadPi_pT->Fill(pT1);
            h_leadPi_pT->Fill(pT2);
            h_leadPi_eta->Fill(eta1);
            h_leadPi_eta->Fill(eta2);
            
            double dPhi = phi1 - phi2;
            if (dPhi > M_PI) dPhi -= 2.0*M_PI;
            if (dPhi < -M_PI) dPhi += 2.0*M_PI;
            h_leadPi_deltaPhi->Fill(dPhi);
        }
        
        // Analyze leading two kaons
        if (kaons.size() >= 2) {
            ++eventsWithLeadingKaons;
            int k1_idx = kaons[0].second;
            int k2_idx = kaons[1].second;
            
            AncestryInfo anc1 = traceAncestry(pythia.event, k1_idx);
            AncestryInfo anc2 = traceAncestry(pythia.event, k2_idx);
            
            double pT1 = pythia.event[k1_idx].pT();
            double pT2 = pythia.event[k2_idx].pT();
            double phi1 = pythia.event[k1_idx].phi();
            double phi2 = pythia.event[k2_idx].phi();
            
            h_leadK_pT->Fill(pT1);
            h_leadK_pT->Fill(pT2);
            
            double dPhi = phi1 - phi2;
            if (dPhi > M_PI) dPhi -= 2.0*M_PI;
            if (dPhi < -M_PI) dPhi += 2.0*M_PI;
            h_leadK_deltaPhi->Fill(dPhi);
        }

        // JET-BASED OBSERVABLES
        if (!hadrons.empty()) {
            std::vector<PseudoJet> fjInputs;
            fjInputs.reserve(hadrons.size());
            for (int idx : hadrons) {
                double px = pythia.event[idx].px();
                double py = pythia.event[idx].py();
                double pz = pythia.event[idx].pz();
                double E  = pythia.event[idx].e();
                PseudoJet pj(px, py, pz, E);
                pj.set_user_index(idx);
                fjInputs.push_back(pj);
            }

            JetDefinition jetDef(antikt_algorithm, R);
            ClusterSequence cs(fjInputs, jetDef);
            std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));

            int nJets = jets.size();
            h_jetMult->Fill(nJets);
            h_jet_nJets_dist->Fill(nJets);

            EventShapes shapes = calculateEventShapes(fjInputs);
            h_thrust->Fill(shapes.thrust);
            h_sphericity->Fill(shapes.sphericity);
            h_circularity->Fill(shapes.circularity);
            
            // Feature 1: Thrust vs R scan
            for (int iR = 0; iR < NR; ++iR) {
                double R_scan = Rgrid[iR];
                JetDefinition jetDef_scan(antikt_algorithm, R_scan);
                ClusterSequence cs_scan(fjInputs, jetDef_scan);
                std::vector<PseudoJet> jets_scan = sorted_by_pt(cs_scan.inclusive_jets(jetPtMin_for_Rscan));
                
                std::vector<PseudoJet> jet_constituents;
                for (const PseudoJet &j : jets_scan) {
                    std::vector<PseudoJet> consts = j.constituents();
                    jet_constituents.insert(jet_constituents.end(), consts.begin(), consts.end());
                }
                
                if (!jet_constituents.empty()) {
                    EventShapes shapes_R = calculateEventShapes(jet_constituents);
                    h_thrust_R[iR]->Fill(shapes_R.thrust);
                    sumT_R[iR] += shapes_R.thrust;
                    cntT_R[iR] += 1.0;
                }
            }
            
            // Feature 2: 2D Njets vs Thrust
            h2_Njets_vs_Thrust->Fill(shapes.thrust, nJets);
            
            // IMPROVED: Delta phi analysis with diagnostics
            if (jets.size() >= 2) {
                double phi1 = jets[0].phi();
                double phi2 = jets[1].phi();
                
                h_jet_leadingPt->Fill(jets[0].pt());
                h_jet_subleadingPt->Fill(jets[1].pt());
                h2_jet_phi1_vs_phi2->Fill(phi1, phi2);
                
                double dPhi = phi1 - phi2;
                if (dPhi > M_PI)  dPhi -= 2.0*M_PI;
                if (dPhi < -M_PI) dPhi += 2.0*M_PI;
                
                h_jetDeltaPhi->Fill(dPhi);
                
                if (nJets == 2) {
                    h_jet_deltaPhi_2jetOnly->Fill(dPhi);
                } else if (nJets >= 3) {
                    h_jet_deltaPhi_multijet->Fill(dPhi);
                }
                
                // DEBUG: Print first few events
                if (ievt < 5 && !INTERACTIVE_MODE) {
                    std::cout << "Event " << ievt << ": Njets=" << nJets 
                              << " | jet0: pT=" << jets[0].pt() << " phi=" << phi1
                              << " | jet1: pT=" << jets[1].pt() << " phi=" << phi2
                              << " | dPhi=" << dPhi << "\n";
                }
            }

            // LEADING HADRON - JET CORRELATION
            if (pions.size() >= 2 && jets.size() >= 2) {
                // Find leading pion in jet 0 and jet 1 separately (opposite jets!)
                int pion_in_jet0 = -1;
                double pion_in_jet0_pT = -1;
                int pion_in_jet1 = -1;
                double pion_in_jet1_pT = -1;
                
                // Check leading jet (jet 0)
                std::vector<PseudoJet> consts0 = jets[0].constituents();
                for (const PseudoJet &c : consts0) {
                    int idx = c.user_index();
                    if (idx >= 0 && idx < pythia.event.size()) {
                        int pdg = std::abs(pythia.event[idx].id());
                        if (pdg == 211) {
                            double pT = pythia.event[idx].pT();
                            if (pT > pion_in_jet0_pT) {
                                pion_in_jet0_pT = pT;
                                pion_in_jet0 = idx;
                            }
                        }
                    }
                }
                
                // Check subleading jet (jet 1)
                std::vector<PseudoJet> consts1 = jets[1].constituents();
                for (const PseudoJet &c : consts1) {
                    int idx = c.user_index();
                    if (idx >= 0 && idx < pythia.event.size()) {
                        int pdg = std::abs(pythia.event[idx].id());
                        if (pdg == 211) {
                            double pT = pythia.event[idx].pT();
                            if (pT > pion_in_jet1_pT) {
                                pion_in_jet1_pT = pT;
                                pion_in_jet1 = idx;
                            }
                        }
                    }
                }
                
                // Now fill histograms for leading pions in OPPOSITE jets
                if (pion_in_jet0 >= 0 && pion_in_jet1 >= 0) {
                    double phi0 = pythia.event[pion_in_jet0].phi();
                    double phi1 = pythia.event[pion_in_jet1].phi();
                    double eta0 = pythia.event[pion_in_jet0].eta();
                    double eta1 = pythia.event[pion_in_jet1].eta();
                    
                    h_leadPi_pT_inJet->Fill(pion_in_jet0_pT);
                    h_leadPi_pT_inJet->Fill(pion_in_jet1_pT);
                    h_leadPi_eta_inJet->Fill(eta0);
                    h_leadPi_eta_inJet->Fill(eta1);
                    
                    double dPhi_opposite = phi0 - phi1;
                    if (dPhi_opposite > M_PI) dPhi_opposite -= 2.0*M_PI;
                    if (dPhi_opposite < -M_PI) dPhi_opposite += 2.0*M_PI;
                    h_leadPi_deltaPhi_inJet->Fill(dPhi_opposite);
                }
                
                // Also analyze each pion separately for z and jT
                for (size_t ipi = 0; ipi < std::min(size_t(2), pions.size()); ++ipi) {
                    int pi_idx = pions[ipi].second;
                    
                    double pi_px = pythia.event[pi_idx].px();
                    double pi_py = pythia.event[pi_idx].py();
                    double pi_pz = pythia.event[pi_idx].pz();
                    
                    for (const PseudoJet &jet : jets) {
                        std::vector<PseudoJet> consts = jet.constituents();
                        bool found = false;
                        for (const PseudoJet &c : consts) {
                            if (c.user_index() == pi_idx) {
                                found = true;
                                break;
                            }
                        }
                        
                        if (found) {
                            double jpx = jet.px();
                            double jpy = jet.py();
                            double jpz = jet.pz();
                            double jnorm2 = jpx*jpx + jpy*jpy + jpz*jpz;
                            
                            if (jnorm2 > 0) {
                                double pdotj = pi_px*jpx + pi_py*jpy + pi_pz*jpz;
                                double z = pdotj / jnorm2;
                                
                                double px_par = (pdotj / jnorm2) * jpx;
                                double py_par = (pdotj / jnorm2) * jpy;
                                double pz_par = (pdotj / jnorm2) * jpz;
                                double perp_x = pi_px - px_par;
                                double perp_y = pi_py - py_par;
                                double perp_z = pi_pz - pz_par;
                                double jT = std::sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z);
                                
                                TH1D *hz = getHist1D("h_z_leadPi", "z (leading pions);z;Entries", 100, 0.0, 1.0);
                                TH1D *hjT = getHist1D("h_jT_leadPi", "jT (leading pions);jT [GeV];Entries", 100, 0.0, 5.0);
                                hz->Fill(z);
                                hjT->Fill(jT);
                                
                                // Feature 3: Delta pT (jet - pion)
                                double pion_pT = pythia.event[pi_idx].pT();
                                double jet_pT = jet.pt();
                                double dpt_jet_pion = jet_pT - pion_pT;
                                h_dpt_jet_minus_pion->Fill(dpt_jet_pion);
                                h2_dptJetPi_vs_pionPt->Fill(pion_pT, dpt_jet_pion);
                                
                                // Feature 4: Delta pT (pion - parent)
                                int parent_idx = -1;
                                int current_idx = pi_idx;
                                std::set<int> visited_parent;
                                
                                while (current_idx > 0 && visited_parent.find(current_idx) == visited_parent.end()) {
                                    visited_parent.insert(current_idx);
                                    int mother1 = pythia.event[current_idx].mother1();
                                    if (mother1 > 0 && mother1 < pythia.event.size()) {
                                        int pdg_mother = std::abs(pythia.event[mother1].id());
                                        if ((pdg_mother >= 100 && pdg_mother < 1000) || 
                                            (pdg_mother >= 1000 && pdg_mother < 10000)) {
                                            if (pdg_mother != 211) {
                                                parent_idx = mother1;
                                                break;
                                            }
                                        }
                                        current_idx = mother1;
                                    } else {
                                        break;
                                    }
                                }
                                
                                if (parent_idx > 0) {
                                    double parent_pT = pythia.event[parent_idx].pT();
                                    double dpt_pion_parent = pion_pT - parent_pT;
                                    h_dpt_pion_minus_parent->Fill(dpt_pion_parent);
                                    
                                    if (parent_pT > 0) {
                                        double pt_ratio = pion_pT / parent_pT;
                                        h2_ratioPiOverParent_vs_pionPt->Fill(pion_pT, pt_ratio);
                                    }
                                    
                                    double pion_eta = pythia.event[pi_idx].eta();
                                    double pion_phi = pythia.event[pi_idx].phi();
                                    double parent_eta = pythia.event[parent_idx].eta();
                                    double parent_phi = pythia.event[parent_idx].phi();
                                    double deta = pion_eta - parent_eta;
                                    double dphi = pion_phi - parent_phi;
                                    if (dphi > M_PI) dphi -= 2.0*M_PI;
                                    if (dphi < -M_PI) dphi += 2.0*M_PI;
                                    double dR = std::sqrt(deta*deta + dphi*dphi);
                                    h2_dR_PiParent_vs_pionPt->Fill(pion_pT, dR);
                                    
                                    int parent_pdg = std::abs(pythia.event[parent_idx].id());
                                    bool is_rho_omega_phi_Kst = (parent_pdg == 113 || parent_pdg == 213 || 
                                                                  parent_pdg == 223 || 
                                                                  parent_pdg == 333 || 
                                                                  parent_pdg == 313 || parent_pdg == 323);
                                    
                                    if (is_rho_omega_phi_Kst) {
                                        h_dpt_pion_minus_parent_rhoOmegaPhiKst->Fill(dpt_pion_parent);
                                    } else {
                                        h_dpt_pion_minus_parent_other->Fill(dpt_pion_parent);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
            
            // Same for leading kaons in OPPOSITE jets
            if (kaons.size() >= 2 && jets.size() >= 2) {
                // Find leading kaon in jet 0 and jet 1 separately
                int kaon_in_jet0 = -1;
                double kaon_in_jet0_pT = -1;
                int kaon_in_jet1 = -1;
                double kaon_in_jet1_pT = -1;
                
                // Check leading jet (jet 0)
                std::vector<PseudoJet> consts0 = jets[0].constituents();
                for (const PseudoJet &c : consts0) {
                    int idx = c.user_index();
                    if (idx >= 0 && idx < pythia.event.size()) {
                        int pdg = std::abs(pythia.event[idx].id());
                        if (pdg == 321) {
                            double pT = pythia.event[idx].pT();
                            if (pT > kaon_in_jet0_pT) {
                                kaon_in_jet0_pT = pT;
                                kaon_in_jet0 = idx;
                            }
                        }
                    }
                }
                
                // Check subleading jet (jet 1)
                std::vector<PseudoJet> consts1 = jets[1].constituents();
                for (const PseudoJet &c : consts1) {
                    int idx = c.user_index();
                    if (idx >= 0 && idx < pythia.event.size()) {
                        int pdg = std::abs(pythia.event[idx].id());
                        if (pdg == 321) {
                            double pT = pythia.event[idx].pT();
                            if (pT > kaon_in_jet1_pT) {
                                kaon_in_jet1_pT = pT;
                                kaon_in_jet1 = idx;
                            }
                        }
                    }
                }
                
                // Fill histograms for leading kaons in OPPOSITE jets
                if (kaon_in_jet0 >= 0 && kaon_in_jet1 >= 0) {
                    double phi0 = pythia.event[kaon_in_jet0].phi();
                    double phi1 = pythia.event[kaon_in_jet1].phi();
                    
                    h_leadK_pT_inJet->Fill(kaon_in_jet0_pT);
                    h_leadK_pT_inJet->Fill(kaon_in_jet1_pT);
                    
                    double dPhi_opposite = phi0 - phi1;
                    if (dPhi_opposite > M_PI) dPhi_opposite -= 2.0*M_PI;
                    if (dPhi_opposite < -M_PI) dPhi_opposite += 2.0*M_PI;
                    h_leadK_deltaPhi_inJet->Fill(dPhi_opposite);
                }
                
                // Analyze each kaon separately for z and jT
                for (size_t ik = 0; ik < std::min(size_t(2), kaons.size()); ++ik) {
                    int k_idx = kaons[ik].second;
                    
                    double k_px = pythia.event[k_idx].px();
                    double k_py = pythia.event[k_idx].py();
                    double k_pz = pythia.event[k_idx].pz();
                    
                    for (const PseudoJet &jet : jets) {
                        std::vector<PseudoJet> consts = jet.constituents();
                        bool found = false;
                        for (const PseudoJet &c : consts) {
                            if (c.user_index() == k_idx) {
                                found = true;
                                break;
                            }
                        }
                        
                        if (found) {
                            double jpx = jet.px();
                            double jpy = jet.py();
                            double jpz = jet.pz();
                            double jnorm2 = jpx*jpx + jpy*jpy + jpz*jpz;
                            
                            if (jnorm2 > 0) {
                                double pdotj = k_px*jpx + k_py*jpy + k_pz*jpz;
                                double z = pdotj / jnorm2;
                                
                                double px_par = (pdotj / jnorm2) * jpx;
                                double py_par = (pdotj / jnorm2) * jpy;
                                double pz_par = (pdotj / jnorm2) * jpz;
                                double perp_x = k_px - px_par;
                                double perp_y = k_py - py_par;
                                double perp_z = k_pz - pz_par;
                                double jT = std::sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z);
                                
                                TH1D *hz = getHist1D("h_z_leadK", "z (leading kaons);z;Entries", 100, 0.0, 1.0);
                                TH1D *hjT = getHist1D("h_jT_leadK", "jT (leading kaons);jT [GeV];Entries", 100, 0.0, 5.0);
                                hz->Fill(z);
                                hjT->Fill(jT);
                            }
                            break;
                        }
                    }
                }
            }

            // Per-jet observables
            for (const PseudoJet &jet : jets) {
                double eta = jet.eta();
                double rapidity = jet.rap();
                double jetPt = jet.pt();
                
                h_jetEta->Fill(eta);
                h_jetRapidity->Fill(rapidity);
                h_jetPt->Fill(jetPt);

                std::vector<PseudoJet> consts = jet.constituents();
                int nConst = consts.size();
                h_jetConstMult->Fill(nConst);

                double jpx = jet.px();
                double jpy = jet.py();
                double jpz = jet.pz();
                double jnorm2 = jpx*jpx + jpy*jpy + jpz*jpz;
                if (jnorm2 <= 0.0) continue;

                for (const PseudoJet &c : consts) {
                    double cpx = c.px();
                    double cpy = c.py();
                    double cpz = c.pz();

                    double pdotj = cpx*jpx + cpy*jpy + cpz*jpz;
                    double zfrag = pdotj / jnorm2;

                    double px_par = (pdotj / jnorm2) * jpx;
                    double py_par = (pdotj / jnorm2) * jpy;
                    double pz_par = (pdotj / jnorm2) * jpz;
                    double perp_x = cpx - px_par;
                    double perp_y = cpy - py_par;
                    double perp_z = cpz - pz_par;
                    double jT = std::sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z);

                    TH1D *hz = getHist1D("h_z_all", "z (hadron/jet);z;Entries", 100, 0.0, 1.0);
                    TH1D *hjT = getHist1D("h_jT_all", "jT (hadron rel. to jet) [GeV];jT [GeV];Entries", 100, 0.0, 5.0);
                    hz->Fill(zfrag);
                    hjT->Fill(jT);

                    int pythia_idx = c.user_index();
                    if (pythia_idx >= 0 && pythia_idx < pythia.event.size()) {
                        int pdgid = std::abs(pythia.event[pythia_idx].id());
                        if (pdgid == 211) {
                            TH1D *hz_pi = getHist1D("h_z_pi", "z (pions);z;Entries", 100, 0.0, 1.0);
                            hz_pi->Fill(zfrag);
                        }
                    }
                }
            }
        }

    } // end event loop

    std::cout << "\n========== Event Summary ==========\n";
    std::cout << "Total events processed: " << nEvents << "\n";
    std::cout << "Events with >=2 leading pions: " << eventsWithLeadingPions << "\n";
    std::cout << "Events with >=2 leading kaons: " << eventsWithLeadingKaons << "\n";
    std::cout << "===================================\n\n";

    std::cout << "\n========== Jet Analysis Summary ==========\n";
    std::cout << "Jet parameters: R=" << R << ", pT_min=" << jetPtMin << " GeV\n";
    std::cout << "Average jets per event: " << h_jetMult->GetMean() << "\n";
    if (h_jetMult->Integral() > 0) {
        std::cout << "Events with >=2 jets: " << h_jetMult->Integral(3, 21) << " out of " << h_jetMult->Integral() << "\n";
        std::cout << "Events with exactly 2 jets: " << h_jet_nJets_dist->GetBinContent(3) << "\n";
        std::cout << "Events with 3+ jets: " << h_jet_nJets_dist->Integral(4, 21) << "\n";
        std::cout << "Events with 0 jets: " << h_jet_nJets_dist->GetBinContent(1) << "\n";
        std::cout << "Events with 1 jet: " << h_jet_nJets_dist->GetBinContent(2) << "\n";
    }
    std::cout << "==========================================\n\n";

    // Write histograms
    auto writeIfNonEmpty = [&](TH1D* h) {
        if (!h) return;
        if (h->GetEntries() > 0) {
            std::cout << "Writing " << h->GetName() << " entries=" << h->GetEntries() << "\n";
            h->Write();
        } else {
            std::cout << "Skipping empty hist " << h->GetName() << "\n";
            delete h;
        }
    };
    
    auto writeIfNonEmpty2D = [&](TH2D* h) {
        if (!h) return;
        if (h->GetEntries() > 0) {
            std::cout << "Writing " << h->GetName() << " entries=" << h->GetEntries() << "\n";
            h->Write();
        } else {
            std::cout << "Skipping empty hist " << h->GetName() << "\n";
            delete h;
        }
    };

    // Write global histograms
    writeIfNonEmpty(h_jetMult);
    writeIfNonEmpty(h_jetPt);
    writeIfNonEmpty(h_jetEta);
    writeIfNonEmpty(h_jetRapidity);
    writeIfNonEmpty(h_jetDeltaPhi);
    writeIfNonEmpty(h_jetConstMult);
    writeIfNonEmpty(h_thrust);
    writeIfNonEmpty(h_sphericity);
    writeIfNonEmpty(h_circularity);
    writeIfNonEmpty(h_leadPi_pT);
    writeIfNonEmpty(h_leadPi_eta);
    writeIfNonEmpty(h_leadPi_deltaPhi);
    writeIfNonEmpty(h_leadK_pT);
    writeIfNonEmpty(h_leadK_deltaPhi);
    
    // Write jet-only hadron histograms
    writeIfNonEmpty(h_leadPi_pT_inJet);
    writeIfNonEmpty(h_leadPi_eta_inJet);
    writeIfNonEmpty(h_leadPi_deltaPhi_inJet);
    writeIfNonEmpty(h_leadK_pT_inJet);
    writeIfNonEmpty(h_leadK_deltaPhi_inJet);

    // Write diagnostic histograms
    writeIfNonEmpty(h_jet_nJets_dist);
    writeIfNonEmpty(h_jet_leadingPt);
    writeIfNonEmpty(h_jet_subleadingPt);
    writeIfNonEmpty(h_jet_deltaPhi_2jetOnly);
    writeIfNonEmpty(h_jet_deltaPhi_multijet);
    writeIfNonEmpty2D(h2_jet_phi1_vs_phi2);

    // Write NEW FEATURE histograms
    for (int iR = 0; iR < NR; ++iR) {
        writeIfNonEmpty(h_thrust_R[iR]);
    }
    
    std::cout << "\n========== Mean Thrust vs R ==========\n";
    for (int iR = 0; iR < NR; ++iR) {
        double meanT = (cntT_R[iR] > 0) ? (sumT_R[iR] / cntT_R[iR]) : 0.0;
        std::cout << "R = " << std::fixed << std::setprecision(1) << Rgrid[iR] 
                  << " : <T> = " << std::setprecision(4) << meanT << "\n";
    }
    std::cout << "======================================\n\n";
    
    writeIfNonEmpty2D(h2_Njets_vs_Thrust);
    writeIfNonEmpty(h_dpt_jet_minus_pion);
    writeIfNonEmpty2D(h2_dptJetPi_vs_pionPt);
    writeIfNonEmpty(h_dpt_pion_minus_parent);
    writeIfNonEmpty(h_dpt_pion_minus_parent_rhoOmegaPhiKst);
    writeIfNonEmpty(h_dpt_pion_minus_parent_other);
    writeIfNonEmpty2D(h2_ratioPiOverParent_vs_pionPt);
    writeIfNonEmpty2D(h2_dR_PiParent_vs_pionPt);

    // Write on-demand histograms
    for (auto &p : hmap) {
        TH1D *h = p.second;
        if (!h) continue;
        if (h->GetEntries() > 0) {
            std::cout << "Writing " << h->GetName() << " entries=" << h->GetEntries() << "\n";
            h->Write();
        } else {
            std::cout << "Skipping empty hist " << h->GetName() << "\n";
            delete h;
        }
    }

    fout->Close();
    delete fout;

    pythia.stat();

    return 0;
}
