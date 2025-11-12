// TMD.cc - Dual Method First Pion Analysis
// e+ e- -> Z -> qqbar, comparing two methods of identifying "first formed" pion:
// Method 1: Highest pT pion in jet
// Method 2: Fewest steps to quark in decay chain

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <iomanip>
#include <sstream>

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

// Pythia
#include "Pythia8/Pythia.h"

// FastJet
#include "fastjet/ClusterSequence.hh"

using namespace Pythia8;
using namespace fastjet;

// ---------- Ancestry Tracing ----------
struct AncestryResult {
    int steps;           // Steps to quark
    int quarkFlavor;     // PDG of quark (1-5)
    bool foundQuark;     // Successfully traced?
};

AncestryResult countStepsToQuark(const Event& event, int pion_idx) {
    AncestryResult result = {0, 0, false};
    int current = pion_idx;
    std::set<int> visited;
    
    while (current > 0 && visited.find(current) == visited.end()) {
        visited.insert(current);
        result.steps++;
        
        int pdg = std::abs(event[current].id());
        
        // Found a quark?
        if (pdg >= 1 && pdg <= 5) {
            result.quarkFlavor = pdg;
            result.foundQuark = true;
            break;
        }
        
        // Trace to mother
        int mother = event[current].mother1();
        if (mother > 0 && mother < event.size()) {
            current = mother;
        } else {
            break;  // No more mothers
        }
        
        // Safety: max 100 steps
        if (result.steps > 100) break;
    }
    
    return result;
}

// ---------- Event Shape: Thrust ----------
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

// ---------- Interactive Event Table ----------
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
            case 21: pname = "g"; break;
            case 1: pname = "d"; break;
            case -1: pname = "dbar"; break;
            case 2: pname = "u"; break;
            case -2: pname = "ubar"; break;
            case 3: pname = "s"; break;
            case -3: pname = "sbar"; break;
            case 4: pname = "c"; break;
            case -4: pname = "cbar"; break;
            case 5: pname = "b"; break;
            case -5: pname = "bbar"; break;
            case 111: pname = "pi0"; break;
            case 211: pname = "pi+"; break;
            case -211: pname = "pi-"; break;
            case 321: pname = "K+"; break;
            case -321: pname = "K-"; break;
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

char promptPager(const char* msg = "More") {
    std::cout << msg << " [Enter=+10, n=next event, q=quit]: " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) return 'q';
    if (line == "q" || line == "Q") return 'q';
    if (line == "n" || line == "N") return 'n';
    return 'c';
}

// ---------- Pion Info Struct ----------
struct PionInfo {
    int idx;
    double pT;
    int steps;
    int charge;
};

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
    
    TFile *fout = new TFile("dual_pion_analysis.root", "RECREATE");

    // ========== HISTOGRAMS ==========
    
    // Basic jet observables
    TH1D *h_jetPt = new TH1D("h_jetPt", "Jet pT;p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_jetEta = new TH1D("h_jetEta", "Jet #eta;#eta;Entries", 100, -5, 5);
    TH1D *h_jetDeltaPhi = new TH1D("h_jetDeltaPhi", "#Delta#phi between 2 jets;#Delta#phi [rad];Entries", 64, 0, 3.5);
    
    // Agreement histograms
    TH1D *h_method_agreement = new TH1D("h_method_agreement", 
        "Method Agreement;Agreement (0=different, 1=same);Events", 2, -0.5, 1.5);
    TH1D *h_delta_pT_between_methods = new TH1D("h_delta_pT_between_methods",
        "p_{T} difference between methods;p_{T}(method1) - p_{T}(method2) [GeV];Events", 100, -20, 20);
    TH1D *h_delta_steps_highestPt_pion = new TH1D("h_delta_steps_highestPt_pion",
        "Steps to quark for highest-p_{T} pion;Steps;Events", 50, 0, 50);
    TH2D *h2_agreement_vs_jetPt = new TH2D("h2_agreement_vs_jetPt",
        "Agreement vs jet p_{T};Jet p_{T} [GeV];Agreement", 50, 0, 50, 2, -0.5, 1.5);
    
    // Method 1: Highest pT
    TH1D *h_method1_firstPi_pT = new TH1D("h_method1_firstPi_pT",
        "Method 1: First pion p_{T};p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_method1_firstPi_pT_difference = new TH1D("h_method1_firstPi_pT_difference",
        "Method 1: |p_{T,1} - p_{T,2}|;|#Delta p_{T}| [GeV];Events", 100, 0, 30);
    TH1D *h_method1_firstPi_pT_ratio = new TH1D("h_method1_firstPi_pT_ratio",
        "Method 1: p_{T,1}/p_{T,2};p_{T} ratio;Events", 100, 0, 5);
    TH1D *h_method1_firstPi_deltaPhi = new TH1D("h_method1_firstPi_deltaPhi",
        "Method 1: #Delta#phi between first pions;#Delta#phi [rad];Events", 64, 0, 3.5);
    TH1D *h_method1_firstPi_z_fraction = new TH1D("h_method1_firstPi_z_fraction",
        "Method 1: z fraction;z;Entries", 100, 0, 1);
    TH1D *h_method1_charge_correlation = new TH1D("h_method1_charge_correlation",
        "Method 1: Charge correlation;Correlation (-1=same, +1=opposite);Events", 3, -1.5, 1.5);
    
    // Method 2: Fewest steps
    TH1D *h_method2_firstPi_pT = new TH1D("h_method2_firstPi_pT",
        "Method 2: First pion p_{T};p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_method2_firstPi_pT_difference = new TH1D("h_method2_firstPi_pT_difference",
        "Method 2: |p_{T,1} - p_{T,2}|;|#Delta p_{T}| [GeV];Events", 100, 0, 30);
    TH1D *h_method2_firstPi_pT_ratio = new TH1D("h_method2_firstPi_pT_ratio",
        "Method 2: p_{T,1}/p_{T,2};p_{T} ratio;Events", 100, 0, 5);
    TH1D *h_method2_firstPi_deltaPhi = new TH1D("h_method2_firstPi_deltaPhi",
        "Method 2: #Delta#phi between first pions;#Delta#phi [rad];Events", 64, 0, 3.5);
    TH1D *h_method2_firstPi_z_fraction = new TH1D("h_method2_firstPi_z_fraction",
        "Method 2: z fraction;z;Entries", 100, 0, 1);
    TH1D *h_method2_charge_correlation = new TH1D("h_method2_charge_correlation",
        "Method 2: Charge correlation;Correlation (-1=same, +1=opposite);Events", 3, -1.5, 1.5);
    
    // Ancestry diagnostics
    TH1D *h_steps_to_quark_all_pions = new TH1D("h_steps_to_quark_all_pions",
        "Steps to quark (all pions);Steps;Entries", 50, 0, 50);
    TH2D *h2_pT_vs_steps = new TH2D("h2_pT_vs_steps",
        "Pion p_{T} vs steps to quark;Steps;p_{T} [GeV]", 50, 0, 50, 100, 0, 50);
    TH1D *h_quark_flavor = new TH1D("h_quark_flavor",
        "Quark flavor;Flavor (1=d,2=u,3=s,4=c,5=b);Entries", 6, 0, 6);

    // Set directory to nullptr for all histograms
    h_jetPt->SetDirectory(nullptr);
    h_jetEta->SetDirectory(nullptr);
    h_jetDeltaPhi->SetDirectory(nullptr);
    h_method_agreement->SetDirectory(nullptr);
    h_delta_pT_between_methods->SetDirectory(nullptr);
    h_delta_steps_highestPt_pion->SetDirectory(nullptr);
    h2_agreement_vs_jetPt->SetDirectory(nullptr);
    h_method1_firstPi_pT->SetDirectory(nullptr);
    h_method1_firstPi_pT_difference->SetDirectory(nullptr);
    h_method1_firstPi_pT_ratio->SetDirectory(nullptr);
    h_method1_firstPi_deltaPhi->SetDirectory(nullptr);
    h_method1_firstPi_z_fraction->SetDirectory(nullptr);
    h_method1_charge_correlation->SetDirectory(nullptr);
    h_method2_firstPi_pT->SetDirectory(nullptr);
    h_method2_firstPi_pT_difference->SetDirectory(nullptr);
    h_method2_firstPi_pT_ratio->SetDirectory(nullptr);
    h_method2_firstPi_deltaPhi->SetDirectory(nullptr);
    h_method2_firstPi_z_fraction->SetDirectory(nullptr);
    h_method2_charge_correlation->SetDirectory(nullptr);
    h_steps_to_quark_all_pions->SetDirectory(nullptr);
    h2_pT_vs_steps->SetDirectory(nullptr);
    h_quark_flavor->SetDirectory(nullptr);

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
    
    // FastJet parameters
    double R = 0.4;
    double jetPtMin = 5.0;
    
    // Counters
    int nEventsProcessed = 0;
    int nEvents2Jets = 0;
    int nEventsBackToBack = 0;
    int nEventsWithPions = 0;
    int nEventsMethodsAgree = 0;

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

        nEventsProcessed++;

        // Collect final-state charged hadrons
        std::vector<int> hadrons;
        hadrons.reserve(64);
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            if (!pythia.event[i].isCharged()) continue;
            hadrons.push_back(i);
        }

        if (hadrons.empty()) continue;

        // Create FastJet inputs
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

        // Calculate thrust for quality cut
        double thrust = calculateThrust(fjInputs);
        if (thrust < 0.8) continue;  // Remove 3-jet events

        // Cluster with KT algorithm
        JetDefinition jetDef(kt_algorithm, R);
        ClusterSequence cs(fjInputs, jetDef);
        std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));

        // Require exactly 2 jets
        if (jets.size() != 2) continue;
        nEvents2Jets++;

        PseudoJet jet0 = jets[0];
        PseudoJet jet1 = jets[1];

        // Check back-to-back topology
        double phi0 = jet0.phi();
        double phi1 = jet1.phi();
        double dphi = std::abs(phi0 - phi1);
        if (dphi > M_PI) dphi = 2.0*M_PI - dphi;
        
        if (dphi < 2.8) continue;  // ~160 degrees
        nEventsBackToBack++;

        // Fill basic jet histograms
        h_jetPt->Fill(jet0.pt());
        h_jetPt->Fill(jet1.pt());
        h_jetEta->Fill(jet0.eta());
        h_jetEta->Fill(jet1.eta());
        h_jetDeltaPhi->Fill(dphi);

        // === ANALYZE JET 0 ===
        std::vector<PionInfo> pions_jet0;
        std::vector<PseudoJet> consts0 = jet0.constituents();
        
        for (const PseudoJet &c : consts0) {
            int idx = c.user_index();
            if (idx >= 0 && idx < pythia.event.size()) {
                int pdg = pythia.event[idx].id();
                int abs_pdg = std::abs(pdg);
                if (abs_pdg == 211) {
                    PionInfo info;
                    info.idx = idx;
                    info.pT = pythia.event[idx].pT();
                    info.charge = (pdg > 0) ? 1 : -1;
                    
                    AncestryResult anc = countStepsToQuark(pythia.event, idx);
                    info.steps = anc.steps;
                    
                    pions_jet0.push_back(info);
                    
                    // Fill diagnostic histograms
                    h_steps_to_quark_all_pions->Fill(anc.steps);
                    h2_pT_vs_steps->Fill(anc.steps, info.pT);
                    if (anc.foundQuark) {
                        h_quark_flavor->Fill(anc.quarkFlavor);
                    }
                }
            }
        }

        // === ANALYZE JET 1 ===
        std::vector<PionInfo> pions_jet1;
        std::vector<PseudoJet> consts1 = jet1.constituents();
        
        for (const PseudoJet &c : consts1) {
            int idx = c.user_index();
            if (idx >= 0 && idx < pythia.event.size()) {
                int pdg = pythia.event[idx].id();
                int abs_pdg = std::abs(pdg);
                if (abs_pdg == 211) {
                    PionInfo info;
                    info.idx = idx;
                    info.pT = pythia.event[idx].pT();
                    info.charge = (pdg > 0) ? 1 : -1;
                    
                    AncestryResult anc = countStepsToQuark(pythia.event, idx);
                    info.steps = anc.steps;
                    
                    pions_jet1.push_back(info);
                    
                    // Fill diagnostic histograms
                    h_steps_to_quark_all_pions->Fill(anc.steps);
                    h2_pT_vs_steps->Fill(anc.steps, info.pT);
                    if (anc.foundQuark) {
                        h_quark_flavor->Fill(anc.quarkFlavor);
                    }
                }
            }
        }

        // Need at least one pion in each jet
        if (pions_jet0.empty() || pions_jet1.empty()) continue;
        nEventsWithPions++;

        // === METHOD 1: HIGHEST pT ===
        auto pions_jet0_by_pT = pions_jet0;
        auto pions_jet1_by_pT = pions_jet1;
        std::sort(pions_jet0_by_pT.begin(), pions_jet0_by_pT.end(),
                  [](const PionInfo &a, const PionInfo &b) { return a.pT > b.pT; });
        std::sort(pions_jet1_by_pT.begin(), pions_jet1_by_pT.end(),
                  [](const PionInfo &a, const PionInfo &b) { return a.pT > b.pT; });
        
        int method1_idx0 = pions_jet0_by_pT[0].idx;
        int method1_idx1 = pions_jet1_by_pT[0].idx;
        
        // === METHOD 2: FEWEST STEPS ===
        auto pions_jet0_by_steps = pions_jet0;
        auto pions_jet1_by_steps = pions_jet1;
        std::sort(pions_jet0_by_steps.begin(), pions_jet0_by_steps.end(),
                  [](const PionInfo &a, const PionInfo &b) { return a.steps < b.steps; });
        std::sort(pions_jet1_by_steps.begin(), pions_jet1_by_steps.end(),
                  [](const PionInfo &a, const PionInfo &b) { return a.steps < b.steps; });
        
        int method2_idx0 = pions_jet0_by_steps[0].idx;
        int method2_idx1 = pions_jet1_by_steps[0].idx;

        // === COMPARISON ===
        bool agree_jet0 = (method1_idx0 == method2_idx0);
        bool agree_jet1 = (method1_idx1 == method2_idx1);
        bool both_agree = agree_jet0 && agree_jet1;
        
        h_method_agreement->Fill(both_agree ? 1 : 0);
        h2_agreement_vs_jetPt->Fill(jet0.pt(), agree_jet0 ? 1 : 0);
        h2_agreement_vs_jetPt->Fill(jet1.pt(), agree_jet1 ? 1 : 0);
        
        if (both_agree) nEventsMethodsAgree++;
        
        // Delta pT between methods
        if (!agree_jet0) {
            double dpT0 = pythia.event[method1_idx0].pT() - pythia.event[method2_idx0].pT();
            h_delta_pT_between_methods->Fill(dpT0);
        }
        if (!agree_jet1) {
            double dpT1 = pythia.event[method1_idx1].pT() - pythia.event[method2_idx1].pT();
            h_delta_pT_between_methods->Fill(dpT1);
        }
        
        // Steps for highest-pT pion
        h_delta_steps_highestPt_pion->Fill(pions_jet0_by_pT[0].steps);
        h_delta_steps_highestPt_pion->Fill(pions_jet1_by_pT[0].steps);

        // === METHOD 1 ANALYSIS ===
        double m1_pT0 = pythia.event[method1_idx0].pT();
        double m1_pT1 = pythia.event[method1_idx1].pT();
        double m1_phi0 = pythia.event[method1_idx0].phi();
        double m1_phi1 = pythia.event[method1_idx1].phi();
        int m1_charge0 = pions_jet0_by_pT[0].charge;
        int m1_charge1 = pions_jet1_by_pT[0].charge;
        
        h_method1_firstPi_pT->Fill(m1_pT0);
        h_method1_firstPi_pT->Fill(m1_pT1);
        
        double m1_pT_diff = std::abs(m1_pT0 - m1_pT1);
        h_method1_firstPi_pT_difference->Fill(m1_pT_diff);
        
        if (m1_pT1 > 0) {
            double m1_pT_ratio = m1_pT0 / m1_pT1;
            h_method1_firstPi_pT_ratio->Fill(m1_pT_ratio);
        }
        
        double m1_dphi = std::abs(m1_phi0 - m1_phi1);
        if (m1_dphi > M_PI) m1_dphi = 2.0*M_PI - m1_dphi;
        h_method1_firstPi_deltaPhi->Fill(m1_dphi);
        
        // Charge correlation
        int m1_charge_corr = (m1_charge0 == -m1_charge1) ? 1 : -1;
        h_method1_charge_correlation->Fill(m1_charge_corr);
        
        // Calculate z-fraction for method 1
        double m1_px0 = pythia.event[method1_idx0].px();
        double m1_py0 = pythia.event[method1_idx0].py();
        double m1_pz0 = pythia.event[method1_idx0].pz();
        double j0_px = jet0.px();
        double j0_py = jet0.py();
        double j0_pz = jet0.pz();
        double j0_norm2 = j0_px*j0_px + j0_py*j0_py + j0_pz*j0_pz;
        if (j0_norm2 > 0) {
            double pdotj = m1_px0*j0_px + m1_py0*j0_py + m1_pz0*j0_pz;
            double z0 = pdotj / j0_norm2;
            h_method1_firstPi_z_fraction->Fill(z0);
        }
        
        double m1_px1 = pythia.event[method1_idx1].px();
        double m1_py1 = pythia.event[method1_idx1].py();
        double m1_pz1 = pythia.event[method1_idx1].pz();
        double j1_px = jet1.px();
        double j1_py = jet1.py();
        double j1_pz = jet1.pz();
        double j1_norm2 = j1_px*j1_px + j1_py*j1_py + j1_pz*j1_pz;
        if (j1_norm2 > 0) {
            double pdotj = m1_px1*j1_px + m1_py1*j1_py + m1_pz1*j1_pz;
            double z1 = pdotj / j1_norm2;
            h_method1_firstPi_z_fraction->Fill(z1);
        }

        // === METHOD 2 ANALYSIS ===
        double m2_pT0 = pythia.event[method2_idx0].pT();
        double m2_pT1 = pythia.event[method2_idx1].pT();
        double m2_phi0 = pythia.event[method2_idx0].phi();
        double m2_phi1 = pythia.event[method2_idx1].phi();
        int m2_charge0 = pions_jet0_by_steps[0].charge;
        int m2_charge1 = pions_jet1_by_steps[0].charge;
        
        h_method2_firstPi_pT->Fill(m2_pT0);
        h_method2_firstPi_pT->Fill(m2_pT1);
        
        double m2_pT_diff = std::abs(m2_pT0 - m2_pT1);
        h_method2_firstPi_pT_difference->Fill(m2_pT_diff);
        
        if (m2_pT1 > 0) {
            double m2_pT_ratio = m2_pT0 / m2_pT1;
            h_method2_firstPi_pT_ratio->Fill(m2_pT_ratio);
        }
        
        double m2_dphi = std::abs(m2_phi0 - m2_phi1);
        if (m2_dphi > M_PI) m2_dphi = 2.0*M_PI - m2_dphi;
        h_method2_firstPi_deltaPhi->Fill(m2_dphi);
        
        // Charge correlation
        int m2_charge_corr = (m2_charge0 == -m2_charge1) ? 1 : -1;
        h_method2_charge_correlation->Fill(m2_charge_corr);
        
        // Calculate z-fraction for method 2
        double m2_px0 = pythia.event[method2_idx0].px();
        double m2_py0 = pythia.event[method2_idx0].py();
        double m2_pz0 = pythia.event[method2_idx0].pz();
        if (j0_norm2 > 0) {
            double pdotj = m2_px0*j0_px + m2_py0*j0_py + m2_pz0*j0_pz;
            double z0 = pdotj / j0_norm2;
            h_method2_firstPi_z_fraction->Fill(z0);
        }
        
        double m2_px1 = pythia.event[method2_idx1].px();
        double m2_py1 = pythia.event[method2_idx1].py();
        double m2_pz1 = pythia.event[method2_idx1].pz();
        if (j1_norm2 > 0) {
            double pdotj = m2_px1*j1_px + m2_py1*j1_py + m2_pz1*j1_pz;
            double z1 = pdotj / j1_norm2;
            h_method2_firstPi_z_fraction->Fill(z1);
        }

    } // end event loop

    // ========== SUMMARY ==========
    std::cout << "\n========================================\n";
    std::cout << "         EVENT SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Total events processed:        " << nEventsProcessed << "\n";
    std::cout << "Events with exactly 2 jets:    " << nEvents2Jets << "\n";
    std::cout << "Events back-to-back (>160°):   " << nEventsBackToBack << "\n";
    std::cout << "Events with pions in both jets:" << nEventsWithPions << "\n";
    std::cout << "Events where methods agree:    " << nEventsMethodsAgree << "\n";
    if (nEventsWithPions > 0) {
        double agreement_frac = 100.0 * nEventsMethodsAgree / nEventsWithPions;
        std::cout << "Agreement fraction:            " << std::fixed << std::setprecision(1) 
                  << agreement_frac << "%\n";
    }
    std::cout << "========================================\n\n";

    // ========== WRITE HISTOGRAMS ==========
    std::cout << "Writing histograms to file...\n";
    
    auto writeHist = [&](TH1* h) {
        if (h && h->GetEntries() > 0) {
            h->Write();
            std::cout << "  Wrote: " << h->GetName() << " (entries=" << h->GetEntries() << ")\n";
        }
    };

    // Basic jet observables
    writeHist(h_jetPt);
    writeHist(h_jetEta);
    writeHist(h_jetDeltaPhi);
    
    // Agreement histograms
    writeHist(h_method_agreement);
    writeHist(h_delta_pT_between_methods);
    writeHist(h_delta_steps_highestPt_pion);
    writeHist(h2_agreement_vs_jetPt);
    
    // Method 1 histograms
    writeHist(h_method1_firstPi_pT);
    writeHist(h_method1_firstPi_pT_difference);
    writeHist(h_method1_firstPi_pT_ratio);
    writeHist(h_method1_firstPi_deltaPhi);
    writeHist(h_method1_firstPi_z_fraction);
    writeHist(h_method1_charge_correlation);
    
    // Method 2 histograms
    writeHist(h_method2_firstPi_pT);
    writeHist(h_method2_firstPi_pT_difference);
    writeHist(h_method2_firstPi_pT_ratio);
    writeHist(h_method2_firstPi_deltaPhi);
    writeHist(h_method2_firstPi_z_fraction);
    writeHist(h_method2_charge_correlation);
    
    // Ancestry diagnostics
    writeHist(h_steps_to_quark_all_pions);
    writeHist(h2_pT_vs_steps);
    writeHist(h_quark_flavor);

    fout->Close();
    std::cout << "\nOutput written to: dual_pion_analysis.root\n";
    delete fout;

    pythia.stat();

    return 0;
}