// TMD.cc - Dual Method First Pion Analysis (v4c, fixed)
// e+ e- -> Z -> qqbar
// Compare two "first pion" definitions inside each jet:
//   A) Fewest steps to a quark in ancestry ("Fewest steps")
//   B) Highest transverse momentum pT ("Highest momentum")
// This version:
//  - uses visible (charged+neutral) inputs and anti-kt for jets
//  - KEEPS ONLY FOUR HISTOGRAMS:
//
//    1) First charged pion pT vs highest pT charged pion in jet 0 (0/1 agreement)
//    2) First charged pion pT vs highest pT charged pion in jet 1 (0/1 agreement)
//    3) ΔpT (first charged pion): pT(jet0) - pT(jet1)
//    4) Δφ between first charged pion in each jet
//
//  - interactive event browser is kept

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <iomanip>
#include <sstream>

#include "TFile.h"
#include "TH1D.h"
#include "TH1I.h"

// Pythia
#include "Pythia8/Pythia.h"

// FastJet
#include "fastjet/ClusterSequence.hh"

using namespace Pythia8;
using namespace fastjet;

// ---------- Utilities ----------
static inline double wrapToPi(double x) {
    while (x <= -M_PI) x += 2.0*M_PI;
    while (x >   M_PI) x -= 2.0*M_PI;
    return x;
}

// ---------- Ancestry Tracing ----------
struct AncestryResult {
    int steps;           // pion->mother is 1, ..., mother->quark = N
    int quarkFlavor;     // PDG of quark (1-5)
    bool foundQuark;
};

AncestryResult countStepsToQuark(const Event& event, int pion_idx) {
    AncestryResult result = {0, 0, false};
    int current = pion_idx;
    std::set<int> visited;

    while (current > 0 && visited.find(current) == visited.end()) {
        visited.insert(current);

        int mother = event[current].mother1();
        if (!(mother > 0 && mother < event.size())) break;

        result.steps++;
        current = mother;

        int pdg = std::abs(event[current].id());
        if (pdg >= 1 && pdg <= 5) {
            result.quarkFlavor = pdg;
            result.foundQuark = true;
            break;
        }

        if (result.steps > 200) break; // safety
    }
    return result;
}

// ---------- Event Shape: Thrust (coarse scan; quality cut only) ----------
double calculateThrust(const std::vector<fastjet::PseudoJet> &particles) {
    if (particles.empty()) return 0.0;
    double totalP = 0.0;
    for (const auto &p : particles)
        totalP += std::sqrt(p.px()*p.px() + p.py()*p.py() + p.pz()*p.pz());
    if (totalP <= 0.0) return 0.0;

    double maxThrust = 0.0;
    const int nSamples = 100;
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

// ---------- Interactive Event Table (optional) ----------
static void printEventTableWide(const Pythia8::Event& ev, int ievt,
                                int startRow, int rowsPerPage) {
    using std::cout; using std::left; using std::right; using std::setw; using std::setprecision; using std::fixed;

    const int N = ev.size();
    const int endRow = std::min(N - 1, startRow + rowsPerPage - 1);

    const int W_ROW=6, W_EVT=7, W_SIZE=6, W_NO=6, W_ID=8, W_NAME=14;
    const int W_ST=6,  W_M1=6, W_M2=6,  W_D1=6, W_D2=6,  W_P=13;

    auto sep = [&](){
        cout << std::string(W_ROW,'-') << "+"
             << std::string(W_EVT,'-') << "+"
             << std::string(W_SIZE,'-')<< "+"
             << std::string(W_NO,'-')  << "+"
             << std::string(W_ID,'-')  << "+"
             << std::string(W_NAME,'-')<< "+"
             << std::string(W_ST,'-')  << "+"
             << std::string(W_M1+1+W_M2,'-') << "+"
             << std::string(W_D1+1+W_D2,'-') << "+"
             << std::string(W_P,'-')   << "+"
             << std::string(W_P,'-')   << "+"
             << std::string(W_P,'-')   << "+"
             << std::string(W_P,'-')   << "\n";
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
            case 90: pname="system"; break;
            case 22: pname="gamma"; break;
            case 23: pname="Z0";    break;
            case 11: pname="e-";    break;
            case -11:pname="e+";    break;
            case 13: pname="mu-";   break;
            case -13:pname="mu+";   break;
            case 21: pname="g";     break;
            case 1:  pname="d";     break;
            case -1: pname="dbar";  break;
            case 2:  pname="u";     break;
            case -2: pname="ubar";  break;
            case 3:  pname="s";     break;
            case -3:pname="sbar";  break;
            case 4:  pname="c";     break;
            case -4:pname="cbar";  break;
            case 5:  pname="b";     break;
            case -5:pname="bbar";  break;
            case 111:pname="pi0";   break;
            case 211:pname="pi+";   break;
            case -211:pname="pi-";  break;
            case 2212:pname="p+";   break;
            case -2212:pname="pbar";break;
            default: { std::ostringstream os; os << "id" << pid; pname = os.str(); }
        }

        int d1 = p.daughter1();
        int d2 = p.daughter2();

        std::cout << left
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

    std::cout << "(showing rows " << startRow << " to " << endRow << ")\n";
}

char promptPager(const char* msg = "More") {
    std::cout << msg << " [Enter=+10, n=next event, q=quit]: " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) return 'q';
    if (line == "q" || line == "Q") return 'q';
    if (line == "n" || line == "N") return 'n';
    return 'c';
}

// ---------- Pion Info ----------
struct PionInfo {
    int idx;     // event row index (pythia.event index)
    double pT;
    int steps;
    int charge;
};

int main(int argc, char* argv[]) {
    bool INTERACTIVE_MODE = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--interactive" || arg == "-i") {
            INTERACTIVE_MODE = true;
            std::cout << "\n*** INTERACTIVE MODE ENABLED ***\n\n";
        }
    }

    TFile *fout = new TFile("week10.root", "RECREATE");

    // ========== HISTOGRAMS (ONLY 4 KEPT) ==========
    TH1I *h_samePion_jet0 = new TH1I(
        "h_samePion_jet0",
        "First charged pion p_{T} vs highest p_{T} charged pion in jet 0;class;Jets",
        2, -0.5, 1.5
    );

    TH1I *h_samePion_jet1 = new TH1I(
        "h_samePion_jet1",
        "First charged pion p_{T} vs highest p_{T} charged pion in jet 1;class;Jets",
        2, -0.5, 1.5
    );

    TH1D *h_fewestSteps_firstPi_pT_signed_difference =
        new TH1D("h_fewestSteps_firstPi_pT_signed_difference",
                 "#Delta p_{T} (first charged pion): p_{T,jet0} - p_{T,jet1};#Delta p_{T} [GeV];Events",
                 100, -30, 30);

    TH1D *h_fewestSteps_firstPi_deltaPhi =
        new TH1D("h_fewestSteps_firstPi_deltaPhi",
                 "#Delta#phi between first charged pion in each jet;#Delta#phi [rad];Entries",
                 128, -M_PI, M_PI);

    // Label x-axis bins for the 0/1 agreement histograms
    for (TH1* h : std::vector<TH1*>{h_samePion_jet0, h_samePion_jet1}) {
        h->GetXaxis()->SetBinLabel(1, "0 (different)");
        h->GetXaxis()->SetBinLabel(2, "1 (same)");
    }

    // Detach from current directory
    for (TH1* h : std::vector<TH1*>{
            h_samePion_jet0,
            h_samePion_jet1,
            h_fewestSteps_firstPi_pT_signed_difference,
            h_fewestSteps_firstPi_deltaPhi }) {
        if (h) h->SetDirectory(nullptr);
    }

    // ========== PYTHIA ==========
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
    const double R = 0.4;
    const double jetPtMin = 5.0;

    int nEventsProcessed = 0, nEvents2Jets = 0, nEventsBackToBack = 0;
    int nEventsWithPions = 0, nEventsMethodsAgree = 0;

    bool quitAll = false;

    // ========== EVENT LOOP ==========
    for (int ievt = 0; ievt < nEvents; ++ievt) {
        if (!pythia.next()) continue;
        nEventsProcessed++;

        if (INTERACTIVE_MODE) {
            const int PAGE = 10;
            int startRow = 0;
            printEventTableWide(pythia.event, ievt, startRow, PAGE);
            startRow += PAGE;
            while (true) {
                char cmd = promptPager();
                if (cmd == 'q') { quitAll = true; break; }
                else if (cmd == 'n') break;
                else {
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

        if (!INTERACTIVE_MODE && (ievt + 1) % 1000 == 0)
            std::cout << "Processed " << (ievt + 1) << " events...\n";

        // visible final-state
        std::vector<int> finals;
        finals.reserve(128);
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal())   continue;
            if (!pythia.event[i].isVisible()) continue;
            finals.push_back(i);
        }
        if (finals.empty()) continue;

        std::vector<PseudoJet> fjInputs;
        fjInputs.reserve(finals.size());
        for (int idx : finals) {
            PseudoJet pj(pythia.event[idx].px(),
                         pythia.event[idx].py(),
                         pythia.event[idx].pz(),
                         pythia.event[idx].e());
            pj.set_user_index(idx);
            fjInputs.push_back(pj);
        }

        double thrust = calculateThrust(fjInputs);
        if (thrust < 0.8) continue;

        JetDefinition jetDef(antikt_algorithm, R);
        ClusterSequence cs(fjInputs, jetDef);
        std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));

        if (jets.size() != 2) continue;
        nEvents2Jets++;

        PseudoJet jet0 = jets[0], jet1 = jets[1];

        double dphi_jets = wrapToPi(jet0.phi() - jet1.phi());
        if (std::fabs(dphi_jets) < 2.8) continue; // ~160 degrees
        nEventsBackToBack++;

        auto collectPions = [&](const PseudoJet& j)->std::vector<PionInfo> {
            std::vector<PionInfo> out;
            for (const PseudoJet &c : j.constituents()) {
                int idx = c.user_index();
                if (idx < 0 || idx >= pythia.event.size()) continue;
                int pdg = pythia.event[idx].id();
                if (std::abs(pdg) != 211) continue; // charged pion only
                PionInfo info;
                info.idx    = idx;
                info.pT     = pythia.event[idx].pT();
                info.charge = (pdg > 0) ? 1 : -1;
                AncestryResult anc = countStepsToQuark(pythia.event, idx);
                info.steps  = anc.steps;
                out.push_back(info);
            }
            return out;
        };

        std::vector<PionInfo> pions_jet0 = collectPions(jet0);
        std::vector<PionInfo> pions_jet1 = collectPions(jet1);
        if (pions_jet0.empty() || pions_jet1.empty()) continue;
        nEventsWithPions++;

        auto by_pT_desc   = [](const PionInfo& a, const PionInfo& b){ return a.pT > b.pT; };
        auto by_steps_asc = [](const PionInfo& a, const PionInfo& b){ return a.steps < b.steps; };

        auto pj0_pt = pions_jet0; std::sort(pj0_pt.begin(), pj0_pt.end(), by_pT_desc);
        auto pj1_pt = pions_jet1; std::sort(pj1_pt.begin(), pj1_pt.end(), by_pT_desc);
        int idx_high0 = pj0_pt[0].idx;
        int idx_high1 = pj1_pt[0].idx;

        auto pj0_st = pions_jet0; std::sort(pj0_st.begin(), pj0_st.end(), by_steps_asc);
        auto pj1_st = pions_jet1; std::sort(pj1_st.begin(), pj1_st.end(), by_steps_asc);
        int idx_step0 = pj0_st[0].idx;
        int idx_step1 = pj1_st[0].idx;

        bool same0 = (idx_high0 == idx_step0);
        bool same1 = (idx_high1 == idx_step1);
        bool both  = (same0 && same1);

        h_samePion_jet0->Fill(same0 ? 1 : 0);
        h_samePion_jet1->Fill(same1 ? 1 : 0);

        if (both) nEventsMethodsAgree++;

        auto dphiSigned = [&](int idxA, int idxB)->double{
            return wrapToPi(pythia.event[idxA].phi() - pythia.event[idxB].phi());
        };

        {
            double pT0 = pythia.event[idx_step0].pT();
            double pT1 = pythia.event[idx_step1].pT();

            h_fewestSteps_firstPi_pT_signed_difference->Fill(pT0 - pT1);

            double dphi_pi = dphiSigned(idx_step0, idx_step1);
            h_fewestSteps_firstPi_deltaPhi->Fill(dphi_pi);
        }
    } // end event loop

    std::cout << "\n========================================\n";
    std::cout << "         EVENT SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Total events processed:        " << nEventsProcessed << "\n";
    std::cout << "Events with exactly 2 jets:    " << nEvents2Jets << "\n";
    std::cout << "Events back-to-back (>160°):   " << nEventsBackToBack << "\n";
    std::cout << "Events with pions in both jets:" << nEventsWithPions << "\n";
    std::cout << "Events where both jets agree:  " << nEventsMethodsAgree << "\n";
    if (nEventsWithPions > 0) {
        double frac = 100.0 * nEventsMethodsAgree / nEventsWithPions;
        std::cout << "Agreement fraction (both jets): "
                  << std::fixed << std::setprecision(1) << frac << "%\n";
    }
    std::cout << "========================================\n\n";

    std::cout << "Writing histograms to file...\n";
    auto writeH = [&](TH1* h){
        if (h && h->GetEntries()>0) {
            h->Write();
            std::cout << "  wrote: " << h->GetName() << "\n";
        }
    };

    writeH(h_samePion_jet0);
    writeH(h_samePion_jet1);
    writeH(h_fewestSteps_firstPi_pT_signed_difference);
    writeH(h_fewestSteps_firstPi_deltaPhi);

    fout->Close();
    std::cout << "\nOutput written to: week10.root\n";
    std::cout << "To view: root -l week10.root\n";
    delete fout;

    pythia.stat();
    return 0;
}
