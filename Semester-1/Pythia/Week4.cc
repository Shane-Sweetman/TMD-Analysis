// Produce e+ e- events, study final-state hadrons, and calculate delta phi, pT imbalance,

#include <iostream> 
#include <vector>
#include <cmath>

#include "TFile.h" // Root
#include "TH1D.h"
#include "Pythia8/Pythia.h"

using namespace Pythia8;

int main() {
    TFile *fout = new TFile("ee_hadron_corr.root", "RECREATE"); // creates root file

    // Histograms
    TH1D *h_dPhi        = new TH1D("h_dPhi", "Delta phi between hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_pTimbalance = new TH1D("h_pTimbalance", "pT imbalance;|Delta pT| [GeV];Entries", 100, 0, 10);
    TH1D *h_OS          = new TH1D("h_OS", "Opposite-sign hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2); // Opposite sign
    TH1D *h_SS          = new TH1D("h_SS", "Same-sign hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2); // Same sign

    // Pythia setup
    Pythia pythia;
    pythia.readString("Beams:idA = -11");      // electron beam
    pythia.readString("Beams:idB = 11");       // positron beam
    pythia.readString("Beams:eCM = 91.2");     // center of mass energy 
    pythia.readString("PDF:lepton = off"); // no lepton PDFs
    pythia.readString("HadronLevel:all = on"); // hadronization 
    pythia.readString("WeakSingleBoson:ffbar2gmZ = on"); // enable Z/gamma* → q q̄ production

    // Set fixed random seed for reproducibility
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = 123456788");

    // Initialize Pythia
    if (!pythia.init()) {
        std::cerr << "Pythia initialization failed\n"; // Check for errors
        return 1;
    }

    const int nEvents = 20000; // make 20,000 collisions
    int totalPairs = 0; // counts total hadron pairs

    for (int ievt = 0; ievt < nEvents; ++ievt) { // event loop
        if (!pythia.next()) continue;

        // Collect all final-state charged hadrons
        std::vector<int> hadrons;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            if (!pythia.event[i].isCharged()) continue;
            hadrons.push_back(i);
        }

        // Loop over all unique hadron pairs
        for (size_t a = 0; a < hadrons.size(); ++a) {
            for (size_t b = a + 1; b < hadrons.size(); ++b) {
                int ia = hadrons[a];
                int ib = hadrons[b];

                // x & y components of momentum for each particle in pair
                double px1 = pythia.event[ia].px();
                double py1 = pythia.event[ia].py();
                double px2 = pythia.event[ib].px();
                double py2 = pythia.event[ib].py();

                // Delta phi
                double phi1 = std::atan2(py1, px1);
                double phi2 = std::atan2(py2, px2);
                double dPhi = phi1 - phi2; 

                // Wrap to [-pi, pi] ensures correct periodicity
                if (dPhi > M_PI)  dPhi -= 2*M_PI;
                if (dPhi < -M_PI) dPhi += 2*M_PI;

                // pT imbalance
                double pT1 = std::sqrt(px1*px1 + py1*py1);
                double pT2 = std::sqrt(px2*px2 + py2*py2);
                double pTimb = std::fabs(pT1 - pT2);

                // Fill histograms
                h_dPhi->Fill(dPhi);
                h_pTimbalance->Fill(pTimb);

                // Fill OS/SS histograms
                int qprod = pythia.event[ia].charge() * pythia.event[ib].charge(); // charge product
                if (qprod < 0) h_OS->Fill(dPhi);
                else           h_SS->Fill(dPhi);

                ++totalPairs;
            }
        }
    }

    std::cout << "Done. Processed " << nEvents << " events and " << totalPairs << " hadron pairs." << std::endl;

    // Write histograms
    h_dPhi->Write();
    h_pTimbalance->Write();
    h_OS->Write();
    h_SS->Write();
    fout->Close();

    // Pythia summary
    pythia.stat();

    return 0;
}
