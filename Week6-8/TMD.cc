// TMD.cc
// e+ e- -> qqbar, hadron-pair & jet-based TMD-like observables with FastJet (anti-kt).
// Extended with: jet observables + leading hadron ancestry tracing

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>

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

// ---------- get primary hard-process quark flavor ----------
int getPrimaryQuarkFlavor(const Event &event) {
    // Return codes: 1=down,2=up,3=strange,4=heavy (c/b),0=unknown
    int n = event.size();
    // First pass: status 23 (pythia status code for outgoing hard partons)
    for (int i = 0; i < n; ++i) { // Loop over all particles in the event.
        int st = event[i].status(); // Get the status code of particle i
        int id = std::abs(event[i].id()); // Get the PDG ID (Particle Data Group identifier) of the particle and take absolute value
        if (st == 23 && id >= 1 && id <= 5) { // Check if the particle is:  A hard-process quark (status 23) or One of the light quarks (u, d, s, c, b)
            if (id == 1) return 1; // down quark
            if (id == 2) return 2; // up quark
            if (id == 3) return 3; // strange quark
            return 4; // charm or bottom quark
        }
    }
    // Fallback: any particle with PDG 1..5 when no status 23 found
    for (int i = 0; i < n; ++i) { // Loop over all particles in the event.
        int id = std::abs(event[i].id()); // Get the PDG ID (Particle Data Group identifier) of the particle and take absolute value
        if (id >= 1 && id <= 5) { // Check if the particle is one of the light quarks (u, d, s, c, b)
            if (id == 1) return 1; // down quark
            if (id == 2) return 2; // up quark
            if (id == 3) return 3; // strange quark
            return 4; // charm or bottom quark
        }
    }
    return 0; // Return 0 if nothing found
}


// ---------- Ancestry Tracing ----------
struct AncestryInfo {
    int motherQuarkFlavor;  // 1-5 for d,u,s,c,b; 0=unknown
    int resonanceID;         // PDG ID of intermediate resonance (0 if none)
    bool fromPrimaryQuark;   // true if traceable to hard process quark
    std::vector<int> chain;  // full decay chain indices
};



// Trace hadron back to find mother quark and resonances
AncestryInfo traceAncestry(const Event &event, int hadronIdx) { 
    AncestryInfo info; // Initializes fields to default values
    info.motherQuarkFlavor = 0; // unknown
    info.resonanceID = 0; // none found yet
    info.fromPrimaryQuark = false; // default false
    
    int current = hadronIdx; // stores the index of the particle currently being traced
    std::set<int> visited; // prevent infinite loops
    
    while (current > 0 && visited.find(current) == visited.end()) { //Continue only if current > 0 and not yet visited
        visited.insert(current); // mark current as visited
        info.chain.push_back(current); // add to decay chain
        
        int pdg = std::abs(event[current].id()); // absolute PDG ID
        int status = event[current].status(); // status code, whether it’s a hard-process particle, intermediate, or final
        
        // Check if we found a quark (PDG 1-5)
        if (pdg >= 1 && pdg <= 5) {
            info.motherQuarkFlavor = pdg;
            // Check if it's from hard process (status 23 or similar)
            if (status == 23 || status == 21 || status == 22) {
                info.fromPrimaryQuark = true; // mark as from primary quark
            }
            break; // stop tracing once a quark is found
        }
        
        // Check for resonances (mesons/baryons that decay)
        // D mesons (411-435), B mesons (511-545), K mesons (311-313, 321-323)
        // rho (113, 213), omega (223), phi (333), etc.
        if ((pdg >= 100 && pdg < 1000) || // mesons
            (pdg >= 1000 && pdg < 10000)) { // baryons
            if (info.resonanceID == 0) { // store first resonance found
                info.resonanceID = event[current].id(); // store actual PDG ID (with sign)
            }
        }
        
        // Move to mother
        int mother1 = event[current].mother1(); // get first mother index
        if (mother1 > 0 && mother1 < event.size()) { // checks for valid mother index 
            current = mother1; // continue tracing up the chain
        } else { // no valid mother found
            break; // exit loop
        }
    }
    
    return info; // return the collected ancestry information
}

// ---------- Event Shape Calculations ----------
struct EventShapes { // to hold event shape results
    double thrust; 
    double sphericity;
    double circularity;
};

double calculateThrust(const std::vector<PseudoJet> &particles) { // calculate thrust
    if (particles.empty()) return 0.0; // no particles, thrust = 0
    
    double totalP = 0.0; // total momentum magnitude
    for (const auto &p : particles) { // sum over all particles
        totalP += std::sqrt(p.px()*p.px() + p.py()*p.py() + p.pz()*p.pz()); // magnitude of momentum
    }
    if (totalP <= 0.0) return 0.0; // avoid division by zero
    
    double maxThrust = 0.0; // initialize max thrust
    int nSamples = 100; // number of direction samples (angle grid)
    for (int i = 0; i < nSamples; ++i) { // theta angle
        double theta = M_PI * i / nSamples; 
        for (int j = 0; j < nSamples; ++j) { // phi angle
            double phi = 2.0 * M_PI * j / nSamples; 
            double nx = std::sin(theta) * std::cos(phi); // x-component of the direction vector
            double ny = std::sin(theta) * std::sin(phi); // y-component of the direction vector
            double nz = std::cos(theta); // z-component of the direction vector
            
            double sum = 0.0; // sum of projections
            for (const auto &p : particles) { // loop over particles
                double dot = std::abs(p.px()*nx + p.py()*ny + p.pz()*nz); // magnitude of the particle’s momentum along the chosen direction
                sum += dot; // accumulate
            }
            double thrust = sum / totalP; // normalize by total momentum
            if (thrust > maxThrust) maxThrust = thrust; // update max thrust
        }
    }
    return maxThrust; // return the maximum thrust found
}

EventShapes calculateEventShapes(const std::vector<PseudoJet> &particles) { // calculate event shapes
    EventShapes shapes = {0.0, 0.0, 0.0}; // initialize to zero
    if (particles.empty()) return shapes; // no particles, return zeros
    
    double S[3][3] = {{0,0,0},{0,0,0},{0,0,0}}; // momentum tensor
    double totalP2 = 0.0; // total momentum squared
    
    for (const auto &p : particles) { // loop over particles
        double px = p.px(), py = p.py(), pz = p.pz(); // momentum components
        double p2 = px*px + py*py + pz*pz; // momentum squared
        totalP2 += p2; // accumulate total p^2
        
        S[0][0] += px*px; 
        S[0][1] += px*py; S[1][0] += px*py;
        S[0][2] += px*pz; S[2][0] += px*pz;
        S[1][1] += py*py;
        S[1][2] += py*pz; S[2][1] += py*pz;
        S[2][2] += pz*pz;
    }
    
    if (totalP2 <= 0.0) return shapes; // avoid division by zero
    
    for (int i = 0; i < 3; ++i) { // normalize tensor
        for (int j = 0; j < 3; ++j) { // loop over indices
            S[i][j] /= totalP2; // normalize by total p^2
        }
    }
    
    TMatrixDSym matrix(3); // create symmetric matrix
    for (int i = 0; i < 3; ++i) { // fill matrix
        for (int j = 0; j < 3; ++j) { // loop over indices
            matrix(i,j) = S[i][j]; // set matrix element
        }
    }
    
    TMatrixDSymEigen eigen(matrix); // eigen decomposition
    TVectorD eigenvalues = eigen.GetEigenValues(); // get eigenvalues
    
    std::vector<double> eigs = {eigenvalues[0], eigenvalues[1], eigenvalues[2]}; // copy to vector
    std::sort(eigs.begin(), eigs.end(), std::greater<double>()); // sort descending
    
    double lambda2 = eigs[1]; // second largest eigenvalue
    double lambda3 = eigs[2]; // smallest eigenvalue
    
    shapes.sphericity = 1.5 * (lambda2 + lambda3); // calculate sphericity
    shapes.thrust = calculateThrust(particles); // calculate thrust
    
    // Circularity (2D transverse)
    double S2D[2][2] = {{0,0},{0,0}}; // transverse momentum tensor
    double totalPT2 = 0.0; // total transverse momentum squared
    for (const auto &p : particles) { // loop over particles
        double px = p.px(), py = p.py(); // transverse momentum components
        double pt2 = px*px + py*py; // transverse momentum squared
        totalPT2 += pt2; // accumulate total pT^2
        S2D[0][0] += px*px; // fill tensor
        S2D[0][1] += px*py; S2D[1][0] += px*py; // symmetric
        S2D[1][1] += py*py; // fill tensor
    }
    
    if (totalPT2 > 0.0) { // avoid division by zero
        S2D[0][0] /= totalPT2; 
        S2D[0][1] /= totalPT2; 
        S2D[1][0] /= totalPT2;
        S2D[1][1] /= totalPT2;
        
        double trace = S2D[0][0] + S2D[1][1]; // trace of the 2D tensor
        double det = S2D[0][0]*S2D[1][1] - S2D[0][1]*S2D[1][0]; // determinant
        double discriminant = trace*trace - 4*det; // discriminant for eigenvalues
        if (discriminant >= 0) { // ensure real eigenvalues
            double sqrtDisc = std::sqrt(discriminant); // square root of discriminant
            double eig1 = (trace + sqrtDisc) / 2.0; // first eigenvalue
            double eig2 = (trace - sqrtDisc) / 2.0; // second eigenvalue
            double minEig = std::min(eig1, eig2); // minimum eigenvalue
            double sumEig = eig1 + eig2; // sum of eigenvalues
            if (sumEig > 0) { // avoid division by zero
                shapes.circularity = 2.0 * minEig / sumEig; // calculate circularity
            }
        }
    }
    
    return shapes; // return calculated shapes
}

// ---------- On-demand histogram map ----------
static std::map<std::string, TH1D*> hmap; // 1D histogram map
static std::map<std::string, TH2D*> hmap2D; // 2D histogram map

TH1D* getHist1D(const std::string &name, const std::string &title, // 1D histogram retrieval/creation
                int nbins = 64, double xmin = -3.2, double xmax = 3.2) { // default binning
    auto it = hmap.find(name); // check if histogram exists
    if (it != hmap.end()) return it->second; // return existing histogram
    TH1D *h = new TH1D(name.c_str(), title.c_str(), nbins, xmin, xmax); // create new histogram
    h->SetDirectory(nullptr); // prevent ROOT from managing memory
    hmap[name] = h; // store in map
    return h; // return new histogram
}

TH2D* getHist2D(const std::string &name, const std::string &title, // 2D histogram retrieval/creation
                int nbinsx, double xmin, double xmax, // x-axis binning
                int nbinsy, double ymin, double ymax) { // y-axis binning
    auto it = hmap2D.find(name); // check if histogram exists
    if (it != hmap2D.end()) return it->second; // return existing histogram
    TH2D *h = new TH2D(name.c_str(), title.c_str(), nbinsx, xmin, xmax, nbinsy, ymin, ymax); // create new histogram
    h->SetDirectory(nullptr); // prevent ROOT from managing memory
    hmap2D[name] = h; // store in map
    return h; // return new histogram
}

int main() {
    TFile *fout = new TFile("ee_hadron_corr.root", "RECREATE");

    // Global pair-based histograms
    TH1D *h_dPhi        = new TH1D("h_dPhi", "Delta phi between hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_pTimbalance = new TH1D("h_pTimbalance", "pT imbalance;|Delta pT| [GeV];Entries", 100, 0, 10);
    TH1D *h_OS          = new TH1D("h_OS", "Opposite-sign hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_SS          = new TH1D("h_SS", "Same-sign hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2);

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
    
    // 2D correlations
    TH2D *h_jetPt_vs_Eta = new TH2D("h_jetPt_vs_Eta", "Jet pT vs Eta;#eta;p_{T} [GeV]", 50, -5, 5, 50, 0, 50);
    TH2D *h_thrust_vs_sphericity = new TH2D("h_thrust_vs_sphericity", "Thrust vs Sphericity;S;T", 50, 0, 1, 50, 0, 1);

    // Leading hadron observables (NEW)
    TH1D *h_leadPi_pT = new TH1D("h_leadPi_pT", "Leading pion pT;p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_leadPi_eta = new TH1D("h_leadPi_eta", "Leading pion eta;#eta;Entries", 100, -5, 5);
    TH1D *h_leadPi_deltaPhi = new TH1D("h_leadPi_deltaPhi", "Delta phi between leading pions;#Delta#phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_leadK_pT = new TH1D("h_leadK_pT", "Leading kaon pT;p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_leadK_deltaPhi = new TH1D("h_leadK_deltaPhi", "Delta phi between leading kaons;#Delta#phi [rad];Entries", 64, -3.2, 3.2);

    h_dPhi->SetDirectory(nullptr);
    h_pTimbalance->SetDirectory(nullptr);
    h_OS->SetDirectory(nullptr);
    h_SS->SetDirectory(nullptr);
    h_jetMult->SetDirectory(nullptr);
    h_jetPt->SetDirectory(nullptr);
    h_jetEta->SetDirectory(nullptr);
    h_jetRapidity->SetDirectory(nullptr);
    h_jetDeltaPhi->SetDirectory(nullptr);
    h_jetConstMult->SetDirectory(nullptr);
    h_thrust->SetDirectory(nullptr);
    h_sphericity->SetDirectory(nullptr);
    h_circularity->SetDirectory(nullptr);
    h_jetPt_vs_Eta->SetDirectory(nullptr);
    h_thrust_vs_sphericity->SetDirectory(nullptr);
    h_leadPi_pT->SetDirectory(nullptr);
    h_leadPi_eta->SetDirectory(nullptr);
    h_leadPi_deltaPhi->SetDirectory(nullptr);
    h_leadK_pT->SetDirectory(nullptr);
    h_leadK_deltaPhi->SetDirectory(nullptr);

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
    int totalPairs = 0;
    int eventsWithTag = 0;
    int eventsWithLeadingPions = 0;
    int eventsWithLeadingKaons = 0;

    // FastJet parameters
    double R = 0.6;
    double jetPtMin = 1.0;

    // Event loop
    for (int ievt = 0; ievt < nEvents; ++ievt) {
        if (!pythia.next()) continue;

        int flavorID = getPrimaryQuarkFlavor(pythia.event);
        std::string flavorName;
        if (flavorID == 1) flavorName = "down";
        else if (flavorID == 2) flavorName = "up";
        else if (flavorID == 3) flavorName = "strange";
        else if (flavorID == 4) flavorName = "heavy";
        else flavorName = "unknown";

        if (flavorID != 0) ++eventsWithTag;

        // Collect final-state charged hadrons
        std::vector<int> hadrons;
        hadrons.reserve(64);
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            if (!pythia.event[i].isCharged()) continue;
            hadrons.push_back(i);
        }

        // ========== LEADING HADRON ANALYSIS ==========
        // Find leading pions (±211)
        std::vector<std::pair<double, int>> pions; // (momentum, index)
        for (int idx : hadrons) {
            int pdg = std::abs(pythia.event[idx].id());
            if (pdg == 211) {
                double p = pythia.event[idx].pAbs();
                pions.push_back({p, idx});
            }
        }
        std::sort(pions.begin(), pions.end(), std::greater<std::pair<double,int>>());
        
        // Find leading kaons (±321)
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
            
            // Trace ancestry
            AncestryInfo anc1 = traceAncestry(pythia.event, pi1_idx);
            AncestryInfo anc2 = traceAncestry(pythia.event, pi2_idx);
            
            // Kinematics
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
            
            // Flavor-tagged leading pion observables
            if (anc1.fromPrimaryQuark && anc1.motherQuarkFlavor > 0) {
                std::string qflavor = (anc1.motherQuarkFlavor == 1) ? "down" :
                                     (anc1.motherQuarkFlavor == 2) ? "up" :
                                     (anc1.motherQuarkFlavor == 3) ? "strange" : "heavy";
                TH1D *hpt = getHist1D("h_leadPi_pT_from_"+qflavor, 
                    ("Leading #pi from "+qflavor+" quark;p_{T} [GeV];Entries").c_str(), 100, 0, 50);
                TH1D *heta = getHist1D("h_leadPi_eta_from_"+qflavor,
                    ("Leading #pi from "+qflavor+" quark;#eta;Entries").c_str(), 100, -5, 5);
                hpt->Fill(pT1);
                heta->Fill(eta1);
            }
            
            if (anc2.fromPrimaryQuark && anc2.motherQuarkFlavor > 0) {
                std::string qflavor = (anc2.motherQuarkFlavor == 1) ? "down" :
                                     (anc2.motherQuarkFlavor == 2) ? "up" :
                                     (anc2.motherQuarkFlavor == 3) ? "strange" : "heavy";
                TH1D *hpt = getHist1D("h_leadPi_pT_from_"+qflavor,
                    ("Leading #pi from "+qflavor+" quark;p_{T} [GeV];Entries").c_str(), 100, 0, 50);
                TH1D *heta = getHist1D("h_leadPi_eta_from_"+qflavor,
                    ("Leading #pi from "+qflavor+" quark;#eta;Entries").c_str(), 100, -5, 5);
                hpt->Fill(pT2);
                heta->Fill(eta2);
            }
            
            // Correlation between leading pions from opposite quarks
            if (anc1.fromPrimaryQuark && anc2.fromPrimaryQuark &&
                anc1.motherQuarkFlavor > 0 && anc2.motherQuarkFlavor > 0 &&
                anc1.motherQuarkFlavor != anc2.motherQuarkFlavor) {
                TH1D *hdphi_opp = getHist1D("h_leadPi_deltaPhi_oppositeQuarks",
                    "Leading #pi #Delta#phi (opposite mother quarks);#Delta#phi [rad];Entries", 64, -3.2, 3.2);
                hdphi_opp->Fill(dPhi);
            }
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
            
            // Flavor-tagged kaon observables
            if (anc1.fromPrimaryQuark && anc1.motherQuarkFlavor > 0) {
                std::string qflavor = (anc1.motherQuarkFlavor == 1) ? "down" :
                                     (anc1.motherQuarkFlavor == 2) ? "up" :
                                     (anc1.motherQuarkFlavor == 3) ? "strange" : "heavy";
                TH1D *hpt = getHist1D("h_leadK_pT_from_"+qflavor,
                    ("Leading K from "+qflavor+" quark;p_{T} [GeV];Entries").c_str(), 100, 0, 50);
                hpt->Fill(pT1);
            }
            if (anc2.fromPrimaryQuark && anc2.motherQuarkFlavor > 0) {
                std::string qflavor = (anc2.motherQuarkFlavor == 1) ? "down" :
                                     (anc2.motherQuarkFlavor == 2) ? "up" :
                                     (anc2.motherQuarkFlavor == 3) ? "strange" : "heavy";
                TH1D *hpt = getHist1D("h_leadK_pT_from_"+qflavor,
                    ("Leading K from "+qflavor+" quark;p_{T} [GeV];Entries").c_str(), 100, 0, 50);
                hpt->Fill(pT2);
            }
        }

        // ========== ORIGINAL PAIR-BASED OBSERVABLES ==========
        for (size_t a = 0; a < hadrons.size(); ++a) {
            for (size_t b = a + 1; b < hadrons.size(); ++b) {
                int ia = hadrons[a];
                int ib = hadrons[b];

                double px1 = pythia.event[ia].px();
                double py1 = pythia.event[ia].py();
                double px2 = pythia.event[ib].px();
                double py2 = pythia.event[ib].py();

                double phi1 = std::atan2(py1, px1);
                double phi2 = std::atan2(py2, px2);
                double dPhi = phi1 - phi2;
                if (dPhi > M_PI)  dPhi -= 2.0*M_PI;
                if (dPhi < -M_PI) dPhi += 2.0*M_PI;

                double pT1 = std::sqrt(px1*px1 + py1*py1);
                double pT2 = std::sqrt(px2*px2 + py2*py2);
                double pTimb = std::fabs(pT1 - pT2);

                h_dPhi->Fill(dPhi);
                h_pTimbalance->Fill(pTimb);

                if (flavorID != 0) {
                    std::ostringstream dn, pn;
                    dn << "h_dPhi_" << flavorName;
                    pn << "h_pTimbalance_" << flavorName;
                    TH1D *hdf = getHist1D(dn.str(), ("Delta phi ("+flavorName+");Delta phi [rad];Entries").c_str(), 64, -3.2, 3.2);
                    TH1D *hpT = getHist1D(pn.str(), ("pT imbalance ("+flavorName+");|Delta pT| [GeV];Entries").c_str(), 100, 0., 10.);
                    hdf->Fill(dPhi);
                    hpT->Fill(pTimb);
                }

                int qprod = pythia.event[ia].charge() * pythia.event[ib].charge();
                if (qprod < 0) h_OS->Fill(dPhi);
                else           h_SS->Fill(dPhi);

                ++totalPairs;
            }
        }

        // ========== JET-BASED OBSERVABLES ==========
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

            // Jet multiplicity
            int nJets = jets.size();
            h_jetMult->Fill(nJets);
            if (flavorID != 0) {
                TH1D *hmult = getHist1D("h_jetMult_"+flavorName, ("Jet multiplicity ("+flavorName+");N_{jets};Entries").c_str(), 20, 0, 20);
                hmult->Fill(nJets);
            }

            // Event shapes
            EventShapes shapes = calculateEventShapes(fjInputs);
            h_thrust->Fill(shapes.thrust);
            h_sphericity->Fill(shapes.sphericity);
            h_circularity->Fill(shapes.circularity);
            h_thrust_vs_sphericity->Fill(shapes.sphericity, shapes.thrust);
            
            if (flavorID != 0) {
                TH1D *hT = getHist1D("h_thrust_"+flavorName, ("Thrust ("+flavorName+");T;Entries").c_str(), 100, 0, 1);
                TH1D *hS = getHist1D("h_sphericity_"+flavorName, ("Sphericity ("+flavorName+");S;Entries").c_str(), 100, 0, 1);
                TH1D *hC = getHist1D("h_circularity_"+flavorName, ("Circularity ("+flavorName+");C;Entries").c_str(), 100, 0, 1);
                hT->Fill(shapes.thrust);
                hS->Fill(shapes.sphericity);
                hC->Fill(shapes.circularity);
            }

            // Delta phi between leading jets
            if (jets.size() >= 2) {
                double phi1 = jets[0].phi();
                double phi2 = jets[1].phi();
                double dPhi = phi1 - phi2;
                if (dPhi > M_PI)  dPhi -= 2.0*M_PI;
                if (dPhi < -M_PI) dPhi += 2.0*M_PI;
                h_jetDeltaPhi->Fill(dPhi);
                
                if (flavorID != 0) {
                    TH1D *hdphi = getHist1D("h_jetDeltaPhi_"+flavorName, ("Jet #Delta#phi ("+flavorName+");#Delta#phi [rad];Entries").c_str(), 64, -3.2, 3.2);
                    hdphi->Fill(dPhi);
                }
            }

            // ========== LEADING HADRON - JET CORRELATION ==========
            // For each leading pion, find which jet it belongs to and calculate z, jT
            if (pions.size() >= 2) {
                for (size_t ipi = 0; ipi < 2; ++ipi) {
                    int pi_idx = pions[ipi].second;
                    AncestryInfo anc = traceAncestry(pythia.event, pi_idx);
                    
                    double pi_px = pythia.event[pi_idx].px();
                    double pi_py = pythia.event[pi_idx].py();
                    double pi_pz = pythia.event[pi_idx].pz();
                    
                    // Find which jet contains this pion
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
                                // Calculate z (fragmentation variable)
                                double pdotj = pi_px*jpx + pi_py*jpy + pi_pz*jpz;
                                double z = pdotj / jnorm2;
                                
                                // Calculate jT (transverse momentum relative to jet)
                                double px_par = (pdotj / jnorm2) * jpx;
                                double py_par = (pdotj / jnorm2) * jpy;
                                double pz_par = (pdotj / jnorm2) * jpz;
                                double perp_x = pi_px - px_par;
                                double perp_y = pi_py - py_par;
                                double perp_z = pi_pz - pz_par;
                                double jT = std::sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z);
                                
                                // Fill global leading pion z and jT
                                TH1D *hz = getHist1D("h_z_leadPi", "z (leading pions);z;Entries", 100, 0.0, 1.0);
                                TH1D *hjT = getHist1D("h_jT_leadPi", "jT (leading pions);jT [GeV];Entries", 100, 0.0, 5.0);
                                hz->Fill(z);
                                hjT->Fill(jT);
                                
                                // Flavor-tagged z and jT for leading pions
                                if (anc.fromPrimaryQuark && anc.motherQuarkFlavor > 0) {
                                    std::string qflavor = (anc.motherQuarkFlavor == 1) ? "down" :
                                                         (anc.motherQuarkFlavor == 2) ? "up" :
                                                         (anc.motherQuarkFlavor == 3) ? "strange" : "heavy";
                                    TH1D *hzf = getHist1D("h_z_leadPi_from_"+qflavor,
                                        ("z (leading #pi from "+qflavor+");z;Entries").c_str(), 100, 0.0, 1.0);
                                    TH1D *hjTf = getHist1D("h_jT_leadPi_from_"+qflavor,
                                        ("jT (leading #pi from "+qflavor+");jT [GeV];Entries").c_str(), 100, 0.0, 5.0);
                                    hzf->Fill(z);
                                    hjTf->Fill(jT);
                                    
                                    // 2D correlation: z vs jT (TMD-like)
                                    TH2D *h2d = getHist2D("h_z_vs_jT_leadPi_from_"+qflavor,
                                        ("z vs jT (leading #pi from "+qflavor+");jT [GeV];z").c_str(),
                                        50, 0.0, 5.0, 50, 0.0, 1.0);
                                    h2d->Fill(jT, z);
                                }
                            }
                            break;
                        }
                    }
                }
            }
            
            // Same for leading kaons
            if (kaons.size() >= 2) {
                for (size_t ik = 0; ik < 2; ++ik) {
                    int k_idx = kaons[ik].second;
                    AncestryInfo anc = traceAncestry(pythia.event, k_idx);
                    
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
                                
                                if (anc.fromPrimaryQuark && anc.motherQuarkFlavor > 0) {
                                    std::string qflavor = (anc.motherQuarkFlavor == 1) ? "down" :
                                                         (anc.motherQuarkFlavor == 2) ? "up" :
                                                         (anc.motherQuarkFlavor == 3) ? "strange" : "heavy";
                                    TH1D *hzf = getHist1D("h_z_leadK_from_"+qflavor,
                                        ("z (leading K from "+qflavor+");z;Entries").c_str(), 100, 0.0, 1.0);
                                    TH1D *hjTf = getHist1D("h_jT_leadK_from_"+qflavor,
                                        ("jT (leading K from "+qflavor+");jT [GeV];Entries").c_str(), 100, 0.0, 5.0);
                                    hzf->Fill(z);
                                    hjTf->Fill(jT);
                                    
                                    TH2D *h2d = getHist2D("h_z_vs_jT_leadK_from_"+qflavor,
                                        ("z vs jT (leading K from "+qflavor+");jT [GeV];z").c_str(),
                                        50, 0.0, 5.0, 50, 0.0, 1.0);
                                    h2d->Fill(jT, z);
                                }
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
                h_jetPt_vs_Eta->Fill(eta, jetPt);
                
                if (flavorID != 0) {
                    TH1D *heta = getHist1D("h_jetEta_"+flavorName, ("Jet #eta ("+flavorName+");#eta;Entries").c_str(), 100, -5, 5);
                    TH1D *hrap = getHist1D("h_jetRapidity_"+flavorName, ("Jet rapidity ("+flavorName+");y;Entries").c_str(), 100, -5, 5);
                    TH1D *hpt = getHist1D("h_jetPt_"+flavorName, ("Jet pT ("+flavorName+");p_{T} [GeV];Entries").c_str(), 100, 0, 50);
                    heta->Fill(eta);
                    hrap->Fill(rapidity);
                    hpt->Fill(jetPt);
                }

                std::vector<PseudoJet> consts = jet.constituents();
                int nConst = consts.size();
                h_jetConstMult->Fill(nConst);
                
                if (flavorID != 0) {
                    TH1D *hconst = getHist1D("h_jetConstMult_"+flavorName, ("Jet constituent mult. ("+flavorName+");N_{const};Entries").c_str(), 100, 0, 100);
                    hconst->Fill(nConst);
                }

                // Per-constituent z and jT (all hadrons)
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

                    if (flavorID != 0) {
                        std::ostringstream zn, jn;
                        zn << "h_z_" << flavorName;
                        jn << "h_jT_" << flavorName;
                        TH1D *hzf = getHist1D(zn.str(), ("z ("+flavorName+");z;Entries").c_str(), 100, 0.0, 1.0);
                        TH1D *hjTf = getHist1D(jn.str(), ("jT ("+flavorName+");jT [GeV];Entries").c_str(), 100, 0.0, 5.0);
                        hzf->Fill(zfrag);
                        hjTf->Fill(jT);
                    }

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
    std::cout << "Total hadron pairs: " << totalPairs << "\n";
    std::cout << "Events with primary flavor tag: " << eventsWithTag << "\n";
    std::cout << "Events with >=2 leading pions: " << eventsWithLeadingPions << "\n";
    std::cout << "Events with >=2 leading kaons: " << eventsWithLeadingKaons << "\n";
    std::cout << "===================================\n\n";

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
    writeIfNonEmpty(h_dPhi);
    writeIfNonEmpty(h_pTimbalance);
    writeIfNonEmpty(h_OS);
    writeIfNonEmpty(h_SS);
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
    writeIfNonEmpty2D(h_jetPt_vs_Eta);
    writeIfNonEmpty2D(h_thrust_vs_sphericity);

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
    
    for (auto &p : hmap2D) {
        TH2D *h = p.second;
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
