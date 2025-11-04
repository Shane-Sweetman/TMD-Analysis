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
        int status = event[current].status(); // status code, whether it's a hard-process particle, intermediate, or final
        
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
                double dot = std::abs(p.px()*nx + p.py()*ny + p.pz()*nz); // magnitude of the particle's momentum along the chosen direction
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

TH1D* getHist1D(const std::string &name, const std::string &title, // 1D histogram retrieval/creation
                int nbins = 64, double xmin = -3.2, double xmax = 3.2) { // default binning
    auto it = hmap.find(name); // check if histogram exists
    if (it != hmap.end()) return it->second; // return existing histogram
    TH1D *h = new TH1D(name.c_str(), title.c_str(), nbins, xmin, xmax); // create new histogram
    h->SetDirectory(nullptr); // prevent ROOT from managing memory
    hmap[name] = h; // store in map
    return h; // return new histogram
}

int main() {
    
    
    
    
    TFile *fout = new TFile("ee_hadron_corr.root", "RECREATE"); // creates root file

    // Jet-based observables
    TH1D *h_jetMult     = new TH1D("h_jetMult", "Jet multiplicity per event;N_{jets};Entries", 20, 0, 20); // number of jets
    TH1D *h_jetPt       = new TH1D("h_jetPt", "Jet pT spectrum;p_{T} [GeV];Entries", 100, 0, 50); // jet transverse momentum
    TH1D *h_jetEta      = new TH1D("h_jetEta", "Jet pseudorapidity;#eta;Entries", 100, -5, 5); // jet pseudorapidity
    TH1D *h_jetRapidity = new TH1D("h_jetRapidity", "Jet rapidity;y;Entries", 100, -5, 5); // jet rapidity
    TH1D *h_jetDeltaPhi = new TH1D("h_jetDeltaPhi", "Delta phi between leading jets;#Delta#phi [rad];Entries", 64, -3.2, 3.2); // azimuthal angle difference between leading jets
    TH1D *h_jetConstMult = new TH1D("h_jetConstMult", "Jet constituent multiplicity;N_{constituents};Entries", 100, 0, 100); // number of constituents in jets
    
    // Event shapes
    TH1D *h_thrust      = new TH1D("h_thrust", "Event thrust;T;Entries", 100, 0, 1); // thrust distribution
    TH1D *h_sphericity  = new TH1D("h_sphericity", "Event sphericity;S;Entries", 100, 0, 1); // sphericity distribution
    TH1D *h_circularity = new TH1D("h_circularity", "Event circularity;C;Entries", 100, 0, 1); // circularity distribution

    // Leading hadron observables 
    TH1D *h_leadPi_pT = new TH1D("h_leadPi_pT", "Leading pion pT;p_{T} [GeV];Entries", 100, 0, 50); // leading pion transverse momentum
    TH1D *h_leadPi_eta = new TH1D("h_leadPi_eta", "Leading pion eta;#eta;Entries", 100, -5, 5); // leading pion pseudorapidity
    TH1D *h_leadPi_deltaPhi = new TH1D("h_leadPi_deltaPhi", "Delta phi between leading pions;#Delta#phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_leadK_pT = new TH1D("h_leadK_pT", "Leading kaon pT;p_{T} [GeV];Entries", 100, 0, 50);
    TH1D *h_leadK_deltaPhi = new TH1D("h_leadK_deltaPhi", "Delta phi between leading kaons;#Delta#phi [rad];Entries", 64, -3.2, 3.2);

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

    // Pythia setup
    Pythia pythia;
    pythia.readString("Beams:idA = -11"); // electron beam
    pythia.readString("Beams:idB = 11"); // positron beam
    pythia.readString("Beams:eCM = 91.2"); // center-of-mass energy at Z pole
    pythia.readString("PDF:lepton = off"); // no PDF for leptons
    pythia.readString("HadronLevel:all = on"); // enable hadronization
    pythia.readString("WeakSingleBoson:ffbar2gmZ = on"); // enable e+e- -> Z/gamma*
    pythia.readString("Random:setSeed = on"); // set random seed
    pythia.readString("Random:seed = 123456788"); // specific seed for reproducibility

    if (!pythia.init()) { // initialize Pythia
        std::cerr << "Pythia initialization failed\n"; // error handling
        return 1; // exit if failed
    }

    const int nEvents = 20000; // number of events to generate
    int eventsWithLeadingPions = 0; // events with leading pions
    int eventsWithLeadingKaons = 0; // events with leading kaons

    // FastJet parameters
    double R = 0.6; // jet radius parameter
    double jetPtMin = 1.0; // minimum jet pT

    // Event loop
    for (int ievt = 0; ievt < nEvents; ++ievt) { // loop over events
        if (!pythia.next()) continue; // generate next event, skip if fails

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
        for (int idx : hadrons) { // loop over hadrons
            int pdg = std::abs(pythia.event[idx].id()); // absolute PDG ID
            if (pdg == 211) { // check for charged pions
                double p = pythia.event[idx].pAbs(); // get momentum magnitude
                pions.push_back({p, idx}); // store pair
            }
        }
        std::sort(pions.begin(), pions.end(), std::greater<std::pair<double,int>>()); // sort by momentum descending
        
        // Find leading kaons (±321)
        std::vector<std::pair<double, int>> kaons; // (momentum, index)
        for (int idx : hadrons) { // loop over hadrons
            int pdg = std::abs(pythia.event[idx].id()); // absolute PDG ID
            if (pdg == 321) { // check for charged kaons
                double p = pythia.event[idx].pAbs(); // get momentum magnitude
                kaons.push_back({p, idx}); // store pair
            }
        }
        std::sort(kaons.begin(), kaons.end(), std::greater<std::pair<double,int>>()); // sort by momentum descending

        // Analyze leading two pions
        if (pions.size() >= 2) { // at least two pions found
            ++eventsWithLeadingPions; // count events with leading pions
            int pi1_idx = pions[0].second; // index of leading pion
            int pi2_idx = pions[1].second; // index of subleading pion
            
            // Trace ancestry
            AncestryInfo anc1 = traceAncestry(pythia.event, pi1_idx); // trace first pion
            AncestryInfo anc2 = traceAncestry(pythia.event, pi2_idx); // trace second pion
            
            // Kinematics
            double pT1 = pythia.event[pi1_idx].pT(); // transverse momentum of first pion
            double pT2 = pythia.event[pi2_idx].pT(); // transverse momentum of second pion
            double eta1 = pythia.event[pi1_idx].eta(); // pseudorapidity of first pion
            double eta2 = pythia.event[pi2_idx].eta(); //   pseudorapidity of second pion
            double phi1 = pythia.event[pi1_idx].phi(); // azimuthal angle of first pion
            double phi2 = pythia.event[pi1_idx].phi(); // azimuthal angle of second pion
            
            h_leadPi_pT->Fill(pT1); // fill leading pion pT histogram
            h_leadPi_pT->Fill(pT2); // fill subleading pion pT histogram
            h_leadPi_eta->Fill(eta1); // fill leading pion eta histogram
            h_leadPi_eta->Fill(eta2); // fill subleading pion eta histogram
            
            double dPhi = phi1 - phi2; // calculate delta phi
            if (dPhi > M_PI) dPhi -= 2.0*M_PI; // wrap to [-pi, pi]
            if (dPhi < -M_PI) dPhi += 2.0*M_PI; // wrap to [-pi, pi]
            h_leadPi_deltaPhi->Fill(dPhi); // fill delta phi histogram
        }
        
        // Analyze leading two kaons
        if (kaons.size() >= 2) { // at least two kaons found
            ++eventsWithLeadingKaons; // count events with leading kaons
            int k1_idx = kaons[0].second; // index of leading kaon
            int k2_idx = kaons[1].second; // index of subleading kaon
            
            AncestryInfo anc1 = traceAncestry(pythia.event, k1_idx); // trace first kaon
            AncestryInfo anc2 = traceAncestry(pythia.event, k2_idx); // trace second kaon
            
            double pT1 = pythia.event[k1_idx].pT(); // transverse momentum of first kaon
            double pT2 = pythia.event[k2_idx].pT(); // transverse momentum of second kaon
            double phi1 = pythia.event[k1_idx].phi(); // azimuthal angle of first kaon
            double phi2 = pythia.event[k2_idx].phi(); // azimuthal angle of second kaon
            
            h_leadK_pT->Fill(pT1); // fill leading kaon pT histogram
            h_leadK_pT->Fill(pT2); //   fill subleading kaon pT histogram
            
            double dPhi = phi1 - phi2; // calculate delta phi
            if (dPhi > M_PI) dPhi -= 2.0*M_PI; // wrap to [-pi, pi]
            if (dPhi < -M_PI) dPhi += 2.0*M_PI; // wrap to [-pi, pi]
            h_leadK_deltaPhi->Fill(dPhi); // fill delta phi histogram
        }

        // ========== JET-BASED OBSERVABLES ==========
        if (!hadrons.empty()) { // ensure there are hadrons to cluster
            std::vector<PseudoJet> fjInputs; // prepare FastJet inputs
            fjInputs.reserve(hadrons.size()); // reserve space
            for (int idx : hadrons) { // loop over hadrons
                double px = pythia.event[idx].px(); // get x momentum component
                double py = pythia.event[idx].py(); // get y momentum component
                double pz = pythia.event[idx].pz(); // get z momentum component
                double E  = pythia.event[idx].e(); // get energy
                PseudoJet pj(px, py, pz, E); // create PseudoJet
                pj.set_user_index(idx); // store original index
                fjInputs.push_back(pj); // add to FastJet inputs
            }

            JetDefinition jetDef(antikt_algorithm, R); // define jet algorithm
            ClusterSequence cs(fjInputs, jetDef); // cluster jets
            std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin)); // get jets above pT threshold

            // Jet multiplicity
            int nJets = jets.size(); // number of jets found
            h_jetMult->Fill(nJets); // fill jet multiplicity histogram

            // Event shapes
            EventShapes shapes = calculateEventShapes(fjInputs); // calculate event shapes
            h_thrust->Fill(shapes.thrust); // fill thrust histogram
            h_sphericity->Fill(shapes.sphericity); // fill sphericity histogram
            h_circularity->Fill(shapes.circularity); // fill circularity histogram
            
            // Delta phi between leading jets
            if (jets.size() >= 2) { // at least two jets found
                double phi1 = jets[0].phi(); // leading jet phi
                double phi2 = jets[1].phi(); // subleading jet phi
                double dPhi = phi1 - phi2; // calculate delta phi
                if (dPhi > M_PI)  dPhi -= 2.0*M_PI; // wrap to [-pi, pi]
                if (dPhi < -M_PI) dPhi += 2.0*M_PI; // wrap to [-pi, pi]
                h_jetDeltaPhi->Fill(dPhi); // fill delta phi histogram
            }

            // ========== LEADING HADRON - JET CORRELATION ==========
            // For each leading pion, find which jet it belongs to and calculate z, jT
            if (pions.size() >= 2) { // at least two pions
                for (size_t ipi = 0; ipi < 2; ++ipi) { // loop over leading two pions
                    int pi_idx = pions[ipi].second; // get pion index
                    
                    double pi_px = pythia.event[pi_idx].px(); // pion x momentum
                    double pi_py = pythia.event[pi_idx].py(); // pion y momentum
                    double pi_pz = pythia.event[pi_idx].pz(); // pion z momentum
                    
                    // Find which jet contains this pion
                    for (const PseudoJet &jet : jets) { // loop over jets
                        std::vector<PseudoJet> consts = jet.constituents(); // get jet constituents
                        bool found = false; // flag for finding pion
                        for (const PseudoJet &c : consts) { // loop over constituents
                            if (c.user_index() == pi_idx) { // check if pion found
                                found = true; // set flag
                                break; // exit loop
                            }
                        }
                        
                        if (found) { // if pion found in this jet
                            double jpx = jet.px(); // jet x momentum
                            double jpy = jet.py(); // jet y momentum
                            double jpz = jet.pz(); // jet z momentum
                            double jnorm2 = jpx*jpx + jpy*jpy + jpz*jpz; // jet momentum squared
                            
                            if (jnorm2 > 0) { // avoid division by zero
                                // Calculate z (fragmentation variable) 
                                double pdotj = pi_px*jpx + pi_py*jpy + pi_pz*jpz; // dot product
                                double z = pdotj / jnorm2; // calculate z
                                
                                // Calculate jT (transverse momentum relative to jet)
                                double px_par = (pdotj / jnorm2) * jpx; // parallel component x
                                double py_par = (pdotj / jnorm2) * jpy; // parallel component y
                                double pz_par = (pdotj / jnorm2) * jpz; // parallel component z
                                double perp_x = pi_px - px_par; // perpendicular component x
                                double perp_y = pi_py - py_par; // perpendicular component y
                                double perp_z = pi_pz - pz_par; // perpendicular component z
                                double jT = std::sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z); // calculate jT
                                
                                // Fill global leading pion z and jT
                                TH1D *hz = getHist1D("h_z_leadPi", "z (leading pions);z;Entries", 100, 0.0, 1.0); // create/fill z histogram
                                TH1D *hjT = getHist1D("h_jT_leadPi", "jT (leading pions);jT [GeV];Entries", 100, 0.0, 5.0); // create/fill jT histogram
                                hz->Fill(z); // fill z
                                hjT->Fill(jT); // fill jT
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
            for (const PseudoJet &jet : jets) { // loop over jets
                double eta = jet.eta(); // jet pseudorapidity
                double rapidity = jet.rap(); // jet rapidity
                double jetPt = jet.pt(); // jet transverse momentum
                
                h_jetEta->Fill(eta); // fill jet eta histogram
                h_jetRapidity->Fill(rapidity); // fill jet rapidity histogram
                h_jetPt->Fill(jetPt);// fill jet pT histogram

                std::vector<PseudoJet> consts = jet.constituents(); // get jet constituents
                int nConst = consts.size(); // number of constituents
                h_jetConstMult->Fill(nConst); // fill jet constituent multiplicity histogram

                // Per-constituent z and jT (all hadrons)
                double jpx = jet.px(); // jet x momentum
                double jpy = jet.py(); // jet y momentum
                double jpz = jet.pz(); // jet z momentum
                double jnorm2 = jpx*jpx + jpy*jpy + jpz*jpz; // jet momentum squared
                if (jnorm2 <= 0.0) continue; // skip if zero momentum

                for (const PseudoJet &c : consts) { // loop over constituents
                    double cpx = c.px(); // constituent x momentum
                    double cpy = c.py(); // constituent y momentum
                    double cpz = c.pz(); // constituent z momentum

                    double pdotj = cpx*jpx + cpy*jpy + cpz*jpz; // dot product
                    double zfrag = pdotj / jnorm2; // fragmentation variable

                    double px_par = (pdotj / jnorm2) * jpx; // parallel component x
                    double py_par = (pdotj / jnorm2) * jpy;// parallel component y
                    double pz_par = (pdotj / jnorm2) * jpz; // parallel component z
                    double perp_x = cpx - px_par; // perpendicular component x
                    double perp_y = cpy - py_par; // perpendicular component y
                    double perp_z = cpz - pz_par; // perpendicular component z
                    double jT = std::sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z); // transverse momentum rel. to jet

                    TH1D *hz = getHist1D("h_z_all", "z (hadron/jet);z;Entries", 100, 0.0, 1.0); // create/fill z histogram
                    TH1D *hjT = getHist1D("h_jT_all", "jT (hadron rel. to jet) [GeV];jT [GeV];Entries", 100, 0.0, 5.0); // create/fill jT histogram
                    hz->Fill(zfrag); // fill z
                    hjT->Fill(jT); // fill jT

                    int pythia_idx = c.user_index(); // get original Pythia index
                    if (pythia_idx >= 0 && pythia_idx < pythia.event.size()) { // check index validity
                        int pdgid = std::abs(pythia.event[pythia_idx].id()); // get absolute PDG ID
                        if (pdgid == 211) { // charged pions
                            TH1D *hz_pi = getHist1D("h_z_pi", "z (pions);z;Entries", 100, 0.0, 1.0); // create/fill pion z histogram
                            hz_pi->Fill(zfrag); // fill pion z
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
