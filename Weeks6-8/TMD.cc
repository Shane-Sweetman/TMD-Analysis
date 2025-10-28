// TMD.cc
// e+ e- -> qqbar, hadron-pair & jet-based TMD-like observables with FastJet (anti-kt).
// Extended with: jet multiplicity, eta/rapidity, pT spectrum, Delta phi, event shapes, constituent multiplicity

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

// ---------- Helper: get primary hard-process quark flavor ----------
int getPrimaryQuarkFlavor(const Event &event) {
    // Return codes: 1=down,2=up,3=strange,4=heavy (c/b),0=unknown
    int n = event.size();
    // First pass: status 23 (common for outgoing hard partons)
    for (int i = 0; i < n; ++i) {
        int st = event[i].status();
        int id = std::abs(event[i].id());
        if (st == 23 && id >= 1 && id <= 5) {
            if (id == 1) return 1;
            if (id == 2) return 2;
            if (id == 3) return 3;
            // charm (4) or bottom (5) -> heavy
            return 4;
        }
    }
    // Fallback: any particle with PDG 1..5
    for (int i = 0; i < n; ++i) {
        int id = std::abs(event[i].id());
        if (id >= 1 && id <= 5) {
            if (id == 1) return 1;
            if (id == 2) return 2;
            if (id == 3) return 3;
            return 4;
        }
    }
    return 0;
}

// ---------- Event Shape Calculations ----------
struct EventShapes {
    double thrust;
    double sphericity;
    double circularity;
};

// Calculate thrust: maximize sum of |p_i . n| / sum |p_i|
double calculateThrust(const std::vector<PseudoJet> &particles) {
    if (particles.empty()) return 0.0;
    
    double totalP = 0.0;
    for (const auto &p : particles) {
        totalP += std::sqrt(p.px()*p.px() + p.py()*p.py() + p.pz()*p.pz());
    }
    if (totalP <= 0.0) return 0.0;
    
    double maxThrust = 0.0;
    // Sample many directions to find maximum
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

// Calculate sphericity from momentum tensor eigenvalues
EventShapes calculateEventShapes(const std::vector<PseudoJet> &particles) {
    EventShapes shapes = {0.0, 0.0, 0.0};
    if (particles.empty()) return shapes;
    
    // Build momentum tensor (3x3)
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
    
    // Normalize
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S[i][j] /= totalP2;
        }
    }
    
    // Use ROOT to find eigenvalues
    TMatrixDSym matrix(3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix(i,j) = S[i][j];
        }
    }
    
    TMatrixDSymEigen eigen(matrix);
    TVectorD eigenvalues = eigen.GetEigenValues();
    
    // Sort eigenvalues: lambda1 >= lambda2 >= lambda3
    std::vector<double> eigs = {eigenvalues[0], eigenvalues[1], eigenvalues[2]};
    std::sort(eigs.begin(), eigs.end(), std::greater<double>());
    
    double lambda1 = eigs[0];
    double lambda2 = eigs[1];
    double lambda3 = eigs[2];
    
    // Sphericity: S = 3/2 * (lambda2 + lambda3)
    shapes.sphericity = 1.5 * (lambda2 + lambda3);
    
    // Thrust approximation from eigenvalues
    shapes.thrust = calculateThrust(particles);
    
    // Circularity (transverse plane): 2*min(lambda_x, lambda_y) / (lambda_x + lambda_y)
    // Build 2D transverse momentum tensor
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
        
        // Eigenvalues of 2x2 matrix
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

// ---------- On-demand histogram map ----------
static std::map<std::string, TH1D*> hmap;
static std::map<std::string, TH2D*> hmap2D;

// Helper to get-or-create 1D histogram (on-demand)
TH1D* getHist1D(const std::string &name, const std::string &title,
                int nbins = 64, double xmin = -3.2, double xmax = 3.2) {
    auto it = hmap.find(name);
    if (it != hmap.end()) return it->second;
    TH1D *h = new TH1D(name.c_str(), title.c_str(), nbins, xmin, xmax);
    h->SetDirectory(nullptr);
    hmap[name] = h;
    return h;
}

// Helper to get-or-create 2D histogram (on-demand)
TH2D* getHist2D(const std::string &name, const std::string &title,
                int nbinsx, double xmin, double xmax,
                int nbinsy, double ymin, double ymax) {
    auto it = hmap2D.find(name);
    if (it != hmap2D.end()) return it->second;
    TH2D *h = new TH2D(name.c_str(), title.c_str(), nbinsx, xmin, xmax, nbinsy, ymin, ymax);
    h->SetDirectory(nullptr);
    hmap2D[name] = h;
    return h;
}

int main() {
    TFile *fout = new TFile("ee_hadron_corr.root", "RECREATE");

    // Global pair-based histograms (always created)
    TH1D *h_dPhi        = new TH1D("h_dPhi", "Delta phi between hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_pTimbalance = new TH1D("h_pTimbalance", "pT imbalance;|Delta pT| [GeV];Entries", 100, 0, 10);
    TH1D *h_OS          = new TH1D("h_OS", "Opposite-sign hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2);
    TH1D *h_SS          = new TH1D("h_SS", "Same-sign hadron pairs;Delta phi [rad];Entries", 64, -3.2, 3.2);

    // New jet-based observables (global)
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

    // FastJet parameters
    double R = 0.6;         // anti-kt radius
    double jetPtMin = 1.0;  // GeV

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

        // Collect final-state charged hadrons indices
        std::vector<int> hadrons;
        hadrons.reserve(64);
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            if (!pythia.event[i].isCharged()) continue;
            hadrons.push_back(i);
        }

        // --- Pair-based observables (as before) ---
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

                // flavor-specific pair histograms (on-demand)
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

        // --- FastJet clustering for jet-based observables ---
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

            // anti-kt clustering
            JetDefinition jetDef(antikt_algorithm, R);
            ClusterSequence cs(fjInputs, jetDef);
            std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(jetPtMin));

            // 1. Jet multiplicity
            int nJets = jets.size();
            h_jetMult->Fill(nJets);
            if (flavorID != 0) {
                TH1D *hmult = getHist1D("h_jetMult_"+flavorName, ("Jet multiplicity ("+flavorName+");N_{jets};Entries").c_str(), 20, 0, 20);
                hmult->Fill(nJets);
            }

            // Calculate event shapes using all input particles
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

            // 4. Delta phi between leading two jets
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

            // Loop over jets for per-jet observables
            for (const PseudoJet &jet : jets) {
                // 2. Jet eta and rapidity
                double eta = jet.eta();
                double rapidity = jet.rap();
                h_jetEta->Fill(eta);
                h_jetRapidity->Fill(rapidity);
                
                // 3. Jet pT spectrum
                double jetPt = jet.pt();
                h_jetPt->Fill(jetPt);
                h_jetPt_vs_Eta->Fill(eta, jetPt);
                
                // Flavor-specific jet observables
                if (flavorID != 0) {
                    TH1D *heta = getHist1D("h_jetEta_"+flavorName, ("Jet #eta ("+flavorName+");#eta;Entries").c_str(), 100, -5, 5);
                    TH1D *hrap = getHist1D("h_jetRapidity_"+flavorName, ("Jet rapidity ("+flavorName+");y;Entries").c_str(), 100, -5, 5);
                    TH1D *hpt = getHist1D("h_jetPt_"+flavorName, ("Jet pT ("+flavorName+");p_{T} [GeV];Entries").c_str(), 100, 0, 50);
                    heta->Fill(eta);
                    hrap->Fill(rapidity);
                    hpt->Fill(jetPt);
                }

                // 6. Jet constituent multiplicity
                std::vector<PseudoJet> consts = jet.constituents();
                int nConst = consts.size();
                h_jetConstMult->Fill(nConst);
                
                if (flavorID != 0) {
                    TH1D *hconst = getHist1D("h_jetConstMult_"+flavorName, ("Jet constituent mult. ("+flavorName+");N_{const};Entries").c_str(), 100, 0, 100);
                    hconst->Fill(nConst);
                }

                // Per-constituent z and jT (existing code)
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

    std::cout << "Done. Processed " << nEvents << " events and " << totalPairs << " hadron pairs.\n";
    std::cout << "Events with primary flavor tag (non-zero): " << eventsWithTag << "\n";

    // Write histograms: global (if non-empty) + on-demand (non-empty only)
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
    writeIfNonEmpty2D(h_jetPt_vs_Eta);
    writeIfNonEmpty2D(h_thrust_vs_sphericity);

    // Write on-demand 1D histograms
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
    
    // Write on-demand 2D histograms
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