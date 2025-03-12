#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <H5Cpp.h>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TPaveStats.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TVector3.h>
#include <Event/CdWaveformHeader.h>
#include <Event/CdLpmtCalibHeader.h>
#include <Event/CdTriggerHeader.h>
#include <Event/SimHeader.h>
#include <Event/GenHeader.h>
#include <Event/SimEvt.h>
#include <Event/GenEvt.h>
#include <Identifier/Identifier.h>
#include <Identifier/CdID.h>
#include <map>
#include <vector>

const int MAX_TIMESTEPS = 1000;
const int MAX_PMTID = 17612;
const float radius = 20.5;

std::vector<float> flatten(const std::vector<std::vector<float>>& vect) {
    std::vector<float> flat;
    for (const auto& pos : vect) {
        flat.insert(flat.end(), pos.begin(), pos.end());
    }
    return flat;
}

struct Vec3 {
    double x, y, z;

    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    double length2() const { return x * x + y * y + z * z; }
};

bool LineIntersectsSphere(const Vec3& p0, const Vec3& p1, double r) {
    Vec3 d = p1 - p0;        // Direction of the segment
    Vec3 f = p0;             // Since the sphere is at (0,0,0), f = p0

    double a = d.dot(d);
    double b = 2 * f.dot(d);
    double c = f.dot(f) - r * r;

    double discriminant = b * b - 4 * a * c; // Quadratic formula discriminant

    if (discriminant < 0) return false; // No real roots, no intersection

    // Compute roots
    double sqrtD = sqrt(discriminant);
    double t1 = (-b - sqrtD) / (2 * a);
    double t2 = (-b + sqrtD) / (2 * a);

    // Check if at least one solution is within [0,1] (segment range)
    return (t1 >= 0 && t1 <= 1) || (t2 >= 0 && t2 <= 1);
}

class JUNOPMTLocation {
public:
    static JUNOPMTLocation &get_instance() {
        static JUNOPMTLocation instance;
        return instance;
    }
    JUNOPMTLocation(
        std::string input =
            "/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.1.3/data/Detector/Geometry/PMTPos_CD_LPMT.csv") {
        std::ifstream file(input);

        // Skip the first 4 lines
        for (int i = 0; i < 4 && file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); ++i);

        for (std::string line; std::getline(file, line);) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ' ')) {
            tokens.push_back(token);
        }
        // an assumption that the csv file is well formatted
        // and the first column is pmtid
        // and one by one in order
        m_pmt_pos.emplace_back(std::stod(tokens[1]), std::stod(tokens[2]),
                                std::stod(tokens[3]));
        max_pmt_id++;
        }
    }

    double GetPMTX(int pmtid) const { return m_pmt_pos[pmtid].X(); }
    double GetPMTY(int pmtid) const { return m_pmt_pos[pmtid].Y(); }
    double GetPMTZ(int pmtid) const { return m_pmt_pos[pmtid].Z(); }
    double GetPMTTheta(int pmtid) const { return m_pmt_pos[pmtid].Theta(); }
    double GetPMTPhi(int pmtid) const { return m_pmt_pos[pmtid].Phi(); }
    size_t GetMaxPMTID() const { return max_pmt_id; }

private:
    std::vector<TVector3> m_pmt_pos;
    size_t max_pmt_id{};
};

class EventData {
public:
    // Elecsim data (not used)
    std::vector<std::vector<std::vector<short unsigned int>>> adcValues;

    // Detsim data
    // std::vector<std::vector<float>> Edep;
    // std::vector<std::vector<std::vector<float>>> EdepPos;
    // std::vector<std::vector<float>> vertex;
    std::vector<float> energy;
    std::vector<float> zenith; 
    std::vector<int> nuType;

    // Calib data
    std::vector<std::vector<int>> PDGid;
    std::vector<std::vector<std::vector<float>>> featureVector;
    std::map<int, std::vector<std::vector<std::vector<float>>>> pmtData;
    
    int MaxEvents;

    // Add a new member variable to store tracks by event ID
    std::map<int, std::vector<std::vector<float>>> eventTracks;

    void loadElecEDM(const std::string& filename) {
        TFile *elecEDM = TFile::Open(filename.c_str());
        if (!elecEDM || elecEDM->IsZombie()) {
            std::cerr << "Error opening elecsim file" << std::endl;
            return;
        }       
        
        TTree *triggerEvt = (TTree*)elecEDM->Get("Event/CdTrigger/CdTriggerEvt");
        if (!triggerEvt) {
            std::cerr << "Error accessing TTree in elecsim file" << std::endl;
            return;
        }

        JM::CdTriggerEvt *trigger = new JM::CdTriggerEvt();
        triggerEvt->SetBranchAddress("CdTriggerEvt", &trigger);
        triggerEvt->GetBranch("CdTriggerEvt")->SetAutoDelete(true);

        int TriggerMax = triggerEvt->GetEntries();
        std::cout << "TriggerMax: " << TriggerMax << std::endl;

        for(int i = 0; i < TriggerMax; i++){
            triggerEvt->GetEntry(i);

            auto timeTrig = trigger->triggerTime();
            // std::cout << "Trigger Time: " << timeTrig << std::endl;
        }

        elecEDM->Close();
        delete elecEDM;
    }

    void loadDetSimEDM(const std::string& filename) {
        TFile *detEDM = TFile::Open(filename.c_str());
        if (!detEDM || detEDM->IsZombie()) {
            std::cerr << "Error opening detsim file" << std::endl;
            return;
        }

        TTree *genevt = (TTree*)detEDM->Get("Event/Gen/GenEvt");
        if (!genevt) {
            std::cerr << "Error accessing TTree in detsim file" << std::endl;
            return;
        }

        TTree *simevt = (TTree*)detEDM->Get("Event/Sim/SimEvt");
        if (!simevt) {
            std::cerr << "Error accessing TTree in file" << std::endl;
            return;
        }
        JM::SimEvt *se = nullptr; 
        simevt->SetBranchAddress("SimEvt", &se);
        simevt->GetBranch("SimEvt")->SetAutoDelete(true);

        JM::GenEvt* ge = nullptr;
        genevt->SetBranchAddress("GenEvt", &ge);
        genevt->GetBranch("GenEvt")->SetAutoDelete(true);

        int GenMax = genevt->GetEntries();

        std::cout << "GenMax: " << GenMax << std::endl;

        int simid = -1;

        for(int i = 0; i < GenMax; i++){
            genevt->GetEntry(i);
            simevt->GetEntry(i);

            auto hepmc_genevt = ge->getEvent();

            if(simid == se->getEventID()){
                continue;
            }

            simid = se->getEventID();

            std::cout << "-------- Event ID: " << simid << " --------" << std::endl;

            auto tracks = se->getTracksVec();
            int countTrks = 0;

            for(auto itTrk = tracks.begin(); itTrk != tracks.end(); ++itTrk){
                float edep = (*itTrk)->getEdep();
                
                if(edep < 100){
                    continue;
                }

                countTrks++;

                float initTime = (*itTrk)->getInitT();
                float finalTime = (*itTrk)->getExitT();

                std::cout << "Edep: " << edep << std::endl
                          << " InitTime: " << initTime << std::endl
                          << " FinalTime: " << finalTime << std::endl;
            }

            std::cout << "Number of Tracks: " << countTrks << std::endl;

            for(auto iVtx = hepmc_genevt->vertices_begin(); iVtx != hepmc_genevt->vertices_end(); ++iVtx){

                for(auto iPart = (*iVtx)->particles_in_const_begin(); iPart != (*iVtx)->particles_in_const_end(); ++iPart){
                    const auto& part_momentum = (*iPart)->momentum();

                    int pdgid = (*iPart)->pdg_id();

                    if(pdgid == 12 || pdgid == -12 || pdgid == 14 || pdgid == -14){
                        nuType.push_back(pdgid);
                        energy.push_back(part_momentum.e());
                        zenith.push_back(part_momentum.theta());
                    }
                }
            }
        }

        detEDM->Close();
    }

    void loadCalibEDM(const std::string& filename) {
        TFile *calibEDM = TFile::Open(filename.c_str());
        if (!calibEDM || calibEDM->IsZombie()) {
            std::cerr << "Error opening calib file" << std::endl;
            return;
        }

        TTree *calib = (TTree*)calibEDM->Get("Event/CdLpmtCalib/CdLpmtCalibEvt");
        if (!calib) {
            std::cerr << "Error accessing TTree in calib file" << std::endl;
            return;
        }

        TTree *simCalib = (TTree*)calibEDM->Get("Event/Sim/SimEvt");
        if (!simCalib) {
            std::cerr << "Error accessing TTree in calib file" << std::endl;
            return;
        }

        JM::SimEvt *sim = new JM::SimEvt();
        simCalib->SetBranchAddress("SimEvt", &sim);
        simCalib->GetBranch("SimEvt")->SetAutoDelete(true);

        JM::CdLpmtCalibEvt *cal = new JM::CdLpmtCalibEvt();
        calib->SetBranchAddress("CdLpmtCalibEvt", &cal);
        calib->GetBranch("CdLpmtCalibEvt")->SetAutoDelete(true);

        int calibMax = calib->GetEntries();

        std::cout << "CalibMax: " << calibMax << std::endl;

        TH1F *invalidCpNo = new TH1F("invalidCpNo", "Invalid Copy Number", 1000, 279000000, 401000000);

        MaxEvents = calibMax;

        for(int i = 0; i < calibMax; i++){
            calib->GetEntry(i);
            simCalib->GetEntry(i);

            int evtid = sim->getEventID();

            const auto &calibCh = cal->calibPMTCol();

            int calibSize = calibCh.size();
            
            std::vector<int> calibChID;
            calibChID.reserve(calibSize);
            std::vector<std::vector<float>> calibFeatures;
            calibFeatures.reserve(MAX_PMTID);

            for(auto it = calibCh.begin(); it != calibCh.end(); ++it){
                int CalibId = CdID::module(Identifier((*it)->pmtId()));
                
                if(CalibId < 0){
                    invalidCpNo->Fill((*it)->pmtId());
                    continue;
                }

                float phiIt = JUNOPMTLocation::get_instance().GetPMTPhi(CalibId);
                float thetaIt = JUNOPMTLocation::get_instance().GetPMTTheta(CalibId);

                float maxChargeIt   = (*it)->maxCharge();
                float maxTimeIt     = (*it)->time()[(*it)->maxChargeIndex()];
                float sumChargeIt   = (*it)->sumCharge();
                float FHTIt         = (*it)->firstHitTime();
                
                std::vector<float> features;

                features.push_back(phiIt);
                features.push_back(thetaIt);
                features.push_back(FHTIt);
                features.push_back(maxTimeIt);
                features.push_back(maxChargeIt);
                features.push_back(sumChargeIt);

                calibFeatures.push_back(features);
                features.clear();
                calibChID.push_back(CalibId);
            }

            // for(int j = 0; j < MAX_PMTID; j++){
            //     if(std::find(calibChID.begin(), calibChID.end(), j) == calibChID.end()){
            //         std::vector<float> features;

                    // features.push_back(JUNOPMTLocation::get_instance().GetPMTPhi(j));
                    // features.push_back(JUNOPMTLocation::get_instance().GetPMTTheta(j));
                    // features.push_back(0);
                    // features.push_back(0);
                    // features.push_back(0);
                    // features.push_back(0);

            //         calibFeatures.push_back(features);
            //         features.clear();
            //     }
            // }

            featureVector.push_back(calibFeatures);
            pmtData[evtid].push_back(calibFeatures);
            calibFeatures.clear();
        }

        delete cal;
        delete calib;

        calibEDM->Close();
        delete calibEDM;
    }

    void h5Save(const std::string& filename) {
    try {
        // Create or overwrite the HDF5 file
        H5::H5File file(filename, H5F_ACC_TRUNC);

        std::cout << "Saving " << pmtData.size() << " events to " << filename << std::endl;

        for (const auto& event : pmtData) {
            int eventID = event.first;
            const auto& iterations = event.second;

            // Create a group for each event
            std::string eventGroupPath = "/Event_" + std::to_string(eventID);
            H5::Group eventGroup(file.createGroup(eventGroupPath));

            std::cout << "Event " << eventID << " has " << iterations.size() << " iterations." << std::endl;

            H5::DataSpace scalarDataspace = H5::DataSpace(H5S_SCALAR);
            
            // Save event-level data
            H5::DataSet energyDataset = eventGroup.createDataSet("Energy", H5::PredType::NATIVE_FLOAT, scalarDataspace);
            energyDataset.write(&energy[eventID], H5::PredType::NATIVE_FLOAT);

            H5::DataSet zenithDataset = eventGroup.createDataSet("Zenith", H5::PredType::NATIVE_FLOAT, scalarDataspace);
            zenithDataset.write(&zenith[eventID], H5::PredType::NATIVE_FLOAT);

            H5::DataSet nuTypeDataset = eventGroup.createDataSet("NuType", H5::PredType::NATIVE_INT, scalarDataspace);
            nuTypeDataset.write(&nuType[eventID], H5::PredType::NATIVE_FLOAT);

            for (size_t iterIdx = 0; iterIdx < iterations.size(); ++iterIdx) {
                const auto& pmtList = iterations[iterIdx];

                // Create a subgroup for each iteration
                std::string iterationGroupPath = eventGroupPath + "/Trigger_" + std::to_string(iterIdx);
                H5::Group iterationGroup(file.createGroup(iterationGroupPath));

                if (pmtList.empty()) {
                    std::cerr << "Warning: Empty PMT list in Event " << eventID << " Iteration " << iterIdx << std::endl;
                    continue;
                }

                // Determine dimensions for HDF5 dataset
                hsize_t dims[2] = { pmtList.size(), pmtList[0].size() };  // [Number of PMTs, Number of Features]

                // Create dataspace
                H5::DataSpace dataspace(2, dims);

                // Flatten PMT data for writing
                std::vector<float> flatData;
                for (const auto& pmt : pmtList) {
                    flatData.insert(flatData.end(), pmt.begin(), pmt.end());
                }

                // Create dataset in the iteration group
                H5::DataSet dataset = iterationGroup.createDataSet(
                    "PMT_Features", H5::PredType::NATIVE_FLOAT, dataspace
                );

                // Write data to dataset
                dataset.write(flatData.data(), H5::PredType::NATIVE_FLOAT);

                std::cout << "Saved Iteration " << iterIdx << " in Event " << eventID 
                        << " with " << pmtList.size() << " PMTs." << std::endl;
            }
        }

        file.close();
        std::cout << "HDF5 file saved successfully!\n";

    } catch (H5::FileIException& error) {
        std::cerr << "HDF5 File Error: " << error.getDetailMsg() << std::endl;
    } catch (H5::GroupIException& error) {
        std::cerr << "HDF5 Group Error: " << error.getDetailMsg() << std::endl;
    } catch (H5::DataSetIException& error) {
        std::cerr << "HDF5 Dataset Error: " << error.getDetailMsg() << std::endl;
    } catch (H5::DataSpaceIException& error) {
        std::cerr << "HDF5 Dataspace Error: " << error.getDetailMsg() << std::endl;
    } catch (std::exception& error) {
        std::cerr << "Standard Exception: " << error.what() << std::endl;
    }
    }

};

class SignalProcessor {
public:
    // This is now a bit useless
    static double PeakCharge(const std::vector<short unsigned int>& signal) {
        return *std::max_element(signal.begin(), signal.end());
    }

    static double PeakChargeTime(const std::vector<short unsigned int>& signal) {
        auto maxIter = std::max_element(signal.begin(), signal.end());
        return std::distance(signal.begin(), maxIter);
    }

    static double FHT(const std::vector<short unsigned int>& signal) {
        double minValue = *std::min_element(signal.begin(), signal.end());
        double threshold = 0.5 * (PeakCharge(signal) - minValue);

        for(size_t i = 0; i < signal.size(); i++){
            if((signal[i] - minValue) > threshold){
                return i;
            }
        }
        return -1;
    }

    static double totalCharge(const std::vector<short unsigned int>& signal) {
        double total = 0;
        for(size_t i = 0; i < signal.size(); i++){
            total += signal[i];
        }
        return total;
    }
};

std::string nuTypeString(int pdg) {
    if(pdg == 12) return "NuE";
    else if(pdg == -12) return "NuEBar";
    else if(pdg == 14) return "NuMu";
    else if(pdg == -14) return "NuMuBar";
    else return "Unknown";
}

int main() {
    EventData eventData;

    std::string detsimfile = "data/productions/J24_GENIE_Flat_FC/detsim_nu_e/detsim-0.root";
    std::string calibfile = "data/productions/J24_GENIE_Flat_FC/rec_nu_e/rec-0.root";

    eventData.loadDetSimEDM(detsimfile);
    // eventData.loadCalibEDM(calibfile);

    // std::string filename = "data/libAtmos.h5";
    // eventData.h5Save(filename);

    return 0;
}