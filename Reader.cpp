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
#include <unordered_set>

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
    // Detsim data
    std::vector<float> energy;
    std::vector<float> zenith; 
    std::vector<int> nuType;

    // Calib data
    std::vector<std::vector<int>> PDGid;
    std::vector<std::vector<std::vector<float>>> featureVector;
    std::map<int, std::vector<std::vector<std::vector<float>>>> pmtData;
    std::unordered_set<int> evtList;
    
    int MaxEvents;

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
        int SimMax = simevt->GetEntries();

        std::cout << "GenMax: " << GenMax << std::endl;
        std::cout << "SimMax: " << SimMax << std::endl;

        int simid = -1;

        for(int i = 0; i < GenMax; i++){
            genevt->GetEntry(i);
            simevt->GetEntry(i);

            auto hepmc_genevt = ge->getEvent();

            if(simid == se->getEventID()){
                continue;
            }

            simid = se->getEventID();

            if(evtList.find(simid) != evtList.end()){
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

        JM::CdLpmtCalibEvt *cal = new JM::CdLpmtCalibEvt();
        calib->SetBranchAddress("CdLpmtCalibEvt", &cal);
        calib->GetBranch("CdLpmtCalibEvt")->SetAutoDelete(true);

        JM::SimEvt *sim = new JM::SimEvt();
        simCalib->SetBranchAddress("SimEvt", &sim);
        simCalib->GetBranch("SimEvt")->SetAutoDelete(true);

        int calibMax = calib->GetEntries();
        int simMax = simCalib->GetEntries();

        std::cout << "CalibMax: " << calibMax << std::endl;
        std::cout << "SimCalibMax: " << simMax << std::endl;

        for(int i = 0; i < calibMax; i++){
            calib->GetEntry(i);
            simCalib->GetEntry(i);

            int evtid = sim->getEventID();

            if (evtList.find(evtid) != evtList.end()) {
                continue;
            }

            evtList.insert(evtid);

            const auto &calibCh = cal->calibPMTCol();

            int calibSize = calibCh.size();
            
            std::vector<int> calibChID;
            calibChID.reserve(calibSize);
            std::vector<std::vector<float>> calibFeatures;
            calibFeatures.reserve(MAX_PMTID);

            for(auto it = calibCh.begin(); it != calibCh.end(); ++it){
                int CalibId = CdID::module(Identifier((*it)->pmtId()));

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

        std::cout << "Saving " << featureVector.size() << " events to " << filename << std::endl;

        for (int iH5 = 0; iH5 < featureVector.size(); iH5++) {
            // Create a group for each event
            std::string eventGroupPath = "/Event_" + std::to_string(iH5);
            H5::Group eventGroup(file.createGroup(eventGroupPath));

            H5::DataSpace scalarDataspace = H5::DataSpace(H5S_SCALAR);
            
            // Save event-level data
            H5::DataSet energyDataset = eventGroup.createDataSet("energy", H5::PredType::NATIVE_FLOAT, scalarDataspace);
            energyDataset.write(&energy[iH5], H5::PredType::NATIVE_FLOAT);
            energyDataset.close();

            H5::DataSet zenithDataset = eventGroup.createDataSet("zenith", H5::PredType::NATIVE_FLOAT, scalarDataspace);
            zenithDataset.write(&zenith[iH5], H5::PredType::NATIVE_FLOAT);
            zenithDataset.close();

            H5::DataSet nuTypeDataset = eventGroup.createDataSet("nuType", H5::PredType::NATIVE_INT, scalarDataspace);
            nuTypeDataset.write(&nuType[iH5], H5::PredType::NATIVE_INT);
            nuTypeDataset.close();

            // Save pmt-level data
            hsize_t dimsfeatureSpace[2] = {featureVector[iH5].size(), featureVector[iH5][0].size()};
            H5::DataSpace featureSpace(2, dimsfeatureSpace);
            std::vector<float> flatfeatureVector = flatten(featureVector[iH5]);

            H5::DataSet featureVectorDataset = eventGroup.createDataSet("featureVector", H5::PredType::NATIVE_FLOAT, featureSpace);
            featureVectorDataset.write(flatfeatureVector.data(), H5::PredType::NATIVE_FLOAT);
            featureVectorDataset.close();
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

std::string nuTypeString(int pdg) {
    if(pdg == 12) return "NuE";
    else if(pdg == -12) return "NuEBar";
    else if(pdg == 14) return "NuMu";
    else if(pdg == -14) return "NuMuBar";
    else return "Unknown";
}

int main() {
    EventData eventData;

    std::vector<std::string> detsimFiles;
    std::vector<std::string> calibFiles;

    std::ifstream detsimList("data/fileLists/detsim_nu_e.txt");
    std::ifstream calibList("data/fileLists/rec_nu_e.txt");
    if(!detsimList.is_open()){
        std::cerr << "Error opening file!";
        return 1;
    }

    std::string filepathDetsim, filepathCalib;

    for(int itPath = 0; itPath < 100; itPath++){
        if(std::getline(detsimList, filepathDetsim)){
            detsimFiles.push_back(filepathDetsim);
        }

        if(std::getline(calibList, filepathCalib)){
            calibFiles.push_back(filepathCalib);
        }
    }

    std::cout << "No files: " << detsimFiles.size() << std::endl;

    for(int iFile = 0; iFile < detsimFiles.size(); iFile++){
        std::string eosPath = "root://junoeos01.ihep.ac.cn/";
        std::string iterCalib = eosPath + calibFiles.at(iFile);
        std::string iterDetsim = eosPath + detsimFiles.at(iFile);

        std::cout << "calibeos path: " << iterCalib << std::endl;

        eventData.loadCalibEDM(iterCalib);
        eventData.loadDetSimEDM(iterDetsim);

        eventData.evtList.clear();
    }
    

    for(int i = 0; i < eventData.featureVector.size(); i++){
        std::cout << "Event " << i << " length: " << eventData.featureVector[i].size() << std::endl;
        std::cout << "Energy: " << eventData.energy[i] << std::endl;
        std::cout << "ParticleID: " << eventData.nuType[i] << std::endl; 
    }

    std::string filename = "/junofs/users/ljones/lem_juno/data/libAtmos.h5";
    eventData.h5Save(filename);

    return 0;
}