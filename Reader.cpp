#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <H5Cpp.h>
#include <TFile.h>
#include <TTree.h>
#include <TVector3.h>
#include <Event/CdWaveformHeader.h>
#include <Event/CdLpmtCalibHeader.h>
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

#include <iostream>
#include <cmath>

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
    std::vector<int> muType;

    // Calib data
    std::vector<std::vector<int>> pmtID;
    std::vector<std::vector<float>> Phi;
    std::vector<std::vector<float>> Theta;
    std::vector<std::vector<float>> maxCharge;
    std::vector<std::vector<float>> maxTime;
    std::vector<std::vector<float>> sumCharge;
    std::vector<std::vector<float>> FHT;
    std::vector<std::vector<int>> PDGid;
    std::vector<std::vector<std::vector<float>>> featureVector;
    
    int MaxEvents;

    // Add a new member variable to store tracks by event ID
    std::map<int, std::vector<std::vector<float>>> eventTracks;

    void loadElecEDM(const std::string& filename) {
        TFile *elecEDM = TFile::Open(filename.c_str());
        if (!elecEDM || elecEDM->IsZombie()) {
            std::cerr << "Error opening elec file" << std::endl;
            return;
        }

        TTree *CdWf = (TTree*)elecEDM->Get("Event/CdWaveform/CdWaveformEvt");
        if (!CdWf) {
            std::cerr << "Error accessing TTree in elec file" << std::endl;
            return;
        }

        JM::CdWaveformEvt *wf = new JM::CdWaveformEvt();
        CdWf->SetBranchAddress("CdWaveformEvt", &wf);

        MaxEvents = CdWf->GetEntries();

        for(int i = 0; i < MaxEvents; i++) {
            CdWf->GetEntry(i);

            std::vector<std::vector<short unsigned int>> adcValuesEvent;
            std::vector<int> pmt;
            
            const auto &feeChannels = wf->channelData();

            for (auto it = feeChannels.begin(); it != feeChannels.end(); ++it) {
                int detID = (it->first);
                int id = CdID::module(Identifier(detID));

                if (id < 0){
                    std::cerr << "Error! Invalid detID" << std::endl;
                    std::cerr << "DetID: " << detID << std::endl;
                    std::cerr << "id   : " << id << std::endl;
                }

                pmt.push_back(id);

                const auto &channel = *(it->second);
                const auto &adc_int = channel.adc();

                std::vector<short unsigned int> adc;

                for (size_t k = 0; k < adc_int.size(); k++) {
                    adc.push_back(adc_int[k]);
                }

                adcValuesEvent.push_back(adc);
            }

            pmtID.push_back(pmt);
            adcValues.push_back(adcValuesEvent);
        }
        
        delete wf;
        elecEDM->Close();
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

        int SimMax = simevt->GetEntries();

        std::cout << "SimMax: " << SimMax << std::endl;

        int muontrks = 0;

        for(int i = 0; i < SimMax; i++){
            // genevt->GetEntry(i);
            simevt->GetEntry(i);

            // auto hepmc_genevt = ge->getEvent();

            int evtid = se->getEventID();

            const std::vector<JM::SimTrack*>& tracks = se->getTracksVec();
            
            if(tracks.size() == 0){
                continue;
            }

            std::cout << "Event ID: " << evtid << std::endl;

            // std::cout << "Event ID: " << evtid << std::endl;

            std::vector<std::vector<float>> eventTrackData;

            for(auto it = tracks.begin(); it != tracks.end(); ++it){
                int simPDGid = (*it)->getPDGID();
                // float E = (*it)->getEdepCDWaterBuffer();

                if(simPDGid == 13 || simPDGid == -13){
                    muontrks++;
                    
                    std::vector<float> trackData;

                    // std::cout << "Init Position: " << (*it)->getInitX()/1000 << " " << (*it)->getInitY()/1000 << " " << (*it)->getInitZ() /1000 << std::endl;

                    Vec3 p0 = {(*it)->getInitX()/1000, (*it)->getInitY()/1000, (*it)->getInitZ()/1000};
                    Vec3 p1 = {(*it)->getExitX()/1000, (*it)->getExitY()/1000, (*it)->getExitZ()/1000};

                    if(LineIntersectsSphere(p0, p1, radius)){
                        std::cout << "Muon Track intersects sphere" << std::endl;
                    }

                    trackData.push_back((*it)->getInitX() / 1000);
                    trackData.push_back((*it)->getInitY() / 1000);
                    trackData.push_back((*it)->getInitZ() / 1000);
                    trackData.push_back((*it)->getExitX() / 1000);
                    trackData.push_back((*it)->getExitY() / 1000);
                    trackData.push_back((*it)->getExitZ() / 1000);
                    trackData.push_back((*it)->getInitPx());
                    trackData.push_back((*it)->getInitPy());
                    trackData.push_back((*it)->getInitPz());

                    eventTrackData.push_back(trackData);
                }
            }

            eventTracks[evtid] = eventTrackData;

            // std::cout << "Length of eventTrackData: " << eventTrackData.size() << std::endl;
        }

        std::cout << "Muon Tracks: " << muontrks << std::endl;

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

        JM::CdLpmtCalibEvt *cal = new JM::CdLpmtCalibEvt();
        calib->SetBranchAddress("CdLpmtCalibEvt", &cal);
        calib->GetBranch("CdLpmtCalibEvt")->SetAutoDelete(true);

        int calibMax = calib->GetEntries();

        std::cout << "CalibMax: " << calibMax << std::endl;

        MaxEvents = calibMax;

        for(int i = 0; i < calibMax; i++){
            calib->GetEntry(i);

            const auto &calibCh = cal->calibPMTCol();

            int calibSize = calibCh.size();
            
            std::vector<int> calibChID;
            calibChID.reserve(calibSize);
            std::vector<std::vector<float>> calibFeatures;
            calibFeatures.reserve(MAX_PMTID);

            // std::vector<float> calibPhi;
            // std::vector<float> calibTheta;
            // std::vector<float> calibmaxCharge;
            // std::vector<float> calibmaxTime;
            // std::vector<float> calibsumCharge;
            // std::vector<float> calibFHT;
            // calibmaxCharge.reserve(calibSize);
            // calibmaxTime.reserve(calibSize);
            // calibsumCharge.reserve(calibSize);
            // calibFHT.reserve(calibSize);

            for(auto it = calibCh.begin(); it != calibCh.end(); ++it){
                int CalibId = CdID::module(Identifier((*it)->pmtId()));
                
                if(CalibId < 0){
                    std::cerr << "Error! Invalid CalibID" << std::endl;
                    std::cerr << "CalibID: " << CalibId << std::endl;
                }

                float phiIt = JUNOPMTLocation::get_instance().GetPMTPhi(CalibId);
                float thetaIt = JUNOPMTLocation::get_instance().GetPMTTheta(CalibId);

                // calibPhi.push_back(phiIt);
                // calibTheta.push_back(thetaIt);


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
                

                // calibmaxCharge.push_back(maxChargeIt);
                // calibmaxTime.push_back(maxTimeIt);
                // calibsumCharge.push_back(sumChargeIt);
                // calibFHT.push_back(FHTIt);

                calibFeatures.push_back(features);
                features.clear();
                calibChID.push_back(CalibId);
            }

            for(int j = 0; j < MAX_PMTID; j++){
                if(std::find(calibChID.begin(), calibChID.end(), j) == calibChID.end()){
                    std::vector<float> features;

                    features.push_back(JUNOPMTLocation::get_instance().GetPMTPhi(j));
                    features.push_back(JUNOPMTLocation::get_instance().GetPMTTheta(j));
                    features.push_back(0);
                    features.push_back(0);
                    features.push_back(0);
                    features.push_back(0);

                    // calibPhi.push_back(JUNOPMTLocation::get_instance().GetPMTPhi(j));
                    // calibTheta.push_back(JUNOPMTLocation::get_instance().GetPMTTheta(j));
                    // calibmaxCharge.push_back(0);
                    // calibmaxTime.push_back(0);
                    // calibsumCharge.push_back(0);
                    // calibFHT.push_back(0);
                    // calibChID.push_back(j);

                    calibFeatures.push_back(features);
                    features.clear();
                }
            }

            std::cout << "Calib Event: " << i << std::endl;

            featureVector.push_back(calibFeatures);
            calibFeatures.clear();

            // Phi.push_back(calibPhi);
            // Theta.push_back(calibTheta);
            // maxCharge.push_back(calibmaxCharge);
            // maxTime.push_back(calibmaxTime);
            // sumCharge.push_back(calibsumCharge);
            // FHT.push_back(calibFHT);

            // calibPhi.clear();
            // calibTheta.clear();
            // calibmaxCharge.clear();
            // calibmaxTime.clear();
            // calibsumCharge.clear();
            // calibFHT.clear();
            
            // pmtID.push_back(calibChID);
            // calibChID.clear();
        }

        calibEDM->Close();
    }

    void h5Save(const std::string& filename) {
        try {
            H5::H5File file(filename, H5F_ACC_TRUNC);

            for (int i = 0; i < MaxEvents; ++i) {
                // Create a group for each event
                std::string eventPath = "/Event_" + std::to_string(i);
                H5::Group eventGroup(file.createGroup(eventPath));

                // Define the shape of the data
                H5::DataSpace scalarSpace(H5S_SCALAR);
                H5::DataSpace vectorSpace(1, new hsize_t[1]{3});

                // hsize_t dimsFeature[1] = {maxCharge[i].size()};
                // H5::DataSpace FeatureSpace(1, dimsFeature);

                // hsize_t dimsEdep[1] = {Edep[i].size()};
                // H5::DataSpace EdepSpace(1, dimsEdep);

                // hsize_t dimsEdepPos[2] = {EdepPos[i].size(), 3};
                // H5::DataSpace EdepPosSpace(2, dimsEdepPos);

                // if (eventTracks.find(i) != eventTracks.end()) {
                //     const auto& tracks = eventTracks[i];
                //     hsize_t muTrackDims[2] = {tracks.size(), 9}; // 9 elements per track
                //     H5::DataSpace muTrackSpace(2, muTrackDims);

                //     std::vector<float> flatMuTrack;
                //     for (const auto& track : tracks) {
                //         flatMuTrack.insert(flatMuTrack.end(), track.begin(), track.end());
                //     }

                //     H5::DataSet muTrackDataset = eventGroup.createDataSet("muTrack", H5::PredType::NATIVE_FLOAT, muTrackSpace);
                //     muTrackDataset.write(flatMuTrack.data(), H5::PredType::NATIVE_FLOAT);
                //     muTrackDataset.close();
                // }

                std::string pmtGroupPath = eventPath + "/PMT";
                H5::Group pmtGroup(eventGroup.createGroup(pmtGroupPath));
                
                if (i < featureVector.size()) {
                    hsize_t dimsPmtFeatureVector[2] = {featureVector[i].size(), 6};
                    H5::DataSpace PmtFeatureVectorSpace(2, dimsPmtFeatureVector);

                    std::vector<float> flatFeatureVector = flatten(featureVector[i]);
                    H5::DataSet pmtFeatureVectorDataset = pmtGroup.createDataSet("FeatureVector", H5::PredType::NATIVE_FLOAT, PmtFeatureVectorSpace);
                    pmtFeatureVectorDataset.write(flatFeatureVector.data(), H5::PredType::NATIVE_FLOAT);
                    pmtFeatureVectorDataset.close();
                }

                // H5::DataSet pmtIDDataset = pmtGroup.createDataSet("PMTID", H5::PredType::NATIVE_INT, FeatureSpace);
                // pmtIDDataset.write(pmtID[i].data(), H5::PredType::NATIVE_INT);
                // pmtIDDataset.close();

                // H5::DataSet PhiDataset = pmtGroup.createDataSet("Phi", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                // PhiDataset.write(Phi[i].data(), H5::PredType::NATIVE_FLOAT);
                // PhiDataset.close();

                // H5::DataSet ThetaDataset = pmtGroup.createDataSet("Theta", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                // ThetaDataset.write(Theta[i].data(), H5::PredType::NATIVE_FLOAT);
                // ThetaDataset.close();

                // H5::DataSet maxChargeDataset = pmtGroup.createDataSet("maxCharge", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                // maxChargeDataset.write(maxCharge[i].data(), H5::PredType::NATIVE_FLOAT);
                // maxChargeDataset.close();

                // H5::DataSet maxTimeDataset = pmtGroup.createDataSet("maxTime", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                // maxTimeDataset.write(maxTime[i].data(), H5::PredType::NATIVE_FLOAT);
                // maxTimeDataset.close();

                // H5::DataSet sumChargeDataset = pmtGroup.createDataSet("sumCharge", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                // sumChargeDataset.write(sumCharge[i].data(), H5::PredType::NATIVE_FLOAT);
                // sumChargeDataset.close();

                // H5::DataSet FHTDataset = pmtGroup.createDataSet("FHT", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                // FHTDataset.write(FHT[i].data(), H5::PredType::NATIVE_FLOAT);
                // FHTDataset.close();
            }

            file.close();

        } catch (H5::FileIException &error) {
            std::cerr << "File I/O error: " << error.getDetailMsg() << std::endl;
        } catch (H5::GroupIException &error) {
            std::cerr << "Group I/O error: " << error.getDetailMsg() << std::endl;
        } catch (H5::DataSetIException &error) {
            std::cerr << "Dataset I/O error: " << error.getDetailMsg() << std::endl;
        } catch (H5::DataSpaceIException &error) {
            std::cerr << "Dataspace I/O error: " << error.getDetailMsg() << std::endl;
        } catch (std::exception &error) {
            std::cerr << "Standard exception: " << error.what() << std::endl;
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

    std::string detsimfile = "data/Mu/detsimWP.root";
    std::string calibfile = "data/atmos/calibAtmos.root";

    // eventData.loadDetSimEDM(detsimfile);
    eventData.loadCalibEDM(calibfile);

    // Print the contents of eventTracks
    // for (const auto& event : eventData.eventTracks) {
    //     int eventID = event.first;
    //     const auto& tracks = event.second;

    //     std::cout << "Event ID: " << eventID << std::endl;
    //     for (const auto& track : tracks) {
    //         std::cout << "Track: ";
    //         for (const auto& value : track) {
    //             std::cout << value << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    std::string filename = "data/libAtmos.h5";
    eventData.h5Save(filename);

    return 0;
}