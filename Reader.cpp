#include <iostream>
#include <fstream>
#include <sstream>
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

const int MAX_TIMESTEPS = 1000;

std::vector<float> flattenEdepPos(const std::vector<std::vector<float>>& eventEdepPos) {
    std::vector<float> flatEdepPos;
    for (const auto& pos : eventEdepPos) {
        flatEdepPos.insert(flatEdepPos.end(), pos.begin(), pos.end());
    }
    return flatEdepPos;
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
    std::vector<std::vector<std::vector<short unsigned int>>> adcValues;
    std::vector<std::vector<int>> pmtID;
    std::vector<std::vector<float>> Phi;
    std::vector<std::vector<float>> Theta;
    std::vector<std::vector<float>> maxCharge;
    std::vector<std::vector<float>> maxTime;
    std::vector<std::vector<float>> sumCharge;
    std::vector<std::vector<float>> FHT;
    std::vector<std::vector<int>> PDGid;
    std::vector<std::vector<float>> initTrack;
    std::vector<std::vector<float>> exitTrack;
    std::vector<std::vector<float>> Edep;
    std::vector<std::vector<std::vector<float>>> EdepPos;
    std::vector<std::vector<float>> vertex;
    std::vector<float> energy;
    std::vector<int> nuType;
    int MaxEvents;

    void loadElecEDM(const std::string& filename) {
        TFile *elecEDM = TFile::Open(filename.c_str());
        if (!elecEDM || elecEDM->IsZombie()) {
            std::cerr << "Error opening file" << std::endl;
            return;
        }

        TTree *CdWf = (TTree*)elecEDM->Get("Event/CdWaveform/CdWaveformEvt");
        if (!CdWf) {
            std::cerr << "Error accessing TTree in file" << std::endl;
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
            std::cerr << "Error opening file" << std::endl;
            return;
        }

        TTree *genevt = (TTree*)detEDM->Get("Event/Gen/GenEvt");
        if (!genevt) {
            std::cerr << "Error accessing TTree in file" << std::endl;
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

        MaxEvents = genevt->GetEntries();

        for(int i = 0; i < MaxEvents; i++){
            genevt->GetEntry(i);
            simevt->GetEntry(i);

            std::cout << "Event: " << i << std::endl;

            auto hepmc_genevt = ge->getEvent();

            const std::vector<JM::SimTrack*>& tracks = se->getTracksVec();

            std::vector<float> SimEdep;
            std::vector<std::vector<float>> SimEdepPos;

            for(auto it = tracks.begin(); it != tracks.end(); ++it){
                int simPDGid = (*it)->getPDGID();
                float E = (*it)->getEdep();

                std::vector<float> Edepvtx;

                if(E < 0.5){
                    continue;
                }

                SimEdep.push_back(E);

                float EdepX = (*it)->getEdepX();
                float EdepY = (*it)->getEdepY();
                float EdepZ = (*it)->getEdepZ();

                Edepvtx.push_back(EdepX);
                Edepvtx.push_back(EdepY);
                Edepvtx.push_back(EdepZ);

                SimEdepPos.push_back(Edepvtx);
                Edepvtx.clear();
            }

            Edep.push_back(SimEdep);
            EdepPos.push_back(SimEdepPos);
            SimEdep.clear();
            SimEdepPos.clear();

            std::vector<float> vtx;

            for (auto iVtx = hepmc_genevt->vertices_begin();
                 iVtx != hepmc_genevt->vertices_end();
                 ++iVtx) {

                const auto& vertex_pos = (*iVtx)->position();
                
                vtx.push_back(vertex_pos.x());
                vtx.push_back(vertex_pos.y());
                vtx.push_back(vertex_pos.z());

                for (auto iPart = (*iVtx)->particles_in_const_begin(); 
                     iPart != (*iVtx)->particles_in_const_end();
                     ++iPart) {
                    const auto& part_momentum = (*iPart)->momentum();

                    int ParticleType = (*iPart)->pdg_id();

                    if(ParticleType == 12 || ParticleType == -12 || ParticleType == 14 || ParticleType == -14){
                        energy.push_back(part_momentum.e());
                        nuType.push_back(ParticleType);
                    }
                }
            }

            vertex.push_back(vtx);
        }

        detEDM->Close();
    }

    void loadCalibEDM(const std::string& filename) {
        TFile *calibEDM = TFile::Open(filename.c_str());
        if (!calibEDM || calibEDM->IsZombie()) {
            std::cerr << "Error opening file" << std::endl;
            return;
        }

        TTree *calib = (TTree*)calibEDM->Get("Event/CdLpmtCalib/CdLpmtCalibEvt");
        if (!calib) {
            std::cerr << "Error accessing TTree in file" << std::endl;
            return;
        }

        JM::CdLpmtCalibEvt *cal = new JM::CdLpmtCalibEvt();
        calib->SetBranchAddress("CdLpmtCalibEvt", &cal);
        calib->GetBranch("CdLpmtCalibEvt")->SetAutoDelete(true);

        int calNum = calib->GetEntries();

        for(int i = 0; i < calNum; i++){
            calib->GetEntry(i);

            const auto &calibCh = cal->calibPMTCol();

            int calibSize = calibCh.size();
            
            std::vector<int> calibChID;
            calibChID.reserve(calibSize);
            // std::vector<std::vector<float>> calibFeatures;
            // calibFeatures.reserve(calibSize);

            std::vector<float> calibPhi;
            std::vector<float> calibTheta;
            std::vector<float> calibmaxCharge;
            std::vector<float> calibmaxTime;
            std::vector<float> calibsumCharge;
            std::vector<float> calibFHT;
            calibmaxCharge.reserve(calibSize);
            calibmaxTime.reserve(calibSize);
            calibsumCharge.reserve(calibSize);
            calibFHT.reserve(calibSize);

            for(auto it = calibCh.begin(); it != calibCh.end(); ++it){
                int CalibId = CdID::module(Identifier((*it)->pmtId()));
                
                if(CalibId < 0){
                    std::cerr << "Error! Invalid CalibID" << std::endl;
                    std::cerr << "CalibID: " << CalibId << std::endl;
                }

                calibPhi.push_back(JUNOPMTLocation::get_instance().GetPMTPhi(CalibId));
                calibTheta.push_back(JUNOPMTLocation::get_instance().GetPMTTheta(CalibId));


                float maxChargeIt   = (*it)->maxCharge();
                float maxTimeIt     = (*it)->time()[(*it)->maxChargeIndex()];
                float sumChargeIt   = (*it)->sumCharge();
                float FHTIt         = (*it)->firstHitTime();
                
                // features.push_back(maxChargeIt);
                // features.push_back(maxTimeIt);
                // features.push_back(sumChargeIt);
                // features.push_back(FHTIt);

                calibmaxCharge.push_back(maxChargeIt);
                calibmaxTime.push_back(maxTimeIt);
                calibsumCharge.push_back(sumChargeIt);
                calibFHT.push_back(FHTIt);

                // calibFeatures.push_back(features);
                calibChID.push_back(CalibId);
            }

            Phi.push_back(calibPhi);
            Theta.push_back(calibTheta);
            maxCharge.push_back(calibmaxCharge);
            maxTime.push_back(calibmaxTime);
            sumCharge.push_back(calibsumCharge);
            FHT.push_back(calibFHT);

            calibPhi.clear();
            calibTheta.clear();
            calibmaxCharge.clear();
            calibmaxTime.clear();
            calibsumCharge.clear();
            calibFHT.clear();
            
            pmtID.push_back(calibChID);
            calibChID.clear();
        }

        calibEDM->Close();
    }

    void h5Test(const std::string& filename) {
        try {
            H5::H5File file(filename, H5F_ACC_TRUNC);

            for (int i = 0; i < MaxEvents; ++i) {
                // Create a group for each event
                std::string eventPath = "/Event_" + std::to_string(i);
                H5::Group eventGroup(file.createGroup(eventPath));

                // Define the shape of the data
                H5::DataSpace scalarSpace(H5S_SCALAR);
                H5::DataSpace vectorSpace(1, new hsize_t[1]{3});

                hsize_t dimsFeature[1] = {maxCharge[i].size()};
                H5::DataSpace FeatureSpace(1, dimsFeature);

                hsize_t dimsEdep[1] = {Edep[i].size()};
                H5::DataSpace EdepSpace(1, dimsEdep);
                hsize_t dimsEdepPos[2] = {EdepPos[i].size(), 3};
                H5::DataSpace EdepPosSpace(2, dimsEdepPos);

                // Write the data
                H5::DataSet energyDataset = eventGroup.createDataSet("energy", H5::PredType::NATIVE_FLOAT, scalarSpace);
                energyDataset.write(&energy[i], H5::PredType::NATIVE_FLOAT);
                energyDataset.close();

                H5::DataSet nuTypeDataset = eventGroup.createDataSet("nuType", H5::PredType::NATIVE_INT, scalarSpace);
                nuTypeDataset.write(&nuType[i], H5::PredType::NATIVE_INT);
                nuTypeDataset.close();

                H5::DataSet vertexDataset = eventGroup.createDataSet("vertex", H5::PredType::NATIVE_FLOAT, vectorSpace);
                vertexDataset.write(vertex[i].data(), H5::PredType::NATIVE_FLOAT);
                vertexDataset.close();

                H5::DataSet EdepDataset = eventGroup.createDataSet("Edep", H5::PredType::NATIVE_FLOAT, EdepSpace);
                EdepDataset.write(Edep[i].data(), H5::PredType::NATIVE_FLOAT);
                EdepDataset.close();

                std::vector<float> flatEdepPos = flattenEdepPos(EdepPos[i]);
                H5::DataSet EdepPosDataset = eventGroup.createDataSet("EdepPos", H5::PredType::NATIVE_FLOAT, EdepPosSpace);
                EdepPosDataset.write(flatEdepPos.data(), H5::PredType::NATIVE_FLOAT);
                EdepPosDataset.close();            

                std::string pmtGroupPath = eventPath + "/PMT";
                H5::Group pmtGroup(eventGroup.createGroup(pmtGroupPath));

                H5::DataSet pmtIDDataset = pmtGroup.createDataSet("PMTID", H5::PredType::NATIVE_INT, FeatureSpace);
                pmtIDDataset.write(pmtID[i].data(), H5::PredType::NATIVE_INT);
                pmtIDDataset.close();

                H5::DataSet PhiDataset = pmtGroup.createDataSet("Phi", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                PhiDataset.write(Phi[i].data(), H5::PredType::NATIVE_FLOAT);
                PhiDataset.close();

                H5::DataSet ThetaDataset = pmtGroup.createDataSet("Theta", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                ThetaDataset.write(Theta[i].data(), H5::PredType::NATIVE_FLOAT);
                ThetaDataset.close();

                H5::DataSet maxChargeDataset = pmtGroup.createDataSet("maxCharge", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                maxChargeDataset.write(maxCharge[i].data(), H5::PredType::NATIVE_FLOAT);
                maxChargeDataset.close();

                H5::DataSet maxTimeDataset = pmtGroup.createDataSet("maxTime", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                maxTimeDataset.write(maxTime[i].data(), H5::PredType::NATIVE_FLOAT);
                maxTimeDataset.close();

                H5::DataSet sumChargeDataset = pmtGroup.createDataSet("sumCharge", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                sumChargeDataset.write(sumCharge[i].data(), H5::PredType::NATIVE_FLOAT);
                sumChargeDataset.close();

                H5::DataSet FHTDataset = pmtGroup.createDataSet("FHT", H5::PredType::NATIVE_FLOAT, FeatureSpace);
                FHTDataset.write(FHT[i].data(), H5::PredType::NATIVE_FLOAT);
                FHTDataset.close();
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
    eventData.loadDetSimEDM("data/detsimAtmos.root");
    eventData.loadCalibEDM("data/calibAtmos.root");

    std::string filename = "data/test.h5";
    eventData.h5Test(filename);

    return 0;
}