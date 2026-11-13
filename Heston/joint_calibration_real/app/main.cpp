#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include <filesystem>

#include "../src/smoothStateCalibration.cpp"
#include "../src/getRealData.cpp"

using namespace std;
namespace fs = std::filesystem;

static string valueAfterColon(const string& line) {
    size_t pos = line.find(':');
    if (pos == string::npos) return "";
    string value = line.substr(pos + 1);

    size_t start = value.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";

    size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

int main(){
    
    // get data from input file
    ifstream input("inp.txt");
    if (!input.is_open()) {
        cerr << "Failed to open inp.txt" << endl;
        return 1;
    }
    
    // these are the defauly values
    int numParticles = 200;
    int numVolSamples = 150;
    bool useESS = true;
    int numThreads = 1;
    double alpha = 0.05;
    string folderPath, outputPath;
    bool useLMS = false;
    bool is_daily_rate = true;
    
    string line;
    while (getline(input, line)) {
        if (line.find("number of particles:") == 0) {
            numParticles = stoi(valueAfterColon(line));
        } else if (line.find("number of volatility path samples:") == 0) {
            numVolSamples = stoi(valueAfterColon(line));
        } else if (line.find("use ESS (1 or 0):") == 0) {
            useESS = stoi(valueAfterColon(line)) != 0;
        } else if (line.find("number of threads:") == 0) {
            numThreads = stoi(valueAfterColon(line));
        } else if (line.find("alpha (enter a value between 0 and 1):") == 0) {
            alpha = stod(valueAfterColon(line));
        } else if (line.find("use LMS (1 or 0):") == 0) {
            useLMS = stoi(valueAfterColon(line)) != 0;
        } else if (line.find("is daily rate (1 for daily, 0 for monthly):") == 0) {
            is_daily_rate = stoi(valueAfterColon(line)) != 0;
        } else if (line.find("folder path:") == 0) {
            folderPath = valueAfterColon(line);
        } else if (line.find("Output path:") == 0) {
            outputPath = valueAfterColon(line);
        }
    }
    
    input.close();
    
    cout << "Particles: " << numParticles << endl;
    cout << "Vol samples: " << numVolSamples << endl;
    cout << "Use ESS: " << useESS << endl;
    cout << "Threads: " << numThreads << endl;
    cout << "Alpha: " << alpha << endl;
    cout << "Use LMS: " << useLMS << endl;
    cout << "is daily rate: " << is_daily_rate << endl;
    cout << "Folder path: " << folderPath << endl;
    cout << "Output path: " << outputPath << endl;

    // First we generate the data
    getRealData data(folderPath, is_daily_rate);
    cout<<"Data Generation Done."<<endl;
    
    // Then we need to do calibration
    // We pass data, number of particles, number of volatility path samples, use ESS, number of processes/threads, alpha
    smoothStateCalibration cali(data, numParticles, numVolSamples, useESS, numThreads, alpha, useLMS);

    // make the outut path if not present
    fs::create_directories(outputPath);
    fs::create_directories(outputPath+"/EAPE");
    fs::create_directories(outputPath+"/MAPE");

    cali.main(outputPath);
    return 0;
}