
#include "main.h"

const string baseDir = R"(C:\Code\caffe\caffe-windows\matt\data\)";

void makeSimulationVideo()
{
    cout << "Making simulation video..." << endl;

    SimulationHistories histories = ParticleSystem::makeSimulations(1000);
    histories.saveVideoFrames(baseDir + "videos/goldSimulations/");
}

void generateSimulationDatabase()
{
    cout << "Geneating simulation database..." << endl;

    ParticleSystem::makeDatabase(baseDir + "simulation-train-raw\\", 50000);
    ParticleSystem::makeDatabase(baseDir + "simulation-test-raw\\", 500);
}

void compileSimulationDatabase()
{
    cout << "Compiling database..." << endl;

    const string databaseName = "simulation";

    ImageDatabase databaseA;
    databaseA.loadTargeted(baseDir + databaseName + "-train-raw\\");
    databaseA.save(baseDir + databaseName + "-train-leveldb");

    ImageDatabase databaseB;
    databaseB.loadTargeted(baseDir + databaseName + "-test-raw\\");
    databaseB.save(baseDir + databaseName + "-test-leveldb");
}

void main()
{
    makeSimulationVideo();

    //generateSimulationDatabase();
    //compileSimulationDatabase();
}
