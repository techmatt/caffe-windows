
#include "main.h"

const string baseDir = R"(C:\Code\caffe\caffe-windows\matt\data\)";

void makeSimulationVideo()
{
    cout << "Making simulation video..." << endl;

    SimulationHistories histories = ParticleSystem::makeSimulations(1000);

    Grid2<vec3f> meanValues = histories.histories[0].history[0];
    meanValues.setValues(vec3f(0.0f, 0.0f, 0.0f));

    histories.saveVideoFrames(baseDir + "videos/goldSimulations/", meanValues);
}

void generateSimulationDatabase()
{
    cout << "Geneating simulation database..." << endl;

    ParticleSystem::makeDatabase(baseDir + "simulation-train-raw\\", 50000);
    ParticleSystem::makeDatabase(baseDir + "simulation-test-raw\\", 500);
}

void compileDatabase(const string &databaseName)
{
    cout << "Compiling database..." << endl;

    ImageDatabase databaseA;
    databaseA.loadTargeted(baseDir + databaseName + "-train-raw\\");
    databaseA.save(baseDir + databaseName + "-train-leveldb");

    ImageDatabase databaseB;
    databaseB.loadTargeted(baseDir + databaseName + "-test-raw\\");
    databaseB.save(baseDir + databaseName + "-test-leveldb");
}

void main()
{
    //const string databaseName = "simulation";
    const string databaseName = "clouds";

    //helper::processVideoFolder(R"(C:\Code\caffe\caffe-windows\matt\data\cloudsRaw\)", R"(C:\Code\caffe\caffe-windows\matt\data\cloudsProcessed\)", bbox2i(vec2i(3, 63), vec2i(476, 295)), 1);
    //helper::videoToSamples(R"(C:\Code\caffe\caffe-windows\matt\data\cloudsProcessed\)", R"(C:\Code\caffe\caffe-windows\matt\data\clouds-train-raw\)", 82, 50000);
    //helper::videoToSamples(R"(C:\Code\caffe\caffe-windows\matt\data\cloudsProcessed\)", R"(C:\Code\caffe\caffe-windows\matt\data\clouds-test-raw\)", 82, 500);
    //helper::videoToDatabase(baseDir + "cloudsProcessed/", baseDir + "clouds-train-leveldb", vec2i(82, 82), 20000);
    //helper::videoToDatabase(baseDir + "cloudsProcessed/", baseDir + "clouds-test-leveldb", vec2i(82, 82), 500);
    ParticleSystem::makeDatabase(baseDir + "particles-train-leveldb", 20000);
    ParticleSystem::makeDatabase(baseDir + "particles-test-leveldb", 500);

    
    //makeSimulationVideo();

    //generateSimulationDatabase();

    //compileDatabase(databaseName);
}
