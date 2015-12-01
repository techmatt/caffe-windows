
#include "main.h"

const string mattDir = R"(C:\Code\caffe\caffe-windows\matt\)";
const string baseDir = mattDir + "data/";

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

void main()
{
    RepresentationDataset::initFolder("fire");
    //RepresentationDataset::makeVideoDataset("fire", baseDir + "videos/fireBProcessed/", vec2i(82, 82), 2000);

    //helper::processVideoFolder(R"(C:\Code\caffe\caffe-windows\matt\data\cloudsRaw\)", R"(C:\Code\caffe\caffe-windows\matt\data\cloudsProcessed\)", bbox2i(vec2i(3, 63), vec2i(476, 295)), 1);
    //helper::processVideoFolder(baseDir + "videos/fireBRaw/", baseDir + "videos/fireBProcessed/", bbox2i(vec2i(100, 1), vec2i(1278, 392)), 1);
    //helper::processVideoFolder(baseDir + "videos/fireARaw/", baseDir + "videos/fireAProcessed/", bbox2i(vec2i(100, 1), vec2i(1278, 392)), 1);

    //helper::videoToDatabase(baseDir + "cloudsProcessed/", baseDir + "clouds-train-leveldb", vec2i(82, 82), 20000);
    //helper::videoToDatabase(baseDir + "cloudsProcessed/", baseDir + "clouds-test-leveldb", vec2i(82, 82), 500);

    //helper::videoToDatabase(baseDir + "videos/fireBProcessed/", baseDir + "fireB-train-leveldb", vec2i(82, 82), 20000);
    //helper::videoToDatabase(baseDir + "videos/fireBProcessed/", baseDir + "fireB-test-leveldb", vec2i(82, 82), 500);

    //helper::videoToDatabase(baseDir + "videos/fireAProcessed/", baseDir + "fireA-train-leveldb", vec2i(82, 82), 20000);
    //helper::videoToDatabase(baseDir + "videos/fireAProcessed/", baseDir + "fireA-test-leveldb", vec2i(82, 82), 500);

    //helper::videoToDatabase(baseDir + "videos/fireBProcessed/", baseDir + "fireBBig-train-leveldb", vec2i(164, 164), 20000);
    //helper::videoToDatabase(baseDir + "videos/fireBProcessed/", baseDir + "fireBBig-test-leveldb", vec2i(164, 164), 500);

    //ParticleSystem::makeDatabase(baseDir + "particles-train-leveldb", 20000);
    //ParticleSystem::makeDatabase(baseDir + "particles-test-leveldb", 500);

    
    //makeSimulationVideo();

    //generateSimulationDatabase();
}
