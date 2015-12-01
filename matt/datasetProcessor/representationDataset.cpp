
#include "main.h"

const string baseDir = R"(C:\Code\caffe\caffe-windows\matt\representationData\)";

void RepresentationDataset::copyFile(const string &sourceFilename, const string &targetFilename, const vector< pair<string, string> > &replacements)
{
    ofstream fileOut(targetFilename);
    for (const string &line : util::getFileLines(sourceFilename))
    {
        string newLine = line;
        for (auto &r : replacements)
        {
            newLine = util::replace(newLine, r.first, r.second);
        }
        fileOut << newLine << endl;
    }
}

void RepresentationDataset::makeVideoDataset(const string &databaseName, const string &videoDirectory, vec2i imageDimensions, int trainingSampleCount)
{
    const string databaseDir = baseDir + databaseName + "/";
    helper::videoToDatabase(videoDirectory, databaseDir + databaseName + "-imageset-train-leveldb", imageDimensions, trainingSampleCount);
    helper::videoToDatabase(videoDirectory, databaseDir + databaseName + "-imageset-test-leveldb", imageDimensions, 500);
}

void RepresentationDataset::initFolder(const string &databaseName)
{
    const string databaseDir = baseDir + databaseName + "/";
    util::makeDirectory(databaseDir);

    vector< pair<string, string> > replacements;
    replacements.push_back(make_pair(string("DATASETNAME"), databaseName));

    auto processFile = [&](const string &baseFilename)
    {
        copyFile(baseDir + baseFilename, databaseDir + util::replace(baseFilename, "base", databaseName), replacements);
    };

    processFile("base-autoencoder-net.prototxt");
    processFile("base-autoencoder-solver.prototxt");
    processFile("base-autoencoder-run.bat");

    processFile("base-predictor-net.prototxt");
    processFile("base-predictor-solver.prototxt");
    processFile("base-predictor-run.bat");

    processFile("base-simulationA-net.prototxt");
    processFile("base-simulationA-solver.prototxt");
    processFile("base-simulationA-run.bat");

    processFile("base-simulationB-net.prototxt");
    processFile("base-simulationB-solver.prototxt");
    processFile("base-simulationB-run.bat");

    processFile("base-compute-imageset-mean.bat");
}
