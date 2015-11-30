
#include "main.h"

const string &outputLayerName = "deconv3";

void SimulationState::init(const Netf &_net)
{
    net = _net;
    
    std::vector<Blob<float>*> inputBlobs;
    net->Forward(inputBlobs);

    const Blobf dataBlob = net->blob_by_name("data");

    history.history.clear();
    for (int historyIndex = 0; historyIndex < 4; historyIndex++)
    {
        history.history.push_back(CaffeUtil::blobToGridVec3(*dataBlob.get(), 0, 3 * historyIndex));
    }

    const Blobf outputBlob = net->blob_by_name(outputLayerName);
    debugPrediction = CaffeUtil::blobToGridVec3(*outputBlob.get(), 0, 0);
}

void SimulationState::step()
{
    Blobf dataBlob = net->blob_by_name("data");
    for (int historyIndex = 0; historyIndex < 4; historyIndex++)
    {
        // TODO: vertify the direction is correct...
        CaffeUtil::loadGridVec3IntoBlob(history.history[history.history.size() - 1 - historyIndex], dataBlob, 0, 9 - 3 * historyIndex);
    }
    
    CaffeUtil::runNetForward(net, "data");

    const Blobf outputBlob = net->blob_by_name(outputLayerName);
    history.history.push_back(CaffeUtil::blobToGridVec3(*outputBlob.get(), 0, 0));
}

void SimulationState::save(const string &directory, const Grid2<vec3f> &meanValues)
{
    auto debugImage = helper::gridToImage(debugPrediction, meanValues);
    LodePNG::save(debugImage, directory + "debug.png");

    for (int historyIndex = 0; historyIndex < history.history.size(); historyIndex++)
    {
        auto image = helper::gridToImage(history.history[historyIndex], meanValues);
        LodePNG::save(image, directory + to_string(historyIndex) + "_sim.png");
    }
}
