
#include "main.h"

void SimulationState::init(const Netf &_net)
{
    net = _net;
    
    std::vector<Blob<float>*> inputBlobs;
    net->Forward(inputBlobs);

    const Blobf dataBlob = net->blob_by_name("data");

    history.history.clear();
    for (int historyIndex = 0; historyIndex < 4; historyIndex++)
    {
        history.history.push_back(CaffeUtil::blobToFloatGrid(dataBlob, 0, 3 - historyIndex));
    }

    const Blobf outputBlob = net->blob_by_name("deconv3");
    debugPrediction = CaffeUtil::blobToFloatGrid(outputBlob, 0, 0);
}

void SimulationState::step()
{
    Blobf dataBlob = net->blob_by_name("data");
    for (int channel = 0; channel < 4; channel++)
    {
        CaffeUtil::loadFloatGridIntoBlob(history.history[history.history.size() - 1 - channel], dataBlob, 0, channel);
    }
    
    CaffeUtil::runNetForward(net, "data");

    const Blobf outputBlob = net->blob_by_name("deconv3");
    history.history.push_back(CaffeUtil::blobToFloatGrid(outputBlob, 0, 0));
}

void SimulationState::save(const string &directory)
{
    auto debugImage = helper::gridToImage(debugPrediction);
    LodePNG::save(debugImage, directory + "debug.png");

    for (int historyIndex = 0; historyIndex < history.history.size(); historyIndex++)
    {
        auto image = helper::gridToImage(history.history[historyIndex]);
        LodePNG::save(image, directory + to_string(historyIndex) + "_predicted.png");
    }
}
