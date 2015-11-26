
#include "main.h"

void SimulationState::init(const Netf &_net)
{
    net = _net;
    
    std::vector<Blob<float>*> inputBlobs;
    net->Forward(inputBlobs);

    const Blobf dataBlob = net->blob_by_name("data");

    history.clear();
    for (int historyIndex = 0; historyIndex < 4; historyIndex++)
    {
        history.push_back(helper::blobToFloatGrid(dataBlob, 0, 3 - historyIndex));
    }

    const Blobf outputBlob = net->blob_by_name("deconv3");
    debugPrediction = helper::blobToFloatGrid(outputBlob, 0, 0);
}

void SimulationState::step()
{
    Blobf dataBlob = net->blob_by_name("data");
    for (int channel = 0; channel < 4; channel++)
    {
        helper::loadFloatGridIntoBlob(history[history.size() - 1 - channel], dataBlob, 0, channel);
    }
    
    helper::runNetForward(net, "data");

    const Blobf outputBlob = net->blob_by_name("deconv3");
    history.push_back(helper::blobToFloatGrid(outputBlob, 0, 0));
}

void SimulationState::save(const string &directory)
{
    auto debugImage = helper::gridToImage(debugPrediction);
    LodePNG::save(debugImage, directory + "debug.png");

    for (int historyIndex = 0; historyIndex < history.size(); historyIndex++)
    {
        auto image = helper::gridToImage(history[historyIndex]);
        LodePNG::save(image, directory + to_string(historyIndex) + "_predicted.png");
    }
}
