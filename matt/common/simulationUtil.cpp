
#include "main.h"

void SimulationHistories::saveVideoFrames(const string &directory) const
{
    util::makeDirectory(directory);
    for (int frameIndex = 0; frameIndex < histories[0].history.size(); frameIndex++)
    {
        auto image = makeVideoFrame(frameIndex);
        LodePNG::save(image, directory + to_string(frameIndex) + ".png");
    }
}

ColorImageR8G8B8A8 SimulationHistories::makeVideoFrame(int frameIndex) const
{
    const int padding = 4;
    const int upscaleFactor = 2;

    const int imageSize = histories[0].history[0].getDimX();
    const int cellSize = padding * 2 + imageSize * upscaleFactor;

    ColorImageR8G8B8A8 result(cellSize * videoGridDims);
    result.setPixels(vec4uc(128, 128, 128, 255));

    int historyIndex = 0;
    for (int cellX = 0; cellX < videoGridDims.x; cellX++)
    {
        for (int cellY = 0; cellY < videoGridDims.y; cellY++)
        {
            ColorImageR8G8B8A8 historyImage = helper::gridToImage(histories[historyIndex++].history[frameIndex]);
            historyImage = helper::upsample(historyImage);
            
            for (auto &p : historyImage)
            {
                vec2i pixel = vec2i(cellX, cellY) * cellSize + vec2i(padding, padding) + vec2i(p.x, p.y);
                result(pixel.x, pixel.y) = p.value;
            }
        }
    }
    return result;
}