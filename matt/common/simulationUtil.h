
struct SimulationHistory
{
    vector< Grid2<float> > history;
};

struct SimulationHistories
{
    void saveVideoFrames(const string &directory) const;
    ColorImageR8G8B8A8 makeVideoFrame(int frameIndex) const;

    vector< SimulationHistory > histories;
    vec2i videoGridDims;
};

namespace helper
{
    static ColorImageR8G8B8A8 gridToImage(const Grid2<float> &grid)
    {
        ColorImageR8G8B8A8 image(grid.getDimX(), grid.getDimY());
        for (auto &p : image)
        {
            const BYTE value = util::boundToByte(grid(p.x, p.y) * 255.0f);
            p.value = vec4uc(value, value, value, 255);
        }
        return image;
    }

    static ColorImageR8G8B8A8 upsample(const ColorImageR8G8B8A8 &input)
    {
        ColorImageR8G8B8A8 result(input.getDimensions() * 2);
        input.reSample(result.getWidth(), result.getHeight(), result);
        return result;
    }
}