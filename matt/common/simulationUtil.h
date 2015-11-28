
struct SimulationHistory
{
    vector< Grid2<vec3f> > history;
};

struct SimulationHistories
{
    void saveVideoFrames(const string &directory, const Grid2<vec3f> &meanValues) const;
    ColorImageR8G8B8A8 makeVideoFrame(int frameIndex, const Grid2<vec3f> &meanValues) const;

    vector< SimulationHistory > histories;
    vec2i videoGridDims;
};

namespace helper
{
    inline ColorImageR8G8B8A8 gridToImage(const Grid2<vec3f> &grid, const Grid2<vec3f> &meanValues)
    {
        ColorImageR8G8B8A8 image(grid.getDimX(), grid.getDimY());
        for (auto &p : image)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                const BYTE value = util::boundToByte((grid(p.x, p.y)[channel] + meanValues(p.x, p.y)[channel] / 255.0f) * 255.0f);
                p.value[channel] = value;
            }
            p.value.a = 255;
        }
        return image;
    }

    inline ColorImageR8G8B8A8 upsample(const ColorImageR8G8B8A8 &input)
    {
        ColorImageR8G8B8A8 result(input.getDimensions() * 2);
        input.reSample(result.getWidth(), result.getHeight(), result);
        return result;
    }

    void processVideoFolder(const string &inputDir, const string &outputDir, const bbox2i &rect, int mipmapCount);
    void videoToSamples(const string &inputDir, const string &outputDir, int imageSize, int sampleCount);
    void videoToDatabase(const string &inputVideoDir, const string &outputDatabaseDir, vec2i imageDimensions, int sampleCount);
}