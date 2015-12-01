
class RepresentationProcessor
{
public:
    void makeProcessedDataset(const string &databaseName, const string &inputVideoDir, vec2i videoClipDimensions, const string &databaseSuffix, int sampleCount);

private:
    vector<ColorImageR8G8B8A8> getRandomVideoSequence(const string &inputVideoDir, int sequenceLength);

    vector<ColorImageR8G8B8A8> getRandomVideoClip(const vector<ColorImageR8G8B8A8> &sequence);
    vector< Grid2<vec3f> > applyTranformsVideoClip(const vector<ColorImageR8G8B8A8> &sequence);

    Grid3<float> extractFiveFrameWindow(const vector< Grid2<vec3f> > &transformedClip, int startFrame);

    Grid3<float> applyNetTransform(const Grid3<float> &extractedWindow, const Netf &net);

    vec2i clipDimensions;
    Grid2<vec3f> meanValues;

    vector<string> videoImagePaths;
};
