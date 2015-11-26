
typedef caffe::shared_ptr< Blob<float> > Blobf;
typedef caffe::shared_ptr< Net<float> > Netf;

struct BlobInfo
{
    BlobInfo() {}
    BlobInfo(string _name, string _suffix, int _channels) {
        name = _name;
        suffix = _suffix;
        channelsToOutput = _channels;
    }
    string name;
    string suffix;
    int channelsToOutput;

    Blobf data;
};

struct helper
{
    static ColorImageR8G8B8A8 blobToImage(const Blobf &blob, int imageIndex, int channelCount)
    {
        ColorImageR8G8B8A8 image(blob->width(), blob->height());
        for (auto &p : image)
        {
            for (int channel = 0; channel < channelCount; channel++)
            {
                const float *dataStart = blob->cpu_data() + blob->offset(imageIndex, channel, p.x, p.y);
                const BYTE value = util::boundToByte(*dataStart * 255.0f);
                p.value[channel] = value;
            }
            if (channelCount == 1)
                p.value[2] = p.value[1] = p.value[0];
            p.value.a = 255;
        }
        return image;
    }

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

    static Grid2<float> blobToFloatGrid(const Blobf &blob, int imageIndex, int channelIndex)
    {
        Grid2<float> grid(blob->width(), blob->height());
        for (auto &p : grid)
        {
            const float *dataStart = blob->cpu_data() + blob->offset(imageIndex, channelIndex, p.x, p.y);
            p.value = *dataStart;
        }
        return grid;
    }

    static void loadFloatGridIntoBlob(const Grid2<float> &grid, Blobf &blob, int imageIndex, int channelIndex)
    {
        float *cpuPtr = (float*)blob->data()->mutable_cpu_data();
        for (auto &p : grid)
        {
            float *dataStart = cpuPtr + blob->offset(imageIndex, channelIndex, p.x, p.y);
            *dataStart = p.value;
        }
    }

    static void runNetForward(const Netf &net, const string &inputLayerName)
    {
        int inputLayerIndex = -1;
        for (int layerIndex = 0; layerIndex < net->layers().size(); layerIndex++)
        {
            if (net->layer_names()[layerIndex] == inputLayerName)
                inputLayerIndex = layerIndex;
        }
        if (inputLayerIndex == -1)
            LOG(ERROR) << "Input layer not found: " << inputLayerName;

        net->ForwardFrom(inputLayerIndex + 1);
    }
};