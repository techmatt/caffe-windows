
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

struct CaffeUtil
{
    static ColorImageR8G8B8A8 blobToImage(const Blobf &blob, int imageIndex, int channelStartIndex, const Grid2<vec3f> &meanValues)
    {
        ColorImageR8G8B8A8 image(blob->width(), blob->height());
        for (auto &p : image)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                const float *dataStart = blob->cpu_data() + blob->offset(imageIndex, channelStartIndex + channel, p.x, p.y);
                const BYTE value = util::boundToByte(*dataStart * 255.0f + meanValues(p.x, p.y)[channel]);
                p.value[channel] = value;
            }
            p.value.a = 255;
        }
        return image;
    }

    static Grid2<vec3f> gridVec3FromBinaryProto(const string &filename)
    {
        BlobProto blobProto;
        caffe::ReadProtoFromBinaryFileOrDie(filename.c_str(), &blobProto);

        Blob<float> blob;
        blob.FromProto(blobProto);

        LOG(ERROR) << "binary proto channels: " << blob.channels();
        return blobToGridVec3(blob, 0, 0);
    }

    static Grid2<vec3f> blobToGridVec3(const Blob<float> &blob, int imageIndex, int channelIndexStart)
    {
        Grid2<float> g0 = blobToGridFloat(blob, imageIndex, channelIndexStart + 0);
        Grid2<float> g1 = blobToGridFloat(blob, imageIndex, channelIndexStart + 1);
        Grid2<float> g2 = blobToGridFloat(blob, imageIndex, channelIndexStart + 2);

        Grid2<vec3f> result(g0.getDimensions());
        for (auto &v : result)
        {
            v.value = vec3f(
                g0(v.x, v.y),
                g1(v.x, v.y),
                g2(v.x, v.y));
        }
        return result;
    }

    static Grid2<float> blobToGridFloat(const Blob<float> &blob, int imageIndex, int channelIndex)
    {
        Grid2<float> grid(blob.width(), blob.height());
        for (auto &p : grid)
        {
            const float *dataStart = blob.cpu_data() + blob.offset(imageIndex, channelIndex, p.x, p.y);
            p.value = *dataStart;
        }
        return grid;
    }

    static void loadGridFloatIntoBlob(const Grid2<float> &grid, Blobf &blob, int imageIndex, int channelIndex)
    {
        float *cpuPtr = (float*)blob->data()->mutable_cpu_data();
        for (auto &p : grid)
        {
            float *dataStart = cpuPtr + blob->offset(imageIndex, channelIndex, p.x, p.y);
            *dataStart = p.value;
        }
    }

    static void loadGridVec3IntoBlob(const Grid2<vec3f> &grid, Blobf &blob, int imageIndex, int channelStartIndex)
    {
        float *cpuPtr = (float*)blob->data()->mutable_cpu_data();
        for (auto &p : grid)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                float *dataStart = cpuPtr + blob->offset(imageIndex, channelStartIndex + channel, p.x, p.y);
                *dataStart = p.value[channel];
            }
        }
    }

    static Grid3<float> blobToGrid3(const Blob<float> &blob, int imageIndex)
    {
        Grid3<float> result;
        result.allocate(blob.width(), blob.height(), blob.channels());

        for (int z = 0; z < result.getDimZ(); z++)
            for (int y = 0; y < result.getDimY(); y++)
                for (int x = 0; x < result.getDimX(); x++)
                {
                    const float *dataStart = blob.cpu_data() + blob.offset(imageIndex, z, x, y);
                    result(x, y, z) = *dataStart;
                }
        return result;
    }

    static void loadGrid3IntoBlob(const Grid3<float> &grid, Blobf &blob, int imageIndex)
    {
        float *cpuPtr = (float*)blob->data()->mutable_cpu_data();
        for (int z = 0; z < grid.getDimZ(); z++)
            for (int y = 0; y < grid.getDimY(); y++)
                for (int x = 0; x < grid.getDimX(); x++)
                {
                    float *dataStart = cpuPtr + blob->offset(imageIndex, z, x, y);
                    *dataStart = grid(x, y, z);
                }
    }

    static int getLayerIndex(const Netf &net, const string &layerName)
    {
        int result = -1;
        for (int layerIndex = 0; layerIndex < net->layers().size(); layerIndex++)
        {
            if (net->layer_names()[layerIndex] == layerName)
                result = layerIndex;
        }
        if (result == -1)
            LOG(ERROR) << "Input layer not found: " << layerName;
        return result;
    }

    static void runNetForward(const Netf &net, const string &inputLayerName)
    {
        const int inputLayerIndex = getLayerIndex(net, inputLayerName);
        net->ForwardFrom(inputLayerIndex + 1);
    }

    static void runNetForward(const Netf &net, const string &inputLayerName, const Grid3<float> &inputLayerData)
    {
        const int inputLayerIndex = getLayerIndex(net, inputLayerName);

        Blobf inputBlob = net->blob_by_name(inputLayerName);
        loadGrid3IntoBlob(inputLayerData, inputBlob, 0);
        
        net->ForwardFrom(inputLayerIndex + 1);
    }

    static Grid3<float> getBlobAsGrid(const Netf &net, const string &blobName)
    {
        if (!net->has_blob(blobName))
            LOG(ERROR) << "blob not found: " << blobName;

        auto blob = net->blob_by_name(blobName);

        return blobToGrid3(*blob.get(), 0);
    }
};