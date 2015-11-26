
#include "mLibInclude.h"

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <stdint.h>
#include <sys/stat.h>
#include <direct.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace google;
using std::string;

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

    caffe::shared_ptr< Blob<float> > data;
};

ColorImageR8G8B8A8 blobToImage(const caffe::shared_ptr< Blob<float> > &blob, int imageIndex, int channelCount)
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

void main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    //caffe::GlobalInit(&argc, &argv);
    const string baseDir = R"(C:\Code\caffe\caffe-windows\matt\data\)";

    const string outputDir = baseDir + "output/";
    util::makeDirectory(outputDir);
    
    const bool useGPU = true;
    if (useGPU)
    {
        LOG(ERROR) << "Using GPU";
        uint device_id = 0;
        LOG(ERROR) << "Using Device_id=" << device_id;
        Caffe::SetDevice(device_id);
        Caffe::set_mode(Caffe::GPU);
    }
    else
    {
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    }

    const string pretrainedModelSpec = R"(C:\Code\caffe\caffe-windows\matt\data\autoencoder\simulation-autoencoder-net.prototxt)";
    const string pretrainedModelParams = R"(C:\Code\caffe\caffe-windows\matt\data\autoencoder\simulation_iter_1305.caffemodel)";

    caffe::shared_ptr< Net<float> > net(new Net<float>(pretrainedModelSpec, caffe::TEST));
    net->CopyTrainedLayersFrom(pretrainedModelParams);

    vector<BlobInfo> blobs;
    blobs.push_back(BlobInfo("data", "in", 3));
    blobs.push_back(BlobInfo("f-05", "truth", 1));
    blobs.push_back(BlobInfo("deconv3", "out", 1));

    LOG(ERROR) << "All blobs:";
    for (const string &name : net->blob_names())
    {
        LOG(ERROR) << "  " << name;
    }

    for (const BlobInfo &blob : blobs)
    {
        CHECK(net->has_blob(blob.name))
            << "Unknown blob name " << blob.name << " in the network " << pretrainedModelSpec;
    }

    const int minibatchCount = 1;
    
    LOG(ERROR) << "Extacting Features";

    Datum datum;
    const int maxKeyStrLength = 100;
    char keyStr[maxKeyStrLength];

    std::vector<Blob<float>*> inputBlobs;
    int imageIndex = 0;
    for (int batchIndex = 0; batchIndex < minibatchCount; batchIndex++)
    {
        net->Forward(inputBlobs);

        for (BlobInfo &blob : blobs)
        {
            blob.data = net->blob_by_name(blob.name);
            LOG(ERROR) << blob.name << " dims: " << blob.data->width() << "x" << blob.data->height() << "x" << blob.data->channels();
        }
        
        const int imageCount = blobs[0].data->num();
        const float* blobData;
        for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
        {
            for (const BlobInfo &blob : blobs)
            {
                auto image = blobToImage(blob.data, imageIndex, blob.channelsToOutput);
                LodePNG::save(image, outputDir + "b" + to_string(batchIndex) + "i" + to_string(imageIndex) + "_" + blob.suffix + ".png");
            }

            /*datum.set_height(featureBlob->height());
            datum.set_width(featureBlob->width());
            datum.set_channels(featureBlob->channels());
            datum.clear_data();
            datum.clear_float_data();
            blobData = featureBlob->cpu_data() + featureBlob->offset(n);
            for (int d = 0; d < dim_features; ++d)
                datum.add_float_data(blobData[d]);
            int length = _snprintf(keyStr, maxKeyStrLength, "%010d", imageIndex);
            
            string out;
            CHECK(datum.SerializeToString(&out));
            transaction->Put(std::string(keyStr, length), out);
            ++imageIndex;
            if (imageIndex % 1000 == 0) {
                transaction->Commit();
                txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
                LOG(ERROR) << "Extracted features of " << image_indices[i] <<
                    " query images for feature blob " << blob_names[i];
            }*/
        }
    }

    LOG(ERROR) << "Successfully extracted the features!";
}
