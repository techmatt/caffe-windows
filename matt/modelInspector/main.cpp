
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

namespace caffe
{
    //REGISTER_LAYER_CLASS(Data);
}

ColorImageR8G8B8A8 blobToImage(const caffe::shared_ptr< Blob<float> > &blob, int imageIndex)
{
    ColorImageR8G8B8A8 image(blob->width(), blob->height());
    for (auto &p : image)
    {
        for (int channel = 0; channel < 3; channel++)
        {
            const float *dataStart = blob->cpu_data() + blob->offset(imageIndex, channel, p.x, p.y);
            const BYTE value = util::boundToByte(*dataStart * 255.0f);
            p.value[channel] = value;
        }
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

    const string pretrainedModelSpec = R"(C:\Code\caffe\caffe-windows\matt\data\autoencoder\circle-autoencoder-net.prototxt)";
    const string pretrainedModelParams = R"(C:\Code\caffe\caffe-windows\matt\data\autoencoder\snapshot_iter_11873.caffemodel)";

    caffe::shared_ptr< Net<float> > net(new Net<float>(pretrainedModelSpec, caffe::TEST));
    net->CopyTrainedLayersFrom(pretrainedModelParams);

    const string dataBlobName = "data";
    const string featureBlobName = "deconv3";

    LOG(ERROR) << "All blobs:";
    for (const string &name : net->blob_names())
    {
        LOG(ERROR) << "  " << name;
    }

    CHECK(net->has_blob(dataBlobName))
        << "Unknown feature blob name " << dataBlobName << " in the network " << pretrainedModelSpec;

    CHECK(net->has_blob(featureBlobName))
        << "Unknown feature blob name " << featureBlobName << " in the network " << pretrainedModelSpec;

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

        const caffe::shared_ptr< Blob<float> > dataBlob = net->blob_by_name(dataBlobName);
        const caffe::shared_ptr< Blob<float> > featureBlob = net->blob_by_name(featureBlobName);

        LOG(ERROR) << "data dims: " << dataBlob->width() << "x" << dataBlob->height() << "x" << dataBlob->channels();
        LOG(ERROR) << "feature dims: " << featureBlob->width() << "x" << featureBlob->height() << "x" << dataBlob->channels();

        int imageCount = featureBlob->num();
        const float* blobData;
        for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
        {
            auto imageA = blobToImage(dataBlob, imageIndex);
            auto imageB = blobToImage(featureBlob, imageIndex);

            LodePNG::save(imageA, outputDir + "b" + to_string(batchIndex) + "i" + to_string(imageIndex) + "_in.png");
            LodePNG::save(imageB, outputDir + "b" + to_string(batchIndex) + "i" + to_string(imageIndex) + "_out.png");

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
