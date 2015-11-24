
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

/*#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"*/

using namespace caffe;  // NOLINT(build/namespaces)
using namespace google;
using std::string;

/*namespace caffe
{
    //extern REGISTER_LAYER_CLASS(Data);
    extern INSTANTIATE_CLASS(BaseDataLayer);
    extern INSTANTIATE_CLASS(BasePrefetchingDataLayer);
    extern INSTANTIATE_CLASS(DataLayer);
    extern INSTANTIATE_CLASS(ConvolutionLayer);
    extern INSTANTIATE_CLASS(PoolingLayer);
    extern INSTANTIATE_CLASS(ReLULayer);
    extern INSTANTIATE_CLASS(TanHLayer);
}*/

namespace caffe
{
    //REGISTER_LAYER_CLASS(Data);
}

void main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    //caffe::GlobalInit(&argc, &argv);
    const string baseDir = R"(C:\Code\caffe\caffe-windows\matt\data\)";
    
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

    const string blobName = "deconv3";

    LOG(ERROR) << "All blobs:";
    for (const string &name : net->blob_names())
    {
        LOG(ERROR) << "  " << name;
    }

    return;
    CHECK(net->has_blob(blobName))
        << "Unknown feature blob name " << blobName << " in the network " << pretrainedModelSpec;

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

        const caffe::shared_ptr< Blob<float> > featureBlob = net->blob_by_name(blobName);
        int batch_size = featureBlob->num();
        int dim_features = featureBlob->count() / batch_size;
        const float* blobData;
        for (int n = 0; n < batch_size; ++n)
        {
            datum.set_height(featureBlob->height());
            datum.set_width(featureBlob->width());
            datum.set_channels(featureBlob->channels());
            datum.clear_data();
            datum.clear_float_data();
            blobData = featureBlob->cpu_data() + featureBlob->offset(n);
            /*for (int d = 0; d < dim_features; ++d)
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
