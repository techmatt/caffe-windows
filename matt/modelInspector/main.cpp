
#include "main.h"

void main(int argc, char** argv)
{
    ParameterFile params("parameters.txt");

    google::InitGoogleLogging(argv[0]);
    //caffe::GlobalInit(&argc, &argv);
    const string baseDir = R"(C:\Code\caffe\caffe-windows\matt\data\)";

    const string outputDirAnimation = R"(C:\Code\caffe\caffe-windows\matt\debug\)";
    const string outputDirCompiledVideo = baseDir + "videos/predictionVideo/";

    util::makeDirectory(outputDirAnimation);
    util::makeDirectory(outputDirCompiledVideo);
    
    Grid2<vec3f> meanValues = CaffeUtil::gridVec3FromBinaryProto(params.readString("meanFile"));

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

    const string pretrainedModelSpec = params.readString("pretrainedModelSpec");
    const string pretrainedModelParams = params.readString("pretrainedModelParams");

    caffe::shared_ptr< Net<float> > net(new Net<float>(pretrainedModelSpec, caffe::TEST));
    net->CopyTrainedLayersFrom(pretrainedModelParams);

    SimulationHistories histories;
    const int simulationCount = 15;
    const int simulationDuration = 1000;
    histories.videoGridDims = vec2i(5, 3);
    for (int sim = 0; sim < simulationCount; sim++)
    {
        LOG(ERROR) << "Running simulation " << sim;
        SimulationState simulation;
        simulation.init(net);
        for (int i = 0; i < simulationDuration; i++)
            simulation.step();

        histories.histories.push_back(simulation.history);

        if (sim == 0)
        {
            simulation.save(outputDirAnimation, meanValues);
        }
    }
    histories.saveVideoFrames(outputDirCompiledVideo, meanValues);

    /*vector<BlobInfo> blobs;
    blobs.push_back(BlobInfo("data", "in", 3));
    blobs.push_back(BlobInfo("targetImage", "truth", 3));
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

    LOG(ERROR) << "Input blob count: " << net->input_blobs().size();

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
        for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
        {
            for (const BlobInfo &blob : blobs)
            {
                auto image = CaffeUtil::blobToImage(blob.data, imageIndex, blob.channelsToOutput, meanValues);
                LodePNG::save(image, outputDirSingleFrame + "b" + to_string(batchIndex) + "i" + to_string(imageIndex) + "_" + blob.suffix + ".png");
            }
        }
    }

    LOG(ERROR) << "Successfully extracted the features!";*/
}
