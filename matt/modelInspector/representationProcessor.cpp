
#include "main.h"

const string outputLayerName = "deconv3";

const string baseDir = R"(C:\Code\caffe\caffe-windows\matt\representationData\)";

vector<ColorImageR8G8B8A8> RepresentationProcessor::getRandomVideoSequence(const string &inputVideoDir, int sequenceLength)
{
    vector<ColorImageR8G8B8A8> result;
    const int startImageIndex = util::randomInteger(0, videoImagePaths.size() - sequenceLength - 1);
    for (int i = 0; i < sequenceLength; i++)
        result.push_back(LodePNG::load(videoImagePaths[startImageIndex + i]));
    return result;
}

vector<ColorImageR8G8B8A8> RepresentationProcessor::getRandomVideoClip(const vector<ColorImageR8G8B8A8> &sequence, vec2i clipDimensions)
{
    vec2i sampleStart(
        util::randomInteger(0, sequence[0].getWidth() - clipDimensions.x - 2),
        util::randomInteger(0, sequence[0].getHeight() - clipDimensions.y - 2));

    vector<ColorImageR8G8B8A8> result;
    for (int frameIndex = 0; frameIndex < sequence.size(); frameIndex++)
    {
        ColorImageR8G8B8A8 image(clipDimensions);
        for (auto &p : image)
        {
            p.value = sequence[frameIndex](sampleStart.x + p.x, sampleStart.y + p.y);
        }
        result.push_back(image);
    }
    return result;
}

vector< Grid2<vec3f> > RepresentationProcessor::applyTranformsVideoClip(const vector<ColorImageR8G8B8A8> &sequence)
{
    vector< Grid2<vec3f> > result;
    for (const ColorImageR8G8B8A8 &image : sequence)
    {
        Grid2<vec3f> grid(image.getDimensions());
        for (auto &v : grid)
        {
            v.value = (vec4f(image(v.x, v.y)).getVec3() + meanValues(v.x, v.y)) / 255.0f;
        }
        result.push_back(grid);
    }
    return result;
}

Grid3<float> RepresentationProcessor::extractFiveFrameWindow(const vector< Grid2<vec3f> > &transformedClip, int startFrame)
{
    Grid3<float> result(clipDimensions.x, clipDimensions.y, 15);

    int resultChannel = 0;
    for (int offset = 0; offset < 5; offset++)
    {
        const Grid2<vec3f> &grid = transformedClip[startFrame + offset];
        for (int channel = 0; channel < 3; channel++)
        {
            for (auto &v : grid)
            {
                result(v.x, v.y, resultChannel) = v.value[channel];
            }
            resultChannel++;
        }
    }
    return result;
}

Grid3<float> RepresentationProcessor::applyNetTransform(const Grid3<float> &extractedClip, const Netf &net)
{
    return CaffeUtil::getBlobAsGrid(net, "ip2-squashed");
}

void RepresentationProcessor::makeProcessedDataset(const string &databaseName, const string &inputVideoDir, vec2i videoClipDimensions, const string &databaseSuffix, int sampleCount)
{
    videoImagePaths = Directory::enumerateFilesWithPath(inputVideoDir, ".png");
    sort(videoImagePaths.begin(), videoImagePaths.end());

    const string localDir = baseDir + databaseName + "/";
    clipDimensions = videoClipDimensions;
    meanValues = CaffeUtil::gridVec3FromBinaryProto(localDir + databaseName + "-imageset-mean.binaryproto");

    /*const int frameChannels = 64;
    const int frameX = 16;
    const int frameY = 16;

    const int encodedFrames = 3;

    const int precodeHistoryFrames = 4;
    const int precodeImageChannelCount = 3;

    const int precodeTotalFrames = precodeHistoryFrames + 1;
    const int precodeTotalChannelCount = precodeImageChannelCount * precodeTotalFrames;
    const int precodePixelCount = dimensions.x * dimensions.y;

    //
    // we have to load new images from the original video folder -- we need video histories with 9 frames.
    //

    const string databaseDir = baseDir + databaseName + "/";
    const string databaseSuffix = "train";
    const string outputDatabaseDir = databaseDir + databaseName + "-processed-" + databaseSuffix + "-leveldb";

    cout << "Making processed database for " << databaseName << endl;

    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    leveldb::WriteBatch* batch = NULL;

    // Open db
    cout << "Opening leveldb " << outputDatabaseDir << endl;
    leveldb::Status status = leveldb::DB::Open(options, outputDatabaseDir, &db);
    if (!status.ok())
    {
        cout << "Failed to open " << outputDatabaseDir << " or it already exists" << endl;
        return;
    }
    batch = new leveldb::WriteBatch();

    int count = 0;
    const int kMaxKeyLength = 10;
    char key_cstr[kMaxKeyLength];
    string value;

    Datum datum;
    datum.set_channels(totalChannelCount);
    datum.set_height(dimensions.x);
    datum.set_width(dimensions.y);

    ColorImageR8G8B8A8 dummyImage(dimensions);

    cout << "A total of " << sampleCount << " samples will be generated." << endl;
    cout << "Rows: " << dimensions.x << " Cols: " << dimensions.y << endl;
    for (int sampleIndex = 0; sampleIndex < sampleCount;)
    {
        if (sampleIndex % 1000 == 0)
            cout << "Sample " << sampleIndex << " / " << sampleCount << endl;

        vector<ColorImageR8G8B8A8> sampleImages;
        const int startImageIndex = util::randomInteger(0, videoImagePaths.size() - 6);
        for (int i = 0; i < 5; i++)
            sampleImages.push_back(LodePNG::load(videoImagePaths[startImageIndex + i]));

        for (int blockSampleIndex = 0; blockSampleIndex < samplesPerBlock; blockSampleIndex++)
        {
            vec2i sampleStart(
                util::randomInteger(0, sampleImages[0].getWidth() - dimensions.x - 2),
                util::randomInteger(0, sampleImages[0].getHeight() - dimensions.y - 2));

            int pIndex = 0;

            for (int frameIndex = 0; frameIndex < totalFrames; frameIndex++)
            {
                for (int channel = 0; channel < 3; channel++)
                {
                    for (const auto &p : dummyImage)
                    {
                        rawData[pIndex++] = sampleImages[frameIndex](sampleStart.x + p.x, sampleStart.y + p.y)[channel];
                    }
                }
            }

            datum.set_data(rawData, pixelCount * totalChannelCount);
            datum.set_label(0);

            sprintf_s(key_cstr, kMaxKeyLength, "%08d", sampleIndex);
            datum.SerializeToString(&value);
            string keystr(key_cstr);

            // Put in db
            batch->Put(keystr, value);

            if (++count % 1000 == 0) {
                // Commit txn
                db->Write(leveldb::WriteOptions(), batch);
                delete batch;
                batch = new leveldb::WriteBatch();
            }

            sampleIndex++;
        }
    }
    // write the last batch
    if (count % 1000 != 0) {
        db->Write(leveldb::WriteOptions(), batch);
    }
    delete batch;
    delete db;
    cout << "Processed " << count << " files." << endl;*/
}