
#include "main.h"

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <stdint.h>
#include <sys/stat.h>
#include <direct.h>

#include "caffe/proto/caffe.pb.h"

using namespace caffe;

void SimulationHistories::saveVideoFrames(const string &directory, const Grid2<vec3f> &meanValues) const
{
    util::makeDirectory(directory);
    for (int frameIndex = 0; frameIndex < histories[0].history.size(); frameIndex++)
    {
        auto image = makeVideoFrame(frameIndex, meanValues);
        LodePNG::save(image, directory + to_string(frameIndex) + ".png");
    }
}

ColorImageR8G8B8A8 SimulationHistories::makeVideoFrame(int frameIndex, const Grid2<vec3f> &meanValues) const
{
    const int padding = 4;
    const int upscaleFactor = 2;

    const int imageSize = histories[0].history[0].getDimX();
    const int cellSize = padding * 2 + imageSize * upscaleFactor;

    ColorImageR8G8B8A8 result(cellSize * videoGridDims);
    result.setPixels(vec4uc(128, 128, 128, 255));

    int historyIndex = 0;
    for (int cellX = 0; cellX < videoGridDims.x; cellX++)
    {
        for (int cellY = 0; cellY < videoGridDims.y; cellY++)
        {
            ColorImageR8G8B8A8 historyImage = helper::gridToImage(histories[historyIndex++].history[frameIndex], meanValues);
            historyImage = helper::upsample(historyImage);
            
            for (auto &p : historyImage)
            {
                vec2i pixel = vec2i(cellX, cellY) * cellSize + vec2i(padding, padding) + vec2i(p.x, p.y);
                result(pixel.x, pixel.y) = p.value;
            }
        }
    }
    return result;
}

void helper::processVideoFolder(const string &inputDir, const string &outputDir, const bbox2i &rect, int mipMapCount)
{
    vector<string> allImages;
    for (const string &filename : Directory::enumerateFilesWithPath(inputDir, ".png"))
    {
        allImages.push_back(filename);
    }
    sort(allImages.begin(), allImages.end());
    cout << "Processing " << allImages.size() << " files" << endl;
    util::makeDirectory(outputDir);
    
    for (int imageIndex = 0; imageIndex < allImages.size(); imageIndex++)
    {
        auto imageRaw = LodePNG::load(allImages[imageIndex]);
        ColorImageR8G8B8A8 imageProcessed(rect.getExtentX(), rect.getExtentY());
        for (auto &p : imageProcessed)
        {
            p.value = imageRaw(p.x + rect.getMinX(), p.y + rect.getMinY());
        }
        for (int mipMap = 0; mipMap < mipMapCount; mipMap++)
        {
            ColorImageR8G8B8A8 out(imageProcessed.getWidth() / 2, imageProcessed.getHeight() / 2);
            
            for (unsigned int y = 0; y < out.getHeight(); y++) {
                for (unsigned int x = 0; x < out.getWidth(); x++) {
                    vec4f sum = vec4f(imageProcessed.getPixel(2 * x + 0, 2 * y + 0)) +
                                vec4f(imageProcessed.getPixel(2 * x + 1, 2 * y + 0)) +
                                vec4f(imageProcessed.getPixel(2 * x + 0, 2 * y + 1)) +
                                vec4f(imageProcessed.getPixel(2 * x + 1, 2 * y + 1));
                    out(x, y) = sum * 0.25f;
                }
            }

            imageProcessed = out;
        }
        
        const bool blackAndWhiteFilter = false;
        if (blackAndWhiteFilter)
        {
            for (const auto& p : imageProcessed)
            {
                const vec4f color = vec4f(p.value);
                const float value = 0.299f * p.value.x + 0.587f * p.value.y + 0.114f * p.value.z;
                const BYTE c = util::boundToByte(value);
                p.value = vec4uc(c, c, c, 255);
            }
        }

        LodePNG::save(imageProcessed, outputDir + "frame" + util::zeroPad(imageIndex, 5) + ".png");
    }
}

void helper::videoToSamples(const string &inputDir, const string &outputDir, int imageSize, int sampleCount)
{
    const int samplesPerBlock = 10;

    ColorImageR8G8B8A8 imageHistorySave(imageSize, imageSize);
    ColorImageR8G8B8A8 imageNextSave(imageSize, imageSize);

    auto imagePaths = Directory::enumerateFilesWithPath(inputDir, ".png");
    sort(imagePaths.begin(), imagePaths.end());

    util::makeDirectory(outputDir);
    for (int sampleIndex = 0; sampleIndex < sampleCount;)
    {
        if (sampleIndex % 1000 == 0)
            cout << "Sample " << sampleIndex << " / " << sampleCount << endl;

        vector<ColorImageR8G8B8A8> sampleImages;
        const int startImageIndex = util::randomInteger(0, imagePaths.size() - 6);
        for (int i = 0; i < 5; i++)
            sampleImages.push_back(LodePNG::load(imagePaths[startImageIndex + i]));
        
        for (int blockSampleIndex = 0; blockSampleIndex < samplesPerBlock; blockSampleIndex++)
        {
            vec2i sampleStart(
                        util::randomInteger(0, sampleImages[0].getWidth()  - imageSize - 2),
                        util::randomInteger(0, sampleImages[0].getHeight() - imageSize - 2));
            for (auto &p : imageHistorySave)
            {
                for (int channel = 0; channel < 4; channel++)
                {
                    p.value[channel] = sampleImages[3 - channel](sampleStart.x + p.x, sampleStart.y + p.y).r;
                }
            }

            for (auto &p : imageNextSave)
            {
                char c = sampleImages[4](sampleStart.x + p.x, sampleStart.y + p.y).r;
                p.value = vec4uc(c, c, c, 255);
            }

            LodePNG::save(imageHistorySave, outputDir + to_string(sampleIndex) + "_input.png", true);
            LodePNG::save(imageNextSave, outputDir + to_string(sampleIndex) + "_output.png", true);
            sampleIndex++;
        }
    }
}

void helper::videoToDatabase(const string &inputVideoDir, const string &outputDatabaseDir, vec2i dimensions, int sampleCount)
{
    const int historyFrames = 4;
    const int imageChannelCount = 3;

    const int totalFrames = historyFrames + 1;
    const int totalChannelCount = imageChannelCount * totalFrames;
    const int pixelCount = dimensions.x * dimensions.y;

    const int samplesPerBlock = 10;

    auto videoImagePaths = Directory::enumerateFilesWithPath(inputVideoDir, ".png");
    sort(videoImagePaths.begin(), videoImagePaths.end());

    cout << "Making video database for " << inputVideoDir << endl;

    // leveldb
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

    // Storing to db
    char* rawData = new char[pixelCount * totalChannelCount];

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
                util::randomInteger(0, sampleImages[0].getWidth()  - dimensions.x - 2),
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
    cout << "Processed " << count << " files." << endl;
}
