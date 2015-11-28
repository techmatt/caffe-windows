
#include "main.h"

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <stdint.h>
#include <sys/stat.h>
#include <direct.h>

#include "caffe/proto/caffe.pb.h"

using namespace caffe;

void ImageDatabase::makeTestDatabase(const string &directory, int imageCount)
{
    const int size = 82;
    const float radius = 20.0f;

    util::makeDirectory(directory);
    for (int i = 0; i < imageCount; i++)
    {
        ColorImageR8G8B8A8 image(size, size);
        image.setPixels(vec4uc(0, 0, 0, 255));
        vec2f center(rand() % size, rand() % size);

        for (auto &p : image)
        {
            const float distSq = vec2f::distSq(vec2f(p.x, p.y), center);
            if (distSq < radius * radius)
            {
                const float ratio = sqrtf(distSq) / radius;
                const BYTE c = util::boundToByte((1.0f - ratio) * 255.0f);
                p.value = vec4uc(c, c, c, 255);
            }
        }

        LodePNG::save(image, directory + to_string(i) + ".png");
    }
}

void ImageDatabase::loadStandard(const string &directory)
{
    useOutputChannel = false;
    entries.clear();
    auto files = Directory::enumerateFilesWithPath(directory, ".png");
    cout << "Loading " << directory << " " << files.size() << " files" << endl;
    for (const string &filename : files)
    {
        Entry entry;
        entry.filenameInput = filename;
        entry.label = -1;
        entries.push_back(entry);
    }

    const auto image = LodePNG::load(entries[0].filenameInput);
    dimensions = image.getDimensions();
}

void ImageDatabase::loadTargeted(const string &directory)
{
    useOutputChannel = true;
    entries.clear();
    auto files = Directory::enumerateFilesWithPath(directory, ".png");
    cout << "Loading " << directory << " " << files.size() << " files" << endl;
    for (const string &filename : files)
    {
        if (util::contains(filename, "_input"))
        {
            Entry entry;
            entry.filenameInput = filename;
            entry.filenameOutput = util::replace(filename, "_input", "_output");
            entry.label = -1;
            entries.push_back(entry);
        }
    }

    const auto image = LodePNG::load(entries[0].filenameInput);
    dimensions = image.getDimensions();
}

void ImageDatabase::save(const string &directory)
{
    // leveldb
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    leveldb::WriteBatch* batch = NULL;

    // Open db
    cout << "Opening leveldb " << directory << endl;
    leveldb::Status status = leveldb::DB::Open(options, directory, &db);
    if (!status.ok())
    {
        cout << "Failed to open " << directory << " or it already exists" << endl;
        return;
    }
    batch = new leveldb::WriteBatch();
    
    // Storing to db
    const int pixelCount = dimensions.x * dimensions.y;
    
    int channelCount = useOutputChannel ? 5 : 3;

    char* pixelData = new char[pixelCount * channelCount];

    int count = 0;
    const int kMaxKeyLength = 10;
    char key_cstr[kMaxKeyLength];
    string value;

    Datum datum;
    datum.set_channels(channelCount);
    datum.set_height(dimensions.x);
    datum.set_width(dimensions.y);

    cout << "A total of " << entries.size() << " items." << endl;
    cout << "Rows: " << dimensions.x << " Cols: " << dimensions.y << endl;
    for (int entryIndex = 0; entryIndex < entries.size(); entryIndex++)
    {
        const auto &entry = entries[entryIndex];
        
        auto imageInput = LodePNG::load(entry.filenameInput);

        int pIndex = 0;
        if (useOutputChannel)
        {
            for (int channel = 0; channel < 4; channel++)
            {
                for (const auto &p : imageInput)
                {
                    pixelData[pIndex++] = p.value[channel];
                }
            }

            auto imageOutput = LodePNG::load(entry.filenameOutput);
            for (int channel = 0; channel < 1; channel++)
            {
                for (const auto &p : imageOutput)
                {
                    pixelData[pIndex++] = p.value[channel];
                }
            }
        }
        else
        {
            for (int channel = 0; channel < 3; channel++)
            {
                for (const auto &p : imageInput)
                {
                    pixelData[pIndex++] = p.value[channel];
                }
            }
        }

        datum.set_data(pixelData, pixelCount * channelCount);
        datum.set_label(entry.label);

        sprintf_s(key_cstr, kMaxKeyLength, "%08d", entryIndex);
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
    }
    // write the last batch
    if (count % 1000 != 0) {
        db->Write(leveldb::WriteOptions(), batch);
    }
    delete batch;
    delete db;
    cout << "Processed " << count << " files.";
}
