
#include "mLibInclude.h"

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <stdint.h>
#include <sys/stat.h>
#include <direct.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/proto/caffe.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

class ImageDatabase
{
public:
    struct Entry
    {
        string filename;
        int label;
    };

    static void makeTestDatabase(const string &directory, int imageCount);

    void load(const string &directory);
    void save(const string &directory);

    vec2i dimensions;
    vector<Entry> entries;
};

void ImageDatabase::makeTestDatabase(const string &directory, int imageCount)
{
    util::makeDirectory(directory);
    
}

void ImageDatabase::load(const string &directory)
{
    entries.clear();
    auto files = Directory::enumerateFilesWithPath(directory, ".png");
    cout << "Loading " << directory << " " << files.size() << " files" << endl;
    for (const string &filename : files)
    {
        Entry entry;
        entry.filename = filename;
        entry.label = -1;
        entries.push_back(entry);
    }

    const auto image = LodePNG::load(entries[0].filename);
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
    const int channelCount = 3;
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
        
        auto image = LodePNG::load(entry.filename);
        int pIndex = 0;
        for (const auto &p : image)
        {
            pixelData[pIndex++] = p.value.r;
            pixelData[pIndex++] = p.value.g;
            pixelData[pIndex++] = p.value.b;
        }

        datum.set_data(pixelData, pixelCount * channelCount);
        datum.set_label(entry.label);

        sprintf_s(key_cstr, kMaxKeyLength, "%08d", item_id);
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
        delete batch;
        delete db;
        cout << "Processed " << count << " files.";
    }
}

void main()
{
    ImageDatabase database;
    database.load();
}
