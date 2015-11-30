
#include "main.h"

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <stdint.h>
#include <sys/stat.h>
#include <direct.h>

#include "caffe/proto/caffe.pb.h"

using namespace caffe;

void ParticleSystem::init(int particleCount, float deltaT)
{
    baseDeltaT = deltaT;

    const float avgRadius = 0.06f;
    const float radiusVariance = 0.02f;
    
    const float avgSpeed = 2.0f;
    const float speedVariance = 1.0f;
    
    particles.resize(particleCount);

    auto r = []() { return util::randomUniform(0.0f, 1.0f);  };
    auto s = []() { return util::randomUniform(-1.0f, 1.0f);  };
    auto t = [](float a, float b) { return util::randomUniform(a, b);  };

    vector<vec3f> colorList;
    colorList.push_back(vec3f(1.0f, 1.0f, 1.0f));

    colorList.push_back(vec3f(1.0f, 0.2f, 0.2f));
    colorList.push_back(vec3f(0.2f, 1.0f, 0.2f));
    colorList.push_back(vec3f(0.2f, 0.2f, 1.0f));

    colorList.push_back(vec3f(0.2f, 1.0f, 1.0f));
    colorList.push_back(vec3f(1.0f, 0.2f, 1.0f));
    colorList.push_back(vec3f(1.0f, 1.0f, 0.2f));

    colorList.push_back(vec3f(1.0f, 0.6f, 0.6f));
    colorList.push_back(vec3f(0.6f, 1.0f, 0.6f));
    colorList.push_back(vec3f(0.6f, 0.6f, 1.0f));

    colorList.push_back(vec3f(0.6f, 0.6f, 0.6f));
    int colorIndex = 0;

    for (Particle &p : particles)
    {
        p.position = vec2f(r(), r());
        p.forces = vec2f::origin;

        p.color = vec3f(t(0.5f, 1.0f), t(0.5f, 1.0f), t(0.5f, 1.0f));
        while (p.color.length() < 1.0f)
            p.color = vec3f(t(0.5f, 1.0f), t(0.5f, 1.0f), t(0.5f, 1.0f));
        p.color = vec3f(1.0f, 1.0f, 1.0f);

        //p.color = colorList[colorIndex++];

        //p.radius = avgRadius + s() * radiusVariance;
        p.radius = avgRadius;

        p.speedBase = avgSpeed + s() * speedVariance;
        p.velocity = vec2f(s(), s()).getNormalized() * p.speedBase;
    }
}

void ParticleSystem::macroStep()
{
    const int steps = 15;
    for (int i = 0; i < steps; i++)
        microStep(baseDeltaT / steps);
}

void ParticleSystem::microStep(float deltaT)
{
    const float wallSize = 0.12f;
    const float wallForce = 100.0f;
    const float speedNormFactor = 0.98f;

    const float repulsionRadius = 0.125f;
    const float repulsionForce = 70.0f;

    for (Particle &p : particles)
    {
        p.forces = vec2f::origin;

        //
        // wall forces
        //
        if (p.position.x < wallSize) p.forces += vec2f(wallForce, 0.0f);
        if (p.position.x > 1.0f - wallSize) p.forces += vec2f(-wallForce, 0.0f);
        if (p.position.y < wallSize) p.forces += vec2f(0.0f, wallForce);
        if (p.position.y > 1.0f - wallSize) p.forces += vec2f(0.0f, -wallForce);

        //
        // particle repulsion forces
        //
        for (Particle &pOther : particles)
        {
            const float dist = vec2f::dist(pOther.position, p.position);
            if (dist > 0.0f)
            {
                const float separation = dist - pOther.radius - p.radius;
                if (separation < repulsionRadius)
                {
                    p.forces += (p.position - pOther.position).getNormalized() * repulsionForce;
                }
            }
        }

        p.velocity += deltaT * p.forces;

        const float newSpeed = p.velocity.length() * speedNormFactor + p.speedBase * (1.0f - speedNormFactor);
        p.velocity = p.velocity.getNormalized() * newSpeed;

        p.newPosition = p.position + deltaT * p.velocity;
    }

    for (Particle &p : particles)
    {
        p.position = p.newPosition;
    }
}

void ParticleSystem::render(Grid2<vec3f> &imageOut)
{
    const float gamma = 6.0f;
    imageOut.setValues(vec3f::origin);

    const float radiusScale = 1.0f;
    
    for (Particle &p : particles)
    {
        for (auto &pixel : imageOut)
        {
            const float distSq = vec2f::distSq(vec2f((float)pixel.x / (imageOut.getDimX() - 1), (float)pixel.y / (imageOut.getDimY() - 1)), p.position);
            const float radius = p.radius * radiusScale;
            if (distSq < radius * radius)
            {
                const float ratio = powf(sqrtf(distSq) / radius, gamma);
                pixel.value += p.color * (1.0f - ratio);
            }
        }
    }
}

void ParticleSystem::makeDatabase(const string &databaseDir, int sampleCount)
{
    const int imageSize = 82;
    const int particleCount = 10;
    const float deltaT = 0.015f;
    
    const int historyFrames = 4;
    const int imageChannelCount = 3;

    const int totalFrames = historyFrames + 1;
    const int totalChannelCount = imageChannelCount * totalFrames;
    const int pixelCount = imageSize * imageSize;

    // leveldb
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    leveldb::WriteBatch* batch = NULL;

    // Open db
    cout << "Opening leveldb " << databaseDir << endl;
    leveldb::Status status = leveldb::DB::Open(options, databaseDir, &db);
    if (!status.ok())
    {
        cout << "Failed to open " << databaseDir << " or it already exists" << endl;
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
    datum.set_height(imageSize);
    datum.set_width(imageSize);

    ColorImageR8G8B8A8 dummyImage(imageSize, imageSize);

    cout << "A total of " << sampleCount << " samples will be generated." << endl;
    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        if (sampleIndex % 1000 == 0)
            cout << "Sample " << sampleIndex << " / " << sampleCount << endl;

        SimulationHistory history = ParticleSystem::makeSimulation(totalFrames);

        if (sampleIndex == 0)
        {
            for (int frameIndex = 0; frameIndex < totalFrames; frameIndex++)
            {
                auto image = helper::gridToImage(history.history[0]);
                LodePNG::save(image, R"(C:\Code\caffe\caffe-windows\matt\debug\sim)" + to_string(frameIndex) + ".png");
            }
        }

        int pIndex = 0;
        for (int frameIndex = 0; frameIndex < totalFrames; frameIndex++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                for (const auto &p : dummyImage)
                {
                    rawData[pIndex++] = util::boundToByte(history.history[frameIndex](p.x, p.y)[channel] * 255.0f);
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
    }
    // write the last batch
    if (count % 1000 != 0) {
        db->Write(leveldb::WriteOptions(), batch);
    }
    delete batch;
    delete db;
    cout << "Processed " << count << " files." << endl;
}

SimulationHistory ParticleSystem::makeSimulation(int frameCount)
{
    const int imageSize = 82;
    //cout << "Making simulation with " << frameCount << " frames" << endl;
    SimulationHistory result;
    const int particleCount = 10;
    const float deltaT = 0.015f;

    ParticleSystem system;
    system.init(particleCount, deltaT);
    
    for (int step = 0; step < 150; step++)
        system.macroStep();

    for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
    {
        system.macroStep();
        Grid2<vec3f> image(imageSize, imageSize);
        system.render(image);
        result.history.push_back(image);
    }

    return result;
}

SimulationHistories ParticleSystem::makeSimulations(int frameCount)
{
    SimulationHistories result;
    result.videoGridDims = vec2i(5, 3);
    
    const int simulationCount = result.videoGridDims.x * result.videoGridDims.y;

    for (int i = 0; i < simulationCount; i++)
    {
        result.histories.push_back(makeSimulation(frameCount));
    }
    return result;
}
