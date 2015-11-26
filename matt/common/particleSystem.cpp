
#include "main.h"

void ParticleSystem::init(int particleCount, float deltaT)
{
    baseDeltaT = deltaT;

    const float avgRadius = 0.05f;
    const float radiusVariance = 0.02f;
    
    const float avgSpeed = 2.0f;
    const float speedVariance = 1.0f;
    
    particles.resize(particleCount);

    auto r = []() { return util::randomUniform(0.0f, 1.0f);  };
    auto s = []() { return util::randomUniform(-1.0f, 1.0f);  };

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

        //p.color = vec3f(r(), r(), r());
        //while (p.color.length() < 0.7f)
        //    p.color = vec3f(r(), r(), r());
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

void ParticleSystem::render(Grid2<float> &imageOut)
{
    const float gamma = 1.0f;
    imageOut.setValues(0.0f);

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
                pixel.value += p.color.r * (1.0f - ratio);
            }
        }
    }
}

void ParticleSystem::renderChain(ColorImageR32G32B32A32 &imageHistory, ColorImageR32G32B32A32 &imageNext)
{
    Grid2<float> storage(imageHistory.getWidth(), imageHistory.getHeight());
    for (int history = 0; history < 4; history++)
    {
        int channel = 3 - history;
        
        render(storage);
        for (auto &p : imageHistory)
        {
            p.value[channel] = storage(p.x, p.y);
        }
        macroStep();
    }

    render(storage);
    for (auto &p : imageNext)
    {
        p.value[0] = storage(p.x, p.y);
    }
}

void ParticleSystem::makeDatabase(const string &directory, int imageCount)
{
    const int imageSize = 82;
    const int particleCount = 10;
    const float deltaT = 0.015f;
    
    ColorImageR32G32B32A32 imageHistory(imageSize, imageSize);
    ColorImageR32G32B32A32 imageNext(imageSize, imageSize);
    
    ColorImageR8G8B8A8 imageHistorySave(imageSize, imageSize);
    ColorImageR8G8B8A8 imageNextSave(imageSize, imageSize);

    util::makeDirectory(directory);
    for (int imageIndex = 0; imageIndex < imageCount; imageIndex++)
    {
        if (imageIndex % 1000 == 0)
            cout << "Image " << imageIndex << " / " << imageCount << endl;

        ParticleSystem system;
        system.init(particleCount, deltaT);

        for (int step = 0; step < 150; step++)
            system.macroStep();

        system.renderChain(imageHistory, imageNext);

        for (auto &p : imageHistorySave)
        {
            vec4f color = imageHistory(p.x, p.y);
            p.value = vec4uc(util::boundToByte(color.x * 255.0f),
                             util::boundToByte(color.y * 255.0f),
                             util::boundToByte(color.z * 255.0f),
                             util::boundToByte(color.a * 255.0f));
        }

        for (auto &p : imageNextSave)
        {
            vec4f color = imageNext(p.x, p.y);
            char c = util::boundToByte(color.x * 255.0f);
            p.value = vec4uc(c, c, c, 255);
        }

        LodePNG::save(imageHistorySave, directory + to_string(imageIndex) + "_input.png", true);
        LodePNG::save(imageNextSave, directory + to_string(imageIndex) + "_output.png", true);
    }
}

SimulationHistory ParticleSystem::makeSimulation(int frameCount)
{
    const int imageSize = 82;
    cout << "Making simulation with " << frameCount << " frames" << endl;
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
        Grid2<float> image(imageSize, imageSize);
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