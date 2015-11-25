
#include "main.h"

void ParticleSystem::init(int particleCount)
{
    const float avgRadius = 0.05f;
    const float radiusVariance = 0.02f;
    
    const float avgSpeed = 2.0f;
    const float speedVariance = 1.0f;
    
    particles.resize(particleCount);

    auto r = []() { return util::randomUniform(0.0f, 1.0f);  };
    auto s = []() { return util::randomUniform(-1.0f, 1.0f);  };

    for (Particle &p : particles)
    {
        p.position = vec2f(r(), r());
        p.forces = vec2f::origin;

        p.color = vec3f(r(), r(), r());
        while (p.color.length() < 0.7f)
            p.color = vec3f(r(), r(), r());

        p.color = vec3f(1.0f, 1.0f, 1.0f);

        p.radius = avgRadius + s() * radiusVariance;

        p.speedBase = avgSpeed + s() * speedVariance;
        p.velocity = vec2f(s(), s()).getNormalized() * p.speedBase;
    }
}

void ParticleSystem::step(float deltaT)
{
    const float wallSize = 0.12f;
    const float wallForce = 100.0f;
    const float speedNormFactor = 0.98f;

    const float repulsionRadius = 0.1f;
    const float repulsionForce = 50.0f;

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

void ParticleSystem::render(ColorImageR32G32B32 &image)
{
    const float gamma = 2.0f;
    image.setPixels(vec3f::origin);
    
    for (Particle &p : particles)
    {
        for (auto &pixel : image)
        {
            const float distSq = vec2i::distSq(vec2f((float)pixel.x / (image.getWidth() - 1), (float)pixel.y / (image.getHeight() - 1)), p.position);
            if (distSq < p.radius * p.radius)
            {
                const float ratio = powf(sqrtf(distSq) / p.radius, gamma);
                pixel.value += p.color * (1.0f - ratio);
            }
        }
    }
}
