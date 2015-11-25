
struct Particle
{
    vec2f position;
    vec2f velocity;

    float radius;
    vec3f color;
    float speedBase;

    vec2f forces;
    vec2f newPosition;
};

struct ParticleSystem
{
    void init(int particleCount);
    void step(float deltaT);

    void render(ColorImageR32G32B32 &image);

    vector<Particle> particles;
};
