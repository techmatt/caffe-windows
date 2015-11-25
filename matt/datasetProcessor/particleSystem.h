
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
    void microStep(float deltaT);
    void macroStep(float deltaT);

    void render(ColorImageR32G32B32 &image);

    static void makeDatabase(const string &directory, int imageCount);

    vector<Particle> particles;
};
