
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
    void init(int particleCount, float deltaT);
    void microStep(float deltaT);
    void macroStep();

    void render(ColorImageR32G32B32A32 &image);
    void renderChain(ColorImageR32G32B32A32 &imageStart, ColorImageR32G32B32A32 &imageEnd);

    static void makeDatabase(const string &directory, int imageCount);

    vector<Particle> particles;
    float baseDeltaT;
};
