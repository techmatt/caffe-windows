
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

    void render(Grid2<vec3f> &imageOut);
    
    static void makeDatabase(const string &databaseDir, int sampleCount);
    static SimulationHistory makeSimulation(int frameCount);
    static SimulationHistories makeSimulations(int frameCount);

    vector<Particle> particles;
    float baseDeltaT;
};
