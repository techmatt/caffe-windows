
struct SimulationState
{
    void init(const Netf &_net);
    void run(int steps);
    void step();

    void save(const string &directory, const Grid2<vec3f> &meanValues);

    Netf net;

    SimulationHistory history;
    Grid2<vec3f> debugPrediction;
};
