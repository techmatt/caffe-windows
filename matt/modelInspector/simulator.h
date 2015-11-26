
struct SimulationState
{
    void init(const Netf &_net);
    void run(int steps);
    void step();

    void save(const string &directory);
    
    Netf net;

    SimulationHistory history;
    Grid2<float> debugPrediction;
};
