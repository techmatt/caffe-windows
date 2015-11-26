
struct SimulationState
{
    void init(const Netf &_net);
    void run(int steps);
    void step();

    void save(const string &directory);
    
    Netf net;

    // history 0 is the current (predicted) image.
    deque< Grid2<float> > history;
    Grid2<float> debugPrediction;
};
