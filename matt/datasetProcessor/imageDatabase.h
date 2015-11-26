
class ImageDatabase
{
public:
    struct Entry
    {
        string filenameInput;
        string filenameOutput;
        int label;
    };

    static void makeTestDatabase(const string &directory, int imageCount);

    void loadStandard(const string &directory);
    void loadTargeted(const string &directory);
    void save(const string &directory);

    bool useOutputChannel;
    vec2i dimensions;
    vector<Entry> entries;
};