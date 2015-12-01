
class RepresentationDataset
{
public:
    static void initFolder(const string &databaseName);
    static void makeVideoDataset(const string &databaseName, const string &videoDirectory, vec2i imageDimensions, int trainingSampleCount);

private:
    static void copyFile(const string &sourceFilename, const string &targetFilename, const vector< pair<string, string> > &replacements);
};