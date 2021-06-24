#include<bits/stdc++.h>
#include <vector>

class CuSSP {
    public:
        CuSSP(
            std::vector<double>& weights_,
            std::vector<int>& vertices_,
            std::vector<int>& edges_,
            int sourceVertex
        );
        std::vector<double>& getPerVertexSSP(
            std::vector<double>& ssp,
            int blockSize,
            int asyncIter
        );
    private:
        std::vector<double> weights;
        std::vector<int> vertices;
        std::vector<int> edges;
        int sourceVertex;
};
