#include <vector>

extern "C"{
    void PerVertexSSP(
        std::vector<double>& weights, 
        std::vector<int>& vertices, 
        std::vector<int>& edges,
        int sourceVertex,
        double *h_shortestDistancesGPU,
        int blockSize,
        int asyncIter
    );
}

