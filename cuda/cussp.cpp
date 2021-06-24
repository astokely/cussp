#include "cussp.h"
#include "kernels/cussp.h"
#include <stdio.h>


CuSSP::CuSSP(
        std::vector<double>& weights_,
        std::vector<int>& vertices_,
        std::vector<int>& edges_,
        int source
    ) : weights(weights_), vertices(vertices_), 
        edges(edges_), sourceVertex(source) {}

extern "C" std::vector<double>& CuSSP::getPerVertexSSP(
        std::vector<double>& ssp,
        int blockSize,
        int asyncIter
    ) {
    double *h_shortestDistancesGPU = (double*)malloc(
        sizeof(double) * vertices.size() 
    );
    PerVertexSSP(weights, vertices, edges, sourceVertex, h_shortestDistancesGPU, blockSize, asyncIter);
    ssp.insert(ssp.end(), &h_shortestDistancesGPU[0], &h_shortestDistancesGPU[vertices.size()]);
    free(h_shortestDistancesGPU);
    h_shortestDistancesGPU = NULL;
    return ssp;
}


