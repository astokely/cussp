/*
By Andy Stokely
*/

#include <sstream>
#include <time.h>  
#include <vector>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "cussp.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
        cudaError_t code, 
        const char *file, 
        int line, 
        bool abort=true
) {
    if (code != cudaSuccess) { 
        fprintf(
            stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code),
            file, line
        );
        if (abort) {
            exit(code);
        }
    }
}

int iDivUp(
        int a, 
        int b
) { 
    return (a % b != 0) ? (a / b + 1) : (a / b); 
}

__device__ __forceinline__ double atomicMinDouble(
    double * addr, 
    double val
) {
    unsigned long long int* addr_as_ull = (unsigned long long int*) addr;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            addr_as_ull, 
            assumed,
            __double_as_longlong(
                fmin(val, 
                __longlong_as_double(assumed))
            )
        );
    } while (assumed != old);
    return __longlong_as_double(old);
}

typedef struct {
    unsigned int *vertexArray;
    unsigned int numVertices;
    unsigned int *edgeArray;
    unsigned int numEdges;
    double *weightArray;

} GraphData;

void freeGraphData (
        GraphData *graph
) {
    free(graph -> vertexArray);
    free(graph -> edgeArray);
    free(graph -> weightArray);
}

bool allFinalizedVertices(
        bool *finalizedVertices, 
        unsigned int numVertices
) {

    for (int i = 0; i < numVertices; i++) {
        if (finalizedVertices[i] == true) {
            return false;
        }
    }
    return true;
}

__global__ void initializeArrays(
        bool * __restrict__ d_finalizedVertices, 
        double* __restrict__ d_shortestDistances, 
        double* __restrict__ d_updatingShortestDistances,
        const unsigned int sourceVertex,
        const unsigned int numVertices
) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (sourceVertex == tid) {

            d_finalizedVertices[tid]            = true;
            d_shortestDistances[tid]            = 0.f;
            d_updatingShortestDistances[tid]    = 0.f; }

        else {

            d_finalizedVertices[tid]            = false;
            d_shortestDistances[tid]            = DBL_MAX;
            d_updatingShortestDistances[tid]    = DBL_MAX;
        }
    }
}

__global__  void Kernel1(
        const unsigned int * __restrict__ vertexArray, 
        const unsigned int* __restrict__ edgeArray,
        const double * __restrict__ weightArray, 
        bool * __restrict__ finalizedVertices,
        double* __restrict__ shortestDistances,
        double * __restrict__ updatingShortestDistances,
        const unsigned int numVertices, 
        const unsigned int numEdges
) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (finalizedVertices[tid] == true) {

            finalizedVertices[tid] = false;

            unsigned int edgeStart = vertexArray[tid], edgeEnd;

            if (tid + 1 < (numVertices)) {
                edgeEnd = vertexArray[tid + 1];
            }
            else {
                edgeEnd = numEdges;
            }

            for (unsigned int edge = edgeStart; edge < edgeEnd; edge++) {
                unsigned int nid = edgeArray[edge];
                
                atomicMinDouble(
                    &updatingShortestDistances[nid], 
                    shortestDistances[tid] + weightArray[edge]
                );
            }
        }
    }
}

__global__  void Kernel2(
        const unsigned int * __restrict__ vertexArray, 
        const unsigned int * __restrict__ edgeArray,
        const double* __restrict__ weightArray,
        bool * __restrict__ finalizedVertices,
        double* __restrict__ shortestDistances, 
        double* __restrict__ updatingShortestDistances,
        const unsigned int numVertices
) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (shortestDistances[tid] > updatingShortestDistances[tid]) {
            shortestDistances[tid] = updatingShortestDistances[tid];
            finalizedVertices[tid] = true;
        }

        updatingShortestDistances[tid] = shortestDistances[tid];
    }
}

void dijkstraGPU(
        GraphData *graph,
        const unsigned int sourceVertex,
        double * __restrict__ h_shortestDistances,
        const unsigned blockSize,
        const unsigned asyncIter
) {
    unsigned int *d_vertexArray; gpuErrchk(cudaMalloc(
        &d_vertexArray, 
        sizeof(unsigned int) * graph -> numVertices
    ));
    unsigned int *d_edgeArray; gpuErrchk(cudaMalloc(
        &d_edgeArray, 
        sizeof(unsigned int) * graph -> numEdges
    ));
    double *d_weightArray; gpuErrchk(cudaMalloc(
        &d_weightArray, 
        sizeof(double) * graph -> numEdges
    ));

    gpuErrchk(cudaMemcpy(
        d_vertexArray, 
        graph -> vertexArray, 
        sizeof(unsigned int) * graph -> numVertices, 
        cudaMemcpyHostToDevice
    ));
    gpuErrchk(cudaMemcpy(
        d_edgeArray, 
        graph -> edgeArray,   
        sizeof(unsigned int)   * graph -> numEdges,
        cudaMemcpyHostToDevice
    ));
    gpuErrchk(cudaMemcpy(
        d_weightArray, 
        graph -> weightArray, 
        sizeof(double) * graph -> numEdges,
        cudaMemcpyHostToDevice
    ));

    bool *d_finalizedVertices; gpuErrchk(cudaMalloc(
        &d_finalizedVertices, 
        sizeof(bool) * graph->numVertices
    ));
    double *d_shortestDistances; gpuErrchk(cudaMalloc(
        &d_shortestDistances,
        sizeof(double) * graph->numVertices
    ));
    double *d_updatingShortestDistances; gpuErrchk(cudaMalloc(
        &d_updatingShortestDistances, 
        sizeof(double) * graph->numVertices
    ));

    bool *h_finalizedVertices = (bool *)malloc(
        sizeof(bool) * graph->numVertices
    );

    initializeArrays <<<iDivUp(graph->numVertices, blockSize), blockSize >>>(
        d_finalizedVertices, 
        d_shortestDistances,
        d_updatingShortestDistances,
        sourceVertex, 
        graph -> numVertices
    );
    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(
        h_finalizedVertices, 
        d_finalizedVertices, 
        sizeof(bool) * graph->numVertices, 
        cudaMemcpyDeviceToHost
    ));
    
    while (!allFinalizedVertices(h_finalizedVertices, graph->numVertices)) {

        for (
            int iter = 0; 
            iter < asyncIter; 
            iter++
        ) {
            Kernel1 <<<iDivUp(graph->numVertices, blockSize), blockSize >>>(
                d_vertexArray, 
                d_edgeArray, 
                d_weightArray, 
                d_finalizedVertices, 
                d_shortestDistances,
                d_updatingShortestDistances, 
                graph->numVertices, 
                graph->numEdges
            );
            //gpuErrchk(cudaPeekAtLastError());
            //gpuErrchk(cudaDeviceSynchronize());
            Kernel2 <<<iDivUp(graph->numVertices, blockSize), blockSize >>>(
                d_vertexArray, 
                d_edgeArray, 
                d_weightArray, 
                d_finalizedVertices, 
                d_shortestDistances, 
                d_updatingShortestDistances,
                graph->numVertices
            );
            //gpuErrchk(cudaPeekAtLastError());
            //gpuErrchk(cudaDeviceSynchronize());
        }
        gpuErrchk(cudaMemcpy(
            h_finalizedVertices, 
            d_finalizedVertices, 
            sizeof(bool) * graph->numVertices, 
            cudaMemcpyDeviceToHost
        ));
    }
    gpuErrchk(cudaMemcpy(
        h_shortestDistances, 
        d_shortestDistances, 
        sizeof(double) * graph->numVertices, 
        cudaMemcpyDeviceToHost
    ));

    free(h_finalizedVertices);

    gpuErrchk(cudaFree(d_vertexArray));
    gpuErrchk(cudaFree(d_edgeArray));
    gpuErrchk(cudaFree(d_weightArray));
    gpuErrchk(cudaFree(d_finalizedVertices));
    gpuErrchk(cudaFree(d_shortestDistances));
    gpuErrchk(cudaFree(d_updatingShortestDistances));
}

void PerVertexSSP(
    std::vector<double>& weights, 
    std::vector<int>& vertices, 
    std::vector<int>& edges,
    int sourceVertex,
    double *h_shortestDistancesGPU,
    int asyncIter,
    int blockSize
) {
    GraphData graph;
    graph.numVertices = vertices.size();
    graph.numEdges = edges.size();
    graph.vertexArray = (unsigned int *)malloc(
        graph.numVertices*sizeof(unsigned int)
    );
    graph.edgeArray = (unsigned int *)malloc(
        graph.numEdges*sizeof(unsigned int)
    );
    graph.weightArray = (double *)malloc(
        graph.numEdges*sizeof(double)
    );
    std::copy(vertices.begin(), vertices.end(), graph.vertexArray);
    std::copy(edges.begin(), edges.end(), graph.edgeArray);
    std::copy(weights.begin(), weights.end(), graph.weightArray);
    double *weightMatrix = (double *)malloc(
        graph.numVertices * graph.numVertices * sizeof(double)
    );
    std::fill_n(weightMatrix, graph.numVertices * graph.numVertices, DBL_MAX);
    dijkstraGPU(&graph, sourceVertex, h_shortestDistancesGPU, asyncIter, blockSize);
    free(weightMatrix);
    freeGraphData(&graph);
}



int main() {

    std::vector<double> weights;
    std::vector<int> vertices;
    std::vector<int> edges;

    int sourceVertex = 0;
    int asyncIter = 5;
    int blockSize = 64;

    std::ifstream win("weights.txt");
    double weight;
    while (win >> weight)
    {
        weights.push_back(weight);
    }

    std::ifstream ein("edges.txt");
    int edge;
    while (ein >> edge)
    {
        edges.push_back(edge);
    }

    std::ifstream vin("vertices.txt");
    int vertex;
    while (vin >> vertex)
    {
        vertices.push_back(vertex);
    }

    double *h_shortestDistancesGPU = (double *)malloc(
        vertices.size() * sizeof(double)
    );
    
    PerVertexSSP(
        weights, 
        vertices, 
        edges,
        sourceVertex,
        h_shortestDistancesGPU,
        asyncIter,
        blockSize
    );
    for (int i = 0; i < vertices.size(); i++) {
        std::cout << h_shortestDistancesGPU[i] << std::endl;
    }
    free(h_shortestDistancesGPU);
    return 0;
}



































