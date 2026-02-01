//
// Created by anonymous author on 2022/8/30.
//

#include "command.h"
#include "execution.h"
#include "execution_gpu.h"

void printKernelFunctionInformation(const void* kernelFunc) {
    int numBlocksPerSM = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        kernelFunc,
        BLOCK_DIM,
        0
    );
    if (status != cudaSuccess) {
        std::cout << "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed: %s\n", cudaGetErrorString(status);
        exit(1);
    }
    int maxActiveThreadsPerSM = numBlocksPerSM * BLOCK_DIM;
    int maxActiveWarpsPerSM   = (maxActiveThreadsPerSM + 31) / 32;  // round up

    cudaFuncAttributes attr;
    status = cudaFuncGetAttributes(&attr, kernelFunc);
    if (status != cudaSuccess) {
        std::cout << "cudaFuncGetAttributes failed: %s\n", cudaGetErrorString(status);
        exit(1);
    }
    std::cout << "Kernel uses " << attr.numRegs << " registers per thread." << std::endl;
    std::cout << "Kernel uses " << TO_KB(attr.sharedSizeBytes) << " KB of static shared memory per block." << std::endl;
    std::cout << "Max active blocks per SM: " << numBlocksPerSM << std::endl;
    std::cout << "Max active threads per SM: " << maxActiveThreadsPerSM << std::endl;
    std::cout << "Max active warps per SM: " << maxActiveWarpsPerSM << std::endl;
}

int main(int argc, char **argv) {
    Command cmd(argc, argv);
    std::string queryGraphPath = cmd.getQueryGraphPath();
    std::string dataGraphPath = cmd.getDataGraphPath();
    std::string resultPath = cmd.getResultPath();
    std::string trianglePath = cmd.getTrianglePath();
    bool batchQuery = cmd.getBatchQuery();
    bool shareNode = cmd.getShareNode();
    bool useTriangle = !trianglePath.empty();
    float ratio = cmd.getRatio();
    uint32_t prob_limit = cmd.getProbLimit();
    uint32_t memory_pool_size = cmd.getMemoryPoolSize();
    std::cout << "query graph path: " << queryGraphPath << std::endl;
    std::cout << "data graph path: " << dataGraphPath << std::endl;
    std::cout << "result path: " << resultPath << std::endl;
    std::cout << "using batch query: " << batchQuery << std::endl;
    std::cout << "sharing nodes computation: " << shareNode << std::endl;
    std::cout << "using triangle: " << useTriangle << std::endl;
    std::cout << "set intersection type: " << SI << std::endl;
    std::cout << "max memory pool size: " << memory_pool_size << " GB" << std::endl;
    std::cout << "subgraph matching hashtable ratio: " << ratio << std::endl;
    std::cout << "prob limit: " << prob_limit << std::endl;
    std::cout << "hash table type: " << HASH_TABLE_TYPE << std::endl;
    DataGraph dun = DataGraph();
    dun.loadDataGraph(dataGraphPath);
    const DataGraph din = constructDirectedDataGraph(dun, false);
    const DataGraph dout = constructDirectedDataGraph(dun, true);
    specialsparse *sg = (specialsparse *)malloc(sizeof(specialsparse));
    dout.initSpecialSparse(sg);
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    Triangle triangle;
    if (useTriangle) {
        triangle.load(trianglePath, m / 2);
    }
    auto start = std::chrono::steady_clock::now();
    EdgeID *outID = buildInID2OutID(din, dout);
    EdgeID *unID = buildUnID2OutID(dun, dout);
    EdgeID *reverseID = buildReverseUnID(dun);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "finished building edge ID mappings. time: " << elapsedSeconds.count() << "s" << std::endl;
    std::vector<std::string> files;
    std::vector<PatternGraph> patternGraphs = loadPatternGraph(queryGraphPath, batchQuery, files);
    int patternSize = patternGraphs[0].getNumVertices();
    bool forestShare = patternSize <= 5;
    // what is factorSum?
    // in the computation, we have a list of patterns (trees for each pattern accordingly, and each tree owns its factor \mu)
    // we are going to compute the tiso-count of each tree one by one
    // the result count with the factor \mu is stored in the factorSum accumulated
    // if orbit is vertex, factorSum stores the local subgraph counting for each vertex
    // and then when the next pattern is processed, the values in the factorSum will be updated
    HashTable factorSum = new Count[m + 1];
    
    // what is ht? Why it is initialized with the MAX_NUM_NODE
    // Count* ht[MAX_NUM_NODE] == Count* ht[10]
    // ht is a matrix with 10 rows and m+1 columns
    // ht is designed for a single tree, each row is the count for a single node

    // when a tree node's key is an edge, h stores count for edges
    // that is why we create h with the number of edges instead of the number of vertices
    HashTable ht[MAX_NUM_NODE];
    for (auto & h: ht)
        h = new Count[m + 1];
    // candPos is for each pattern vertex
    // elements in candPos is a pointer points to the current matched data vertex candidate in the candidate list
    // candPos[mappingSize] points to a data vertex in the candidate[nodeID][mappingSize] array to be matched
    ui *candPos = new ui[MAX_PATTERN_SIZE];
    memset(candPos, 0, sizeof(ui) * (MAX_PATTERN_SIZE));
    // patternV, dataV, startOffset are designed that for each tree node and for each pattern vertex
    // patternV is the current matching query vertex id
    // dataV is the current matched data vertex id (embedding_)
    // startOffset records the start position of the candidate in the data graph's offset array, used for computing the edge id for edge orbit case
    // visited is for each tree node and for each data vertex
    // for executedTree, only patternV[0], dataV[0], startOffset[0] and visited[0] are used
    VertexID **patternV = new VertexID *[MAX_NUM_NODE];
    VertexID **dataV = new VertexID *[MAX_NUM_NODE];
    EdgeID **startOffset = new EdgeID *[MAX_NUM_NODE];
    bool **visited = new bool *[MAX_NUM_NODE];
    for (int i = 0; i < MAX_NUM_NODE; ++i) {
        patternV[i] = new VertexID[MAX_PATTERN_SIZE];
        dataV[i] = new VertexID[MAX_PATTERN_SIZE];
        startOffset[i] = new VertexID[MAX_PATTERN_SIZE];
        visited[i] = new bool[n];
        memset(visited[i], false, sizeof(bool) * n);
    }
    // tmp and allV are for each data graph vertex
    // the tmp array is for holding the intersection results temproarly. Its valus will be assigned to the candidate array later
    VertexID *tmp = new VertexID[n];
    // I guess allV is for the candidate list for the first vertex (need to verify)
    VertexID *allV = new VertexID[n];
    for (VertexID i = 0; i < n; ++i) {
        allV[i] = i;
    }
    VertexID *cliqueVertices = new VertexID[MAX_PATTERN_SIZE + 1];
    std::set<CanonType> visitedNode;

    double totalPlanTime = 0.0, totalExeTime = 0.0;
    ui totalNumPatterns = 0, totalNodes = 0;
    double averageNodeSize = 0.0;
    int numVertexTable = 0, numEdgeTable = 0;

    /************************** Initlize GPU  ******************************/
    /****************** print device information *************/
    uint32_t gpu_num = 0u;
    cudaSetDevice(gpu_num);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_num);
    std::cout << std::endl;
    std::cout << "Device ID: " << gpu_num << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "L2 Cache Size: " << TO_MB(prop.l2CacheSize) << " MB" << std::endl;
    std::cout << "Total Global Memory: " << TO_GB(prop.totalGlobalMem) << " GB" << std::endl;
    std::cout << "Registers Per SM: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "Shared Memory Per SM: " << TO_KB(prop.sharedMemPerMultiprocessor) << " KB" << std::endl;
    std::cout << "Max blocks Per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Max threads Per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate << " KHz" << std::endl;
    std::cout << "Memory Bandwidth: " << (prop.memoryClockRate * (uint64_t)prop.memoryBusWidth * 1000 * 2) / 8e9 << " GB/s" << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Constant Memory: " << TO_KB(prop.totalConstMem) << " KB" << std::endl;
    std::cout << "Max Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << "Shared Memory Per Block: " << TO_KB(prop.sharedMemPerBlock) << " KB" << std::endl;
    std::cout << "Registers Per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Max threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Size of NodeGPU: " << sizeof(NodeGPU) << std::endl;
    std::cout << std::endl;
    if (prop.warpSize != WARP_SIZE) {
        throw std::runtime_error("warp size is not 32");
    }

    /****************** print kernel functions information *************/
    std::cout << "countKernel" << std::endl;
    printKernelFunctionInformation(reinterpret_cast<const void*>(countKernel));
    std::cout << std::endl;

    std::cout << "writeKernel" << std::endl;
    printKernelFunctionInformation(reinterpret_cast<const void*>(writeKernel));
    std::cout << std::endl;

    std::cout << "finalLevelWithLocalCacheKernel" << std::endl;
    printKernelFunctionInformation(reinterpret_cast<const void*>(finalLevelWithLocalCacheKernel));
    std::cout << std::endl;

    std::cout << "finalLevelWithEdgeWithLocalCacheKernel" << std::endl;
    printKernelFunctionInformation(reinterpret_cast<const void*>(finalLevelWithEdgeWithLocalCacheKernel));
    std::cout << std::endl;

    std::cout << "writeToHashTableKernel" << std::endl;
    printKernelFunctionInformation(reinterpret_cast<const void*>(writeToHashTableKernel));
    std::cout << std::endl;

    std::cout << "writeToHashTableWithEdgeKeyKernel" << std::endl;
    printKernelFunctionInformation(reinterpret_cast<const void*>(writeToHashTableWithEdgeKeyKernel));
    std::cout << std::endl;

    std::cout << std::endl;
    /****************** allocate the device memory ***********/
    if (memory_pool_size == 0) {
        memory_pool_size = TO_GB(prop.totalGlobalMem) * 0.9;
        std::cout << "Memory pool (-mem) not set. Allocating 90% of device memory: " 
              << memory_pool_size << " GB (Total: " << TO_GB(prop.totalGlobalMem) << " GB)" << std::endl;    
    } else {
        if (memory_pool_size > TO_GB(prop.totalGlobalMem)) {
            throw std::runtime_error("not enough GPU memory, preset: " + std::to_string(memory_pool_size) + ", actual: " + std::to_string(TO_GB(prop.totalGlobalMem)));
        }
    }

    MEM_INIT();
    PRINT_MEM_INFO("Before Allocation");
    MemoryManager memory_manager(static_cast<uint64_t>(memory_pool_size) * 1024 * 1024 * 1024);
    PRINT_MEM_INFO("After allocation");
    std::cout << std::endl;

    // copy data graphs to device memory
    // copy outID, inID, reverseID to device memory
    copyMetaToGPU(din, dout, dun, outID, unID, reverseID, memory_manager);

    memory_manager.printMemoryUsage();
    /************************** End of initlize GPU ************************/
    
    if (!shareNode) {
        std::vector<HashTable> mathCalH(patternGraphs.size());
        std::vector<int> orbitTypes(patternGraphs.size());
#ifdef DEBUG
        HashTable hPattern = new Count[n];
        memset(hPattern, 0, sizeof(Count) * dun.getNumVertices());
#endif
        for (int i = 0; i < patternGraphs.size(); ++i) {
            double exeTime = 0.0;
            std::map<int, std::vector<Pattern>> patterns;
            std::map<int, std::vector<std::vector<Tree>>> trees;
            start = std::chrono::steady_clock::now();
            ConNode cn;
            // if (!patternGraphs[i].isClique()) {
                // set useDirected to false to make things easier
                // I cannot understand the decomposed tree produced with useDirected = true
                genEquation(patternGraphs[i], patterns, trees, cn, useTriangle, true, true, true);
            // }
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            totalPlanTime += elapsedSeconds.count();
            start = std::chrono::steady_clock::now();
            // Count* H. The LSC result is stored in H
            HashTable H;
            int orbitType = patternGraphs[i].getOrbitType();
            orbitTypes[i] = orbitType;
            if (orbitType == 0) {
                H = new Count[1];
                H[0] = 0;
            }
            else if (orbitType == 1) {
                H = new Count[dun.getNumVertices()];
                memset(H, 0, sizeof(Count) * n);
            }
            else {
                H = new Count[m];
                memset(H, 0, sizeof(Count) * m);
            }
            if (cn.num != 0) {
                Pattern p(patternGraphs[i]);
                // prepare hash tables for cn
                if (orbitType == 1) {
                    for (int j = 0; j < cn.aggrePos.size(); ++j)
                        cn.hashTables[j] = H;
                }
                else {
                    for (int j = 0; j < cn.aggrePos.size() / 2; ++j)
                        cn.hashTables[j] = H;
                }
                executeConNode(cn, din, dout, dun, useTriangle, triangle, p, outID, unID, startOffset[0], patternV[0],
                               dataV[0], visited[0], candPos, tmp, allV);
                if (!resultPath.empty()) {
                    int divideFactor = cn.divideFactor;
                    if (orbitType == 0) H[0] /= divideFactor;
                    else if (orbitType == 1)
                        for (VertexID l = 0; l < n; ++l)
                            H[l] /= divideFactor;
                    else {
                        for (EdgeID l = 0; l < m ; ++l)
                            H[l] /= divideFactor;
                    }
                }
            // }
            // else if (patternGraphs[i].isClique()) {
            //     start = std::chrono::steady_clock::now();
            //     int k = patternGraphs[i].getNumVertices();
            //     mkspecial(sg, k);
            //     kclique(k, k, sg, cliqueVertices, H, orbitType);
            //     freesub(sg, k);
            //     end = std::chrono::steady_clock::now();
            //     elapsedSeconds = end - start;
            //     totalExeTime += elapsedSeconds.count();
            } else {
                for (auto it = patterns.begin(); it != patterns.end(); ++it) {
                    int divideFactor = it->first;
                    memset(factorSum, 0, sizeof(Count) * (m + 1));
                    for (int j = 0; j < it->second.size(); ++j) {
                        for (int j2 = 0; j2 < trees[divideFactor][j].size(); ++j2) {
                            for (int l = 0; l < trees[divideFactor][j][0].getNumNodes(); ++l) {
                                memset(ht[l], 0, sizeof(Count) * m);
                            }
                            int k = patterns[divideFactor][j].u.getNumVertices();
// TODO: 
                            if (patterns[divideFactor][j].u.isClique() && k >= 10) {
                                HashTable h = ht[trees[divideFactor][j][j2].getRootID()];
                                int aggreWeight = trees[divideFactor][j][j2].getAggreWeight()[0];
                                mkspecial(sg, k);
                                kclique(k, k, sg, cliqueVertices, h, orbitType);
                                freesub(sg, k);
                                if (aggreWeight != 1) {
                                    if (orbitType == 0) h[0] *= aggreWeight;
                                    else if (orbitType == 1) {
                                        for (VertexID v = 0; v < n; ++v)
                                            h[v] *= aggreWeight;
                                    }
                                    else {
                                        for (EdgeID e = 0; e < m; ++e)
                                            h[e] *= aggreWeight;
                                    }
                                }
                            }
                            else {
                                const Tree &t = trees[divideFactor][j][j2];
/********** Debug *******/
// patterns[divideFactor][j].u.printGraph();
// t.print();
// bool hasEdgeKey = false;
// for (int er = 0; er < t.getNumNodes(); ++er) {
//     if (t.nodeEdgeKey(er)) {
//         hasEdgeKey = true;
//         break;
//     }
// }
// if (hasEdgeKey) {
//     std::cout << "File " << files[i] << " has edge key" << std::endl;
// }
// continue;
/********** Debug *******/
                                if (t.getExecuteMode()) {
                                    // executeTree(t, din, dout, dun, useTriangle, triangle, patterns[divideFactor][j],
                                    //             ht, outID, unID, reverseID, startOffset[0], patternV[0], dataV[0],
                                    //             visited[0], candPos, tmp, allV);
                                    // auto start = std::chrono::steady_clock::now();
                                    executeTreeGPU(t, dun, ht, memory_manager, prob_limit, ratio);
                                    // auto end = std::chrono::steady_clock::now();
                                    // std::chrono::duration<double> elapsedSeconds = end - start;
                                    // std::cout << "execution time: " << elapsedSeconds.count() << "s" << std::endl << std::endl;
                                }
                                else {
                                    // multiJoinTree(t, din, dout, dun, useTriangle, triangle, patterns[divideFactor][j],
                                    //             ht, outID, unID, reverseID, startOffset, patternV, dataV, visited, tmp, allV);
                                    multiJoinTreeGPU(t, dun, ht, memory_manager, prob_limit, ratio);
                                }
                            }
                            // ht is the cout for each node. here h is the count for the root node
                            HashTable h = ht[trees[divideFactor][j][j2].getRootID()];
/********** Debug *******/
// print out the content of the hash table
// int64_t sum = 0;
// for (VertexID l = 1; l < n; ++l) {
//     // std::cout << h[l] << " ";
//     sum += h[l];
// }
// std::cout<<"sum: "<<sum<<std::endl;
// std::cout<<std::endl;
// exit(0);
/********** Debug *******/
                            int multiFactor = trees[divideFactor][j][j2].getMultiFactor();
                            if (!resultPath.empty()) {
                                if (orbitType == 0) factorSum[0] += h[0];
                                else if (orbitType == 1)
                                    for (VertexID l = 0; l < n; ++l) {
                                        factorSum[l] += h[l] * multiFactor;
                                    }
                                else
                                    for (EdgeID l = 0; l < m + 1; ++l) {
                                        factorSum[l] += h[l] * multiFactor;
                                    }
                            }
    #ifdef DEBUG
                            if (divideFactor == 2 && patterns[divideFactor][j].u.getCanonValue() == 281875) {
                            for (VertexID l = 0; l < n; ++l) {
                                hPattern[l] += h[l];
                            }
                        }
    #endif
                            for (VertexID nID = 0; nID < trees[divideFactor][j][j2].getNumNodes(); ++nID) {
                                ++totalNodes;
                                averageNodeSize += trees[divideFactor][j][j2].getNode(nID).numVertices;
                                visitedNode.insert(trees[divideFactor][j][j2].getNode(nID).canonValue);
                                if (nID != trees[divideFactor][j][j2].getPostOrder().back()) {
                                    if (trees[divideFactor][j][j2].getNode(nID).keySize == 1)
                                        ++numVertexTable;
                                    else
                                        ++numEdgeTable;
                                }
                            }
                        }
                    }
                    if (!resultPath.empty()) {
                        if (orbitType == 0) H[0] += factorSum[0] / divideFactor;
                        else if (orbitType == 1)
                            for (VertexID l = 0; l < n; ++l) {
                                H[l] += factorSum[l] / divideFactor;
                            }
                        else
                            for (EdgeID l = 0; l < m + 1; ++l) {
                                H[l] += factorSum[l] / divideFactor;
                            }
                    }
                }
            }
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            exeTime = elapsedSeconds.count();
            if (batchQuery)
                std::cout << "file: " << files[i] << ", ";
            std::cout << "execution time: " << exeTime << "s";
            ui numPatterns = 0;
            for (auto it = patterns.begin(); it != patterns.end(); ++it)
                numPatterns += it->second.size();
            std::cout << ", number of patterns: " << numPatterns;
            std::cout << ", current average batch size: " << global_running_avg << std::endl;
            totalExeTime += exeTime;
            totalNumPatterns += numPatterns;
            mathCalH[i] = H;
        }
        if (!resultPath.empty()) saveCount(resultPath, mathCalH, dun, batchQuery, files, orbitTypes);
        for (int i = 0; i < patternGraphs.size(); ++i)
            delete[] mathCalH[i];
        std::cout << "compute equation time: " << gEquationTime << std::endl;
        std::cout << "compute tree decomposition: " << gTDTime << std::endl;
        std::cout << "compute symmetry time: " << gSymmTime << std::endl;
        std::cout << "compute attribute order: " << gOrderTime << std::endl;
        std::cout << "planning time: " << totalPlanTime << ", total execution time: "
                  << totalExeTime << ", total number of patterns: " << totalNumPatterns << "\ntotal number of nodes: "
                  << totalNodes << ", average size of nodes: "<< averageNodeSize / totalNodes
                  << ", number of distinct nodes: " << visitedNode.size() << std::endl
                  << "number of vertex hash tables (except for root node): " << numVertexTable
                  << ", number of edge hash tables (except for root node): " << numEdgeTable << std::endl;
        std::cout << "number of match: " << gNumMatch << ", number of intersect: " << gNumIntersect << std::endl;
        std::cout << "total average batch size: " << global_running_avg << std::endl;
    }

    if (shareNode) {
        std::vector<ConNode> conNodes;
        std::vector<Pattern> conPatterns;
        std::vector<int> conFactors;
        std::vector<int> conIDs;
        Forest fu, fuCover;
        std::vector<Forest> directedF;
        std::vector<CanonType> coverCanon;
        std::vector<int> coverOrbit;
        std::vector<int> coverRootOrbit;
        std::vector<HashTable> result(patternGraphs.size());
        std::vector<HashTable> coverH;
        std::vector<HashTable> uH;
        std::vector<std::vector<HashTable>> dH;
        // for the ith Pattern, jth cover, its position in the coverH and the factor
        std::vector<std::vector<int>> coverPos(patternGraphs.size());
        // for the ith pattern, position j, the multiply factor of that cover
        std::vector<std::vector<int>> coverMultiFactors(patternGraphs.size());
        std::vector<int> divideFactors(patternGraphs.size());
        std::vector<bool> isUndirected(patternGraphs.size(), false);
        std::vector<HashTable> allocatedHashTable;
        allocatedHashTable.reserve(10000);
        int nTable = 0, mTable = 0;
        for (int i = 0; i < patternGraphs.size(); ++i) {
            const PatternGraph &pg = patternGraphs[i];
            int orbitType = pg.getOrbitType();
            if (orbitType == 0) {
                result[i] = new Count[1];
                result[i][0] = 0;
            }
            else if (orbitType == 1) {
                result[i] = new Count[n];
                memset(result[i], 0, sizeof(Count) * n);
                ++nTable;
            }
            else {
                if (pg.isEOrbitDir()) {
                    result[i] = new Count[m / 2];
                    memset(result[i], 0, sizeof(Count) * m / 2);
                    ++mTable;
                }
                else {
                    result[i] = new Count[m];
                    memset(result[i], 0, sizeof(Count) * m);
                    mTable += 2;
                }
            }
        }
        for (int i = 0; i < patternGraphs.size(); ++i) {
            const PatternGraph &pg = patternGraphs[i];
            int orbitType = pg.getOrbitType();
            start = std::chrono::steady_clock::now();
            if (patternGraphs[i].isClique()) {
                start = std::chrono::steady_clock::now();
                int k = patternGraphs[i].getNumVertices();
                mkspecial(sg, k);
                kclique(k, k, sg, cliqueVertices, result[i], orbitType);
                freesub(sg, k);
                end = std::chrono::steady_clock::now();
                elapsedSeconds = end - start;
                totalExeTime += elapsedSeconds.count();
            }
            else {
                std::map<int, std::vector<Pattern>> patterns;
                std::map<int, std::vector<std::vector<Tree>>> trees;
                ConNode cn;
                bool directed = genEquation(pg, patterns, trees, cn, useTriangle, true, true, true);
                if (cn.num != 0) {
                    if (orbitType == 0) cn.hashTables[0] = result[i];
                    else if (orbitType == 1) {
                        for (int j = 0; j < cn.aggrePos.size(); ++j)
                            cn.hashTables[j] = result[i];
                    }
                    else {
                        for (int j = 0; j < cn.aggrePos.size() / 2; ++j)
                            cn.hashTables[j] = result[i];
                    }
                    bool exists = false;
                    for (int j = 0; j < conNodes.size(); ++j) {
                        if (conNodes[j].num == cn.num && conNodes[j].canonValue == cn.canonValue && conNodes[j].edgeKey == cn.edgeKey) {
                            conNodes[j].merge(cn, conPatterns[j].u, patternGraphs[i]);
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) {
                        conNodes.push_back(cn);
                        conPatterns.emplace_back(pg);
                    }
                    conIDs.push_back(i);
                    conFactors.push_back(cn.divideFactor);
                }
                else if (directed) {
                    std::cout << files[i] << " uses DAG" << std::endl;
                    if (pg.getNumVertices() >= 6 || directedF.empty()) {
                        Forest fd;
                        std::vector<HashTable> h;
                        h.push_back(result[i]);
                        dH.push_back(h);
                        fd.loadQuery(patterns, trees, true);
                        directedF.push_back(fd);
                    }
                    else {
                        directedF[0].loadQuery(patterns, trees, true);
                        dH[0].push_back(result[i]);
                    }
                }
                else {
                    isUndirected[i] = true;
                    // put the first pattern to fu, the remaining patterns to fcover
                    int divideFactor = patterns.begin() -> first;
                    divideFactors[i] = divideFactor;
                    const std::vector<Pattern> &allPattern = patterns.begin()->second;
                    std::vector<std::vector<Tree>> &allTree = trees.begin()->second;
                    std::map<int, std::vector<Pattern>> rootPattern;
                    std::map<int, std::vector<std::vector<Tree>>> rootTree;
                    rootPattern[divideFactor].push_back(allPattern[0]);
                    rootTree[divideFactor].push_back(allTree[0]);
                    coverPos[i] = std::vector<int>(allPattern.size() - 1);
                    coverMultiFactors[i] = std::vector<int>(allPattern.size() - 1 + coverCanon.size());
                    for (int j = 1; j < allPattern.size(); ++j) {
                        bool exists = false;
                        int rootOrbit = allTree[j][0].getNode(allTree[j][0].getRootID()).v2o[0];
                        for (int k = 0; k < coverCanon.size(); ++k) {
                            if (allPattern[j].u.getCanonValue() == coverCanon[k] && allPattern[j].u.getOrbit(0) == coverOrbit[k]
                                && rootOrbit == coverRootOrbit[k]) {
                                coverPos[i][j - 1] = k;
                                coverMultiFactors[i][k] = allTree[j][0].getMultiFactor();
                                exists = true;
                                break;
                            }
                        }
                        if (!exists) {
                            HashTable h;
                            if (orbitType == 0) {
                                h = new Count[1];
                                h[0] = 0;
                            }
                            else if (orbitType == 1) {
                                h = new Count[n];
                                memset(h, 0, sizeof(Count) * n);
                                ++nTable;
                            }
                            else {
                                if (pg.isEOrbitDir()) {
                                    h = new Count[m / 2];
                                    memset(h, 0, sizeof(Count) * m / 2);
                                    ++mTable;
                                }
                                else {
                                    h = new Count[m];
                                    memset(h, 0, sizeof(Count) * m);
                                    mTable += 2;
                                }
                            }
                            coverPos[i][j - 1] = coverCanon.size();
                            coverMultiFactors[i][coverCanon.size()] = allTree[j][0].getMultiFactor();
                            allTree[j][0].setMultiFactor(1);
                            coverCanon.push_back(allPattern[j].u.getCanonValue());
                            coverOrbit.push_back(allPattern[j].u.getOrbit(0));
                            coverRootOrbit.push_back(rootOrbit);
                            std::map<int, std::vector<Pattern>> coverPattern;
                            std::map<int, std::vector<std::vector<Tree>>> coverTree;
                            coverPattern[1].push_back(allPattern[j]);
                            coverTree[1].push_back(allTree[j]);
                            fuCover.loadQuery(coverPattern, coverTree, forestShare);
                            coverH.push_back(h);
                        }
                    }
                    uH.push_back(result[i]);
                    fu.loadQuery(rootPattern, rootTree, forestShare);
                }
            }
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            totalPlanTime += elapsedSeconds.count();
        }
        double tableSize = (double)(nTable * n + mTable * m / 2) * sizeof(Count) / 1e9;
#ifndef ONLY_PLAN
        // 1. execute ConNodes
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < conNodes.size(); ++i) {
            executeConNode(conNodes[i], din, dout, dun, useTriangle, triangle, conPatterns[i], outID, unID,
                           startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV);
        }
        for (int i = 0; i < conIDs.size(); ++i) {
            int divideFactor = conFactors[i];
            int orbitType = patternGraphs[i].getOrbitType();
            bool eOrbitDir = patternGraphs[i].isEOrbitDir();
            int id = conIDs[i];
            if (orbitType == 0) result[id][0] /= divideFactor;
            else if (orbitType == 1) {
                for (int k = 0; k < n; ++k)
                    result[id][k] /= divideFactor;
            }
            else {
                if (eOrbitDir) {
                    for (int k = 0; k < m / 2; ++k)
                        result[id][k] /= divideFactor;
                }
                else {
                    for (int k = 0; k < m; ++k)
                        result[id][k] /= divideFactor;
                }
            }
        }
        end = std::chrono::steady_clock::now();
        elapsedSeconds = end - start;
        totalExeTime += elapsedSeconds.count();
        std::cout << "finished compressed node. time: " << totalExeTime << "s" << std::endl;
        // 2. execute undirected coverCanon, undirected root patterns and directed patterns
        if (patternSize <= 5) {
            start = std::chrono::steady_clock::now();
            executeForest(fuCover, coverH, din, dout, dun, useTriangle, triangle, allocatedHashTable,
                          outID, unID, reverseID, startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV, sg, cliqueVertices);
            for (HashTable h: allocatedHashTable) delete[] h;
            allocatedHashTable.clear();
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            totalExeTime += elapsedSeconds.count();
            std::cout << "finished undirected covers. time: " << totalExeTime << std::endl;
            start = std::chrono::steady_clock::now();
            executeForest(fu, uH,  din, dout, dun, useTriangle, triangle, allocatedHashTable,
                          outID, unID, reverseID, startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV, sg, cliqueVertices);
            for (HashTable h: allocatedHashTable) delete[] h;
            allocatedHashTable.clear();
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            totalExeTime += elapsedSeconds.count();
            std::cout << "finished undirected patterns. time: " << totalExeTime << std::endl;
        }
        else {
            start = std::chrono::steady_clock::now();
            for (int i = 0; i < fuCover.numQuery; ++i) {
                const Tree &t = fuCover.allTree[i];
                const Pattern &p = fuCover.allPattern[i];
                for (int l = 0; l < t.getNumNodes(); ++l) memset(ht[l], 0, sizeof(Count) * m);
                if (t.getExecuteMode()) {
                    executeTree(t, din, dout, dun, useTriangle, triangle, p, ht, outID, unID, reverseID,
                                startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV);
                }
                else {
                    multiJoinTree(t, din, dout, dun, useTriangle, triangle, p, ht, outID, unID, reverseID, startOffset,
                                  patternV, dataV, visited, tmp, allV);
                }
                HashTable h = ht[t.getRootID()];
                int orbitType = patternGraphs[i].getOrbitType();
                if (orbitType == 0) {
                    coverH[i][0] = h[0];
                }
                else if (orbitType == 1) {
                    for (int j = 0; j < n; ++j) {
                        coverH[i][j] = h[j];
                    }
                }
            }
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            totalExeTime += elapsedSeconds.count();
            std::cout << "finished undirected covers. time: " << totalExeTime << std::endl;
            start = std::chrono::steady_clock::now();
            for (int i = 0; i < fu.numQuery; ++i) {
                const Tree &t = fu.allTree[i];
                const Pattern &p = fu.allPattern[i];
                for (int l = 0; l < t.getNumNodes(); ++l) memset(ht[l], 0, sizeof(Count) * m);
                if (t.getExecuteMode()) {
                    executeTree(t, din, dout, dun, useTriangle, triangle, p, ht, outID, unID, reverseID,
                                startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV);
                }
                else {
                    multiJoinTree(t, din, dout, dun, useTriangle, triangle, p, ht, outID, unID, reverseID, startOffset,
                                  patternV, dataV, visited, tmp, allV);
                }
                HashTable h = ht[t.getRootID()];
                int multiFactor = t.getMultiFactor();
                int orbitType = patternGraphs[i].getOrbitType();
                if (orbitType == 0) {
                    uH[i][0] = h[0] * multiFactor;
                }
                else if (orbitType == 1) {
                    for (int j = 0; j < n; ++j) {
                        uH[i][j] = h[j] * multiFactor;
                    }
                }
            }
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            totalExeTime += elapsedSeconds.count();
            std::cout << "finished undirected patterns. time: " << totalExeTime << std::endl;
        }
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < directedF.size(); ++i) {
            executeForest(directedF[i], dH[i], din, dout, dun, useTriangle, triangle, allocatedHashTable,
                          outID, unID, reverseID, startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV, sg, cliqueVertices);
            for (HashTable h: allocatedHashTable) delete[] h;
            allocatedHashTable.clear();
        }
        end = std::chrono::steady_clock::now();
        elapsedSeconds = end - start;
        totalExeTime += elapsedSeconds.count();
        std::cout << "finished directed patterns. time : " << totalExeTime << std::endl;
        // 3. subtract cover count for undirected patterns
        for (int i = 0; i < patternGraphs.size(); ++i) {
            if (!isUndirected[i]) continue;
            const PatternGraph &pg = patternGraphs[i];
            int orbitType = pg.getOrbitType();
            int divideFactor = divideFactors[i];
            for (int j = 0; j < coverPos[i].size(); ++j) {
                HashTable h = coverH[coverPos[i][j]];
                if (orbitType == 0)
                    result[i][0] += h[0] * coverMultiFactors[i][coverPos[i][j]];
                else if (orbitType == 1) {
                    for (int k = 0; k < n; ++k)
                        result[i][k] += h[k] * coverMultiFactors[i][coverPos[i][j]];
                }
                else if (orbitType == 2 && pg.isEOrbitDir()) {
                    for (int k = 0; k < m / 2; ++k)
                        result[i][k] += h[k] * coverMultiFactors[i][coverPos[i][j]];
                }
                else {
                    for (int k = 0; k < m; ++k)
                        result[i][k] += h[k] * coverMultiFactors[i][coverPos[i][j]];
                }
            }
            if (orbitType == 0)
                result[i][0] /= divideFactor;
            else if (orbitType == 1) {
                for (int k = 0; k < n; ++k)
                    result[i][k] /= divideFactor;
            }
            else if (orbitType == 2 && pg.isEOrbitDir()) {
                for (int k = 0; k < m / 2; ++k)
                    result[i][k] /= divideFactor;
            }
            else {
                for (int k = 0; k < m; ++k)
                    result[i][k] /= divideFactor;
            }
        }
#endif
        std::cout << "compute equation time: " << gEquationTime << std::endl;
        std::cout << "compute tree decomposition: " << gTDTime << std::endl;
        std::cout << "compute symmetry time: " << gSymmTime << std::endl;
        std::cout << "compute order time: " << gOrderTime << std::endl;
        std::cout << "total planning time: " << totalPlanTime << ", total execution time: " << totalExeTime
                  << ", total time: " << totalPlanTime + totalExeTime << std::endl;
        std::cout << "number of match: " << gNumMatch << ", number of intersect: " << gNumIntersect  << std::endl;
        int orbitType = patternGraphs[0].getOrbitType();
        std::vector<int> orbitTypes(files.size(), orbitType);
        if (!resultPath.empty()) saveCount(resultPath, result, dun, batchQuery, files, orbitTypes);
    }

    return 0;
}