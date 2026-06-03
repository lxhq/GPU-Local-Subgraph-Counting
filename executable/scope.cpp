//
// Created by anonymous author on 2022/8/30.
//

#include "command.h"
#include "execution.h"
#include "Pexecution.h"
#include <cstdint>
#include <malloc.h>

int main(int argc, char **argv) {
    Command cmd(argc, argv);
    std::string queryGraphPath = cmd.getQueryGraphPath();
    std::string dataGraphPath = cmd.getDataGraphPath();
    std::string resultPath = cmd.getResultPath();
    std::string trianglePath = cmd.getTrianglePath();
    bool batchQuery = cmd.getBatchQuery();
    bool shareNode = cmd.getShareNode();
    bool matchOnly = cmd.getMatchOnly();
    bool profileReset = cmd.getProfileReset();
    bool useTriangle = !trianglePath.empty();
    std::string mode = cmd.getExecutionMode();
    if (mode != "single" && mode != "parallel") {
        std::cerr << "Invalid mode: " << mode << std::endl;
        exit(1);
    }
    if (matchOnly && mode != "parallel") {
        std::cerr << "-match-only is currently supported for -m parallel only." << std::endl;
        exit(1);
    }
    if (matchOnly && shareNode) {
        std::cerr << "-match-only is currently supported for non-share mode only." << std::endl;
        exit(1);
    }
    if (matchOnly && useTriangle) {
        std::cerr << "-match-only is currently supported without triangle acceleration only." << std::endl;
        exit(1);
    }
    if (profileReset && matchOnly) {
        std::cerr << "-profile-reset should be used with normal execution, not -match-only." << std::endl;
        exit(1);
    }
    if (profileReset && mode != "parallel") {
        std::cerr << "-profile-reset is currently supported for -m parallel only." << std::endl;
        exit(1);
    }
    if (profileReset && shareNode) {
        std::cerr << "-profile-reset is currently supported for non-share mode only." << std::endl;
        exit(1);
    }
    if (profileReset && useTriangle) {
        std::cerr << "-profile-reset is currently supported without triangle acceleration only." << std::endl;
        exit(1);
    }
    if (matchOnly) {
        resultPath.clear();
    }
    ui num_threads = cmd.getNumThreads();
    ui node_partition_size = cmd.getNodePartitionSize();
    ui prefix_partition_size = cmd.getPrefixPartitionSize();
    // ui patterns_parallel_size = cmd.getPatternsParallelSize();

    std::cout << "query graph path: " << queryGraphPath << std::endl;
    std::cout << "data graph path: " << dataGraphPath << std::endl;
    std::cout << "result path: " << resultPath << std::endl;
    std::cout << "using batch query: " << batchQuery << std::endl;
    std::cout << "sharing nodes computation: " << shareNode << std::endl;
    std::cout << "using triangle: " << useTriangle << std::endl;
    std::cout << "matching-only dry run: " << matchOnly << std::endl;
    std::cout << "profile reset time: " << profileReset << std::endl;
    std::cout << "set intersection type: " << SI << std::endl;
    std::cout << "mode: " << mode << std::endl;
    if (mode == "parallel") {
        std::cout << "number of threads: " << num_threads << std::endl;
        std::cout << "node partition size: " << node_partition_size << std::endl;
        std::cout << "prefix partition size: " << prefix_partition_size << std::endl;
        // std::cout << "patterns parallel size: " << patterns_parallel_size << std::endl;
    }
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
    // candPos is used for paratition prefix
    // startoffset is used for node enumeration
    ui *candPos = new ui[MAX_PATTERN_SIZE];
    memset(candPos, 0, sizeof(ui) * (MAX_PATTERN_SIZE));
    // patternV, dataV, startOffset are designed that for each tree node and for each pattern vertex
    // patternV is the current matching query vertex id
    // dataV is the current matched data vertex id (embedding_)
    // startOffset records the start position of the candidate in the data graph's offset array
    // for executedTree, only patternV[0] and dataV[0] are used
    VertexID **patternV = new VertexID *[MAX_NUM_NODE];
    VertexID **dataV = new VertexID *[MAX_NUM_NODE];
    EdgeID **startOffset = new EdgeID *[MAX_NUM_NODE];
    //visited is for each tree node and for each data vertex
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
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);
    ParallelProcessingMeta *pMeta = nullptr;
    if (mode == "parallel") {
        pMeta = new ParallelProcessingMeta(num_threads, node_partition_size, prefix_partition_size, din, dout, dun);
        pMeta->setResetProfile(profileReset);
        setResetProfileMeta(pMeta);
    }
    auto resetTableForProfile = [&](HashTable table, ui count) {
        if (profileReset && pMeta != nullptr) {
            auto resetStart = std::chrono::steady_clock::now();
            memset(table, 0, sizeof(Count) * count);
            auto resetEnd = std::chrono::steady_clock::now();
            std::chrono::duration<double> resetElapsed = resetEnd - resetStart;
            pMeta->addResetProfileSample(-1, resetElapsed.count(), false, count, sizeof(Count) * (uint64_t)count);
        }
        else {
            memset(table, 0, sizeof(Count) * count);
        }
    };
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
                genEquation(patternGraphs[i], patterns, trees, cn, useTriangle, true, true, true);
            // }
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            totalPlanTime += elapsedSeconds.count();
            if (profileReset && pMeta != nullptr) pMeta->resetResetProfile();
            start = std::chrono::steady_clock::now();
            // Count* H. The LSC result is stored in H
            HashTable H;
            int orbitType = patternGraphs[i].getOrbitType();
            orbitTypes[i] = orbitType;
            if (orbitType == 0) {
                H = new Count[1];
                if (!matchOnly) H[0] = 0;
            }
            else if (orbitType == 1) {
                H = new Count[dun.getNumVertices()];
                if (!matchOnly) memset(H, 0, sizeof(Count) * n);
            }
            else {
                H = new Count[m];
                if (!matchOnly) memset(H, 0, sizeof(Count) * m);
            }
            if (cn.num != 0) {
                if (matchOnly) {
                    std::cerr << "-match-only does not yet support executeConNode queries." << std::endl;
                    exit(1);
                }
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
                    if (!matchOnly)
                        memset(factorSum, 0, sizeof(Count) * (m + 1));
                    for (int j = 0; j < it->second.size(); ++j) {
                        for (int j2 = 0; j2 < trees[divideFactor][j].size(); ++j2) {
                            if (!matchOnly) {
                                for (int l = 0; l < trees[divideFactor][j][0].getNumNodes(); ++l) {
                                    resetTableForProfile(ht[l], m);
                                }
                                if (mode == "parallel") {
                                    for (int i = 0; i < pMeta->_num_threads; i++) {
                                        for (int l = 0; l < trees[divideFactor][j][0].getNumNodes(); ++l) {
                                            resetTableForProfile(pMeta->_total_hash_table[i][l], m);
                                        }
                                    }
                                }
                            }
                            int k = patterns[divideFactor][j].u.getNumVertices();
                            if (patterns[divideFactor][j].u.isClique() && k >= 20) {
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
                                if (t.getExecuteMode()) {
                                    if (mode == "single") {
                                        executeTree(t, din, dout, dun, useTriangle, triangle, patterns[divideFactor][j],
                                                ht, outID, unID, reverseID, startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV);
                                    } else {
                                       executeTree(t, din, dout, dun, useTriangle, triangle, patterns[divideFactor][j],
                                                ht, outID, unID, reverseID, startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV, pMeta, matchOnly);
                                    }
                                }
                                else {
                                    if (mode == "single") {
                                        multiJoinTree(t, din, dout, dun, useTriangle, triangle, patterns[divideFactor][j],
                                                    ht, outID, unID, reverseID, startOffset, patternV, dataV, visited, tmp, allV);
                                    } else {
                                        multiJoinTree(t, din, dout, dun, useTriangle, triangle, patterns[divideFactor][j],
                                                    ht, outID, unID, reverseID, startOffset, patternV, dataV, visited, tmp, allV, pMeta, matchOnly);
                                    }
                                }
                            }
                            // ht is the cout for each node. here h is the count for the root node
                            HashTable h = ht[trees[divideFactor][j][j2].getRootID()];
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
            if (matchOnly)
                std::cout << "match-only execution time: " << exeTime << "s";
            else
                std::cout << "execution time: " << exeTime << "s";
            if (profileReset && pMeta != nullptr) {
                std::cout << ", reset wall time estimate: " << pMeta->getResetWallTimeEstimate() << "s"
                          << ", reset serial time: " << pMeta->getResetSerialTime() << "s"
                          << ", reset thread max time: " << pMeta->getResetThreadMaxTime() << "s"
                          << ", reset thread sum time: " << pMeta->getResetThreadSumTime() << "s"
                          << ", reset serial calls: " << pMeta->getResetSerialCalls()
                          << ", reset sparse calls: " << pMeta->getResetSparseCalls()
                          << ", reset full calls: " << pMeta->getResetFullCalls()
                          << ", reset sparse entries: " << pMeta->getResetSparseEntries()
                          << ", reset full bytes: " << pMeta->getResetFullBytes();
            }
            ui numPatterns = 0;
            for (auto it = patterns.begin(); it != patterns.end(); ++it)
                numPatterns += it->second.size();
            std::cout << ", number of patterns: " << numPatterns << std::endl;
            totalExeTime += exeTime;
            totalNumPatterns += numPatterns;
            mathCalH[i] = H;
            malloc_trim(0);
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

    delete[] reverseID;
    delete[] outID;
    delete[] unID;
    delete[] sg;
    delete[] cliqueVertices;
    delete[] tmp;
    delete[] allV;
    for (auto & h: ht)
        delete[] h;
    delete [] factorSum;
    for (int i = 0; i < MAX_NUM_NODE; ++i) {
        delete[] patternV[i];
        delete[] dataV[i];
        delete[] startOffset[i];
        delete[] visited[i];
    }
    delete[] patternV;
    delete[] dataV;
    delete[] startOffset;
    delete[] visited;
    delete[] candPos;

    return 0;
}
