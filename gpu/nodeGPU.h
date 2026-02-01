#pragma once

#include <cstdint>
#include <config.h>
#include "globals.h"
#include "gpu_config.h"
#include "tree.h" 

struct NodeGPU {
    uint32_t pre_nbrs_count[MAX_PATTERN_SIZE];                            // number of predecessor neighbors at depth
    uint32_t pre_nbrs_pos[MAX_PATTERN_SIZE][MAX_PATTERN_SIZE];            // predecessor neighbors at depth
    uint32_t pre_nbr_graph_type[MAX_PATTERN_SIZE][MAX_PATTERN_SIZE];           // predecessor neighbors pointer at depth

    uint32_t nodeGreaterPos[MAX_PATTERN_SIZE][MAX_PATTERN_SIZE];
    uint32_t nodeGreaterPosCount[MAX_PATTERN_SIZE];
    uint32_t nodeLessPos[MAX_PATTERN_SIZE][MAX_PATTERN_SIZE];
    uint32_t nodeLessPosCount[MAX_PATTERN_SIZE];

    uint32_t childCount;
    uint32_t children[MAX_NUM_NODE];
    uint32_t childKeyPosCount[MAX_NUM_NODE];
    uint32_t childKeyPos[MAX_NUM_NODE][2];                       // the children information should come from the TreeGPU.child
    uint32_t posChildEdge[MAX_PATTERN_SIZE][MAX_NUM_NODE];       // k = posChildEdge[i] at mapping size i, kth children are covered
    uint32_t posChildEdgeCount[MAX_PATTERN_SIZE];                // there are posChildEdgeCount[i] children get covered
    uint32_t childEdgeType[MAX_NUM_NODE];                        // which children should be visited is decided by TreeGPU.child

    uint32_t keySize;
    uint32_t aggreVCount;
    uint32_t aggreV[MAX_PATTERN_SIZE];
    uint32_t aggreWeightCount;
    int aggreWeight[MAX_PATTERN_SIZE];
    uint32_t aggrePosCount;
    uint32_t aggrePos[MAX_PATTERN_SIZE];
    uint32_t posAggreEdge[MAX_PATTERN_SIZE][MAX_PATTERN_SIZE];   // there are at most MAX_PATTERN_SIZE aggreVertices
    uint32_t posAggreEdgeCount[MAX_PATTERN_SIZE];
    uint32_t aggreEdgeType[MAX_PATTERN_SIZE];                    // the aggreEdge size is 2 in case an edge key, otherwise it will not exceed the number of MAX_PATTERN_SIZE               

    uint32_t prefixPosCount;                                     // the number of prefix position count
    uint32_t prefixPos[MAX_PATTERN_SIZE];                        // prefixs that extended from the parent node

    uint32_t indexPos[MAX_PATTERN_SIZE];                         // the index position for every node

    // TODO: for lock-based hashtable only, can be removed for other cases
    uint32_t prefixLen[MAX_NUM_NODE];
    uint32_t multiJoin;

    NodeGPU(){}

    NodeGPU(const Tree &t, uint32_t nID) {
        Node tau = t.getNode(nID);

        // load predecessor neighbor information
        for (int i = 0; i < tau.numVertices; i++) {
            std::vector<std::pair<uint32_t, uint32_t>> tmp_pre_nbrs;
            for (int j = 0; j < t.getNodeInPos(nID)[i].size(); j++) {
                tmp_pre_nbrs.emplace_back(std::make_pair(t.getNodeInPos(nID)[i][j], IN_GRAPH));
            }
            for (int j = 0; j < t.getNodeOutPos(nID)[i].size(); j++) {
                tmp_pre_nbrs.emplace_back(std::make_pair(t.getNodeOutPos(nID)[i][j], OUT_GRAPH));
            }
            for (int j = 0; j < t.getNodeUnPos(nID)[i].size(); j++) {
                tmp_pre_nbrs.emplace_back(std::make_pair(t.getNodeUnPos(nID)[i][j], UN_GRAPH));
            }
            std::sort(tmp_pre_nbrs.begin(), tmp_pre_nbrs.end());
            pre_nbrs_count[i] = tmp_pre_nbrs.size();
            for (int j = 0; j < pre_nbrs_count[i]; j++) {
                pre_nbrs_pos[i][j] = tmp_pre_nbrs[j].first;
                pre_nbr_graph_type[i][j] = tmp_pre_nbrs[j].second;
            }
        }

        // load symmetry breaking rules information
        for (int i = 0; i < t.getNodeGreaterPos(nID).size(); i++) {
            nodeGreaterPosCount[i] = t.getNodeGreaterPos(nID)[i].size();
            for (int j = 0; j < nodeGreaterPosCount[i]; j++) {
                nodeGreaterPos[i][j] = t.getNodeGreaterPos(nID)[i][j];
            }
        }
        for (int i = 0; i < t.getNodeLessPos(nID).size(); i++) {
            nodeLessPosCount[i] = t.getNodeLessPos(nID)[i].size();
            for (int j = 0; j < nodeLessPosCount[i]; j++) {
                nodeLessPos[i][j] = t.getNodeLessPos(nID)[i][j];
            }
        }

        // load children key information
        childCount = t.getChild()[nID].size();
        for (int i = 0; i < childCount; i++) {
            children[i] = t.getChild()[nID][i];
        }
        for (int i = 0; i < t.getChildKeyPos(nID).size(); i++) {
            childKeyPosCount[i] = t.getChildKeyPos(nID)[i].size();
            for (int j = 0; j < childKeyPosCount[i]; j++) {
                childKeyPos[i][j] = t.getChildKeyPos(nID)[i][j];
            }
        }
        for (int i = 0; i < t.getPosChildEdge(nID).size(); i++) {
            posChildEdgeCount[i] = t.getPosChildEdge(nID)[i].size();
            for (int j = 0; j < posChildEdgeCount[i]; j++) {
                posChildEdge[i][j] = t.getPosChildEdge(nID)[i][j];
            }
        }
        for (int i = 0; i < t.getChildEdgeType(nID).size(); i++) {
            childEdgeType[i] = t.getChildEdgeType(nID)[i];
        }

        // load aggre key information
        keySize = tau.keySize;
        aggreVCount = t.getAggreV().size();
        for (int i = 0; i < aggreVCount; i++) {
            aggreV[i] = t.getAggreV()[i];
        }
        aggreWeightCount = t.getAggreWeight().size();
        for (int i = 0; i < aggreWeightCount; i++) {
            aggreWeight[i] = t.getAggreWeight()[i];
        }
        aggrePosCount = t.getAggrePos(nID).size();
        for (int i = 0; i < aggrePosCount; i++) {
            aggrePos[i] = t.getAggrePos(nID)[i];
        }
        for (int i = 0; i < t.getPosAggreEdge(nID).size(); i++) {
            posAggreEdgeCount[i] = t.getPosAggreEdge(nID)[i].size();
            for (int j = 0; j < posAggreEdgeCount[i]; j++) {
                posAggreEdge[i][j] = t.getPosAggreEdge(nID)[i][j];
            }
        }
        for (int i = 0; i < t.getAggreEdgeType(nID).size(); i++) {
            aggreEdgeType[i] = t.getAggreEdgeType(nID)[i];
        }

        // load prefix information
        if (t.getPrefixPos().size() == 0) {
            prefixPosCount = 0;
        } else {
            prefixPosCount = t.getPrefixPos()[nID].size();
            for (int i = 0; i < prefixPosCount; i++) {
                prefixPos[i] = t.getPrefixPos()[nID][i];
            }
        }
    }

    NodeGPU(
        const std::vector<bool> &partitionInterPos,
        const std::vector<std::vector<int>> &partitionInPos,
        const std::vector<std::vector<int>> &partitionOutPos,
        const std::vector<std::vector<int>> &partitionUnPos,
        const std::vector<std::vector<int>> &greaterPos,
        const std::vector<std::vector<int>> &lessPos
    ) {
        // load predecessor neighbor information
        for (int i = 0; i < partitionInterPos.size(); i++) {
            std::vector<std::pair<uint32_t, uint32_t>> tmp_pre_nbrs;
            for (int j = 0; j < partitionInPos[i].size(); j++) {
                tmp_pre_nbrs.emplace_back(std::make_pair(partitionInPos[i][j], IN_GRAPH));
            }
            for (int j = 0; j < partitionOutPos[i].size(); j++) {
                tmp_pre_nbrs.emplace_back(std::make_pair(partitionOutPos[i][j], OUT_GRAPH));
            }
            for (int j = 0; j < partitionUnPos[i].size(); j++) {
                tmp_pre_nbrs.emplace_back(std::make_pair(partitionUnPos[i][j], UN_GRAPH));
            }
            std::sort(tmp_pre_nbrs.begin(), tmp_pre_nbrs.end());
            pre_nbrs_count[i] = tmp_pre_nbrs.size();
            for (int j = 0; j < pre_nbrs_count[i]; j++) {
                pre_nbrs_pos[i][j] = tmp_pre_nbrs[j].first;
                pre_nbr_graph_type[i][j] = tmp_pre_nbrs[j].second;
            }
        }

        // load symmetry breaking rules information
        for (int i = 0; i < greaterPos.size(); i++) {
            nodeGreaterPosCount[i] = greaterPos[i].size();
            for (int j = 0; j < nodeGreaterPosCount[i]; j++) {
                nodeGreaterPos[i][j] = greaterPos[i][j];
            }
        }
        for (int i = 0; i < lessPos.size(); i++) {
            nodeLessPosCount[i] = lessPos[i].size();
            for (int j = 0; j < nodeLessPosCount[i]; j++) {
                nodeLessPos[i][j] = lessPos[i][j];
            }
        }
        // load prefix information
        this->prefixPosCount = 0;
    }
};