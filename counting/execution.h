//
// Created by anonymous author on 2022/8/29.
//

#ifndef SCOPE_EXECUTION_H
#define SCOPE_EXECUTION_H

#include <chrono>
#include "graph.h"
#include "decompose.h"
#include "equation.h"
#include "compute_set_intersection.h"
#include "forest.h"
#include "triangle.h"
#include "Pexecution.h"

extern Count gNumIntersect;
extern Count gNumMatch;
extern Count gNumIntermediate;
extern Count gNumEdgeID;
extern Count gNumUpdate;

void setResetProfileMeta(ParallelProcessingMeta *pMeta);

// struct Task {
// public:
//         ui _start;
//         ui _end;
//         int _depth;
//         // ui** _dataV;
//         // ui** _patternV;
//         Task(ui start, ui end, ui depth) : _start(start), _end(end), _depth(depth) {
//                 // _dataV = new ui*[MAX_NUM_NODE];
//                 // _patternV = new ui*[MAX_NUM_NODE];
//                 // for (ui i = 0; i < MAX_NUM_NODE; ++i) {
//                 //         _dataV[i] = new ui[MAX_PATTERN_SIZE];
//                 //         _patternV[i] = new ui[MAX_PATTERN_SIZE];
//                 // }
//                 // _dataV = new ui[MAX_PATTERN_SIZE];
//                 // _patternV = new ui[MAX_PATTERN_SIZE];
//                 // std::copy(dataV, dataV + MAX_PATTERN_SIZE, _dataV);
//                 // std::copy(patternV, patternV + MAX_PATTERN_SIZE, _patternV);
//         }

//         ~Task() {
//                 // for (ui i = 0; i < MAX_NUM_NODE; ++i) {
//                 //         delete[] _dataV[i];
//                 //         delete[] _patternV[i];
//                 // }
//                 // delete[] _dataV;
//                 // delete[] _patternV;
//         }
// };

void saveCount(const std::string &resultPath, std::vector<HashTable> H, const DataGraph &d, bool batchQuery,
               const std::vector<std::string> &files, const std::vector<int> &orbitTypes);
void saveCount(const std::string &resultPath, HashTable h, const DataGraph &d, int orbitType);

void generateCandidate(
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        VertexID *dataV,
        int pos,
        const std::vector<int> &inPos,
        const std::vector<int> &outPos,
        const std::vector<int> &unPos,
        VertexID **candidate,
        ui *&candCount,
        VertexID *&tmp
);

void generateCandidateT(
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        VertexID *dataV,
        int pos,
        bool isTriangle,
        EdgeID e,
        int type,
        const std::vector<int> &inPos,
        const std::vector<int> &outPos,
        const std::vector<int> &unPos,
        VertexID **candidate,
        ui *&candCount,
        VertexID *&tmp
);

ui computeEdgeKey(
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        int mappingSize,
        ui *pos,
        int edgeType,
        VertexID src,
        VertexID dst
);

void executeNode(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV,
        bool matchOnly = false
);

void executeNodeT(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV
);

void PexecuteNodeEdgeKey(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV,
        ParallelProcessingMeta *pMeta,
        bool matchOnly = false
);

void executeNodeEdgeKey(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV,
        bool matchOnly = false
);

void executeNodeEdgeKeyT(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV
);

void executePartition(
        VertexID pID,
        const Tree &t,
        VertexID ***candidate,
        ui **candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
);

void executePartition(
        VertexID pID,
        const Tree &t,
        VertexID ***candidate,
        ui **candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV,
        ParallelProcessingMeta *pMeta,
        bool matchOnly = false
);

void executeTree(
        const Tree &t,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        HashTable *H,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV,
        ParallelProcessingMeta* pMeta,
        bool matchOnly = false
);

void executeTree(
        const Tree &t,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        HashTable *H,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
);

void multiJoin(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV,
        bool matchOnly = false
);

void multiJoin(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV,
        ui start,
        ui end,
        bool matchOnly = false
);

void multiJoinT(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV
);

void multiJoinE(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV,
        bool matchOnly = false
);

void multiJoinE(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV,
        ui start,
        ui end,
        bool matchOnly = false
);

void multiJoinET(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV
);

void multiJoinWrapper(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV,
        bool matchOnly = false
);

void multiJoinWrapper(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV,
        ui start,
        ui end,
        bool matchOnly = false
);

void multiJoinTree(
        const Tree &t,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        HashTable *H,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffset,
        VertexID **patternV,
        VertexID **dataV,
        bool **visited,
        VertexID *&tmp,
        VertexID *allV
);

void multiJoinTree(
        const Tree &t,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        HashTable *H,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffset,
        VertexID **patternV,
        VertexID **dataV,
        bool **visited,
        VertexID *&tmp,
        VertexID *allV,
        ParallelProcessingMeta *pMeta,
        bool matchOnly = false
);

void executeSharedNode(
        const Node &rep,
        const std::vector<std::vector<VertexID>> &children,
        const std::vector<VertexID> &nIDs,
        const std::vector<Tree> &trees,
        std::vector<HashTable *> &hashTables,
        const std::vector<std::vector<int>> &aggreWeights,
        const std::vector<int> &id2AggreKey,
        const std::vector<int> &id2ChildKey,
        const std::vector<bool> &isRoot,
        const std::vector<std::vector<std::vector<int>>> &childKeyPoses,
        const std::vector<std::vector<int>> &aggrePoses,
        const std::vector<int> &tableIDs,
        std::vector<ui *> &keyPos,
        std::vector<ui> &keyPosSize,
        const std::vector<ui> &sizeBound,
        const std::vector<bool> &nodeInterPos,
        const std::vector<std::vector<int>> &nodeInPos,
        const std::vector<std::vector<int>> &nodeOutPos,
        const std::vector<std::vector<int>> &nodeUnPos,
        const std::vector<std::vector<int>> &greaterPos,
        const std::vector<std::vector<int>> &lessPos,
        const std::vector<int> &nonLeafIDs,
        VertexID **candidate,
        ui *candCount,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
);

void executeSharedNodeT(
        const Node &rep,
        const std::vector<std::vector<VertexID>> &children,
        const std::vector<VertexID> &nIDs,
        const std::vector<Tree> &trees,
        std::vector<HashTable *> &hashTables,
        const std::vector<std::vector<int>> &aggreWeights,
        const std::vector<int> &id2AggreKey,
        const std::vector<int> &id2ChildKey,
        const std::vector<bool> &isRoot,
        const std::vector<std::vector<std::vector<int>>> &childKeyPoses,
        const std::vector<std::vector<int>> &aggrePoses,
        const std::vector<int> &tableIDs,
        std::vector<ui *> &keyPos,
        std::vector<ui> &keyPosSize,
        const std::vector<ui> &sizeBound,
        const std::vector<bool> &nodeInterPos,
        const std::vector<std::vector<int>> &nodeInPos,
        const std::vector<std::vector<int>> &nodeOutPos,
        const std::vector<std::vector<int>> &nodeUnPos,
        const std::vector<bool> &nodeCandPos,
        const std::vector<std::vector<int>> &greaterPos,
        const std::vector<std::vector<int>> &lessPos,
        const std::vector<std::pair<int, int>> &nodeTriPos,
        const std::vector<int> &triEdgeType,
        const std::vector<int> &triEndType,
        const std::vector<int> &nonLeafIDs,
        VertexID **candidate,
        ui *candCount,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
);

void executeSharedNodeEdgeKey(
        const Node &rep,
        const std::vector<std::vector<VertexID>> &children,
        const std::vector<VertexID> &nIDs,
        const std::vector<Tree> &trees,
        std::vector<HashTable *> &hashTables,
        const std::vector<std::vector<int>> &aggreWeights,
        const std::vector<int> &id2AggreKey,
        const std::vector<int> &id2ChildKey,
        const std::vector<bool> &isRoot,
        const std::vector<std::vector<std::vector<int>>> &childKeyPoses,
        const std::vector<std::vector<int>> &aggrePoses,
        const std::vector<int> &tableIDs,
        std::vector<ui *> &keyPos,
        std::vector<ui> &keyPosSize,
        const std::vector<ui> &sizeBound,
        const std::vector<bool> &nodeInterPos,
        const std::vector<std::vector<int>> &nodeInPos,
        const std::vector<std::vector<int>> &nodeOutPos,
        const std::vector<std::vector<int>> &nodeUnPos,
        const std::vector<std::vector<int>> &greaterPos,
        const std::vector<std::vector<int>> &lessPos,
        const std::vector<std::vector<std::pair<int, int>>> &posChildEdge,
        const std::vector<std::vector<std::pair<int, int>>> &posAggreEdge,
        const std::vector<std::vector<int>> &childEdgeType,
        const std::vector<std::vector<int>> &aggreEdgeType,
        const std::vector<int> &nonLeafIDs,
        VertexID **candidate,
        ui *candCount,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
);

void executeSharedNodeEdgeKeyT(
        const Node &rep,
        const std::vector<std::vector<VertexID>> &children,
        const std::vector<VertexID> &nIDs,
        const std::vector<Tree> &trees,
        std::vector<HashTable *> &hashTables,
        const std::vector<std::vector<int>> &aggreWeights,
        const std::vector<int> &id2AggreKey,
        const std::vector<int> &id2ChildKey,
        const std::vector<bool> &isRoot,
        const std::vector<std::vector<std::vector<int>>> &childKeyPoses,
        const std::vector<std::vector<int>> &aggrePoses,
        const std::vector<int> &tableIDs,
        std::vector<ui *> &keyPos,
        std::vector<ui> &keyPosSize,
        const std::vector<ui> &sizeBound,
        const std::vector<bool> &nodeInterPos,
        const std::vector<std::vector<int>> &nodeInPos,
        const std::vector<std::vector<int>> &nodeOutPos,
        const std::vector<std::vector<int>> &nodeUnPos,
        const std::vector<bool> &nodeCandPos,
        const std::vector<std::vector<int>> &greaterPos,
        const std::vector<std::vector<int>> &lessPos,
        const std::vector<std::vector<std::pair<int, int>>> &posChildEdge,
        const std::vector<std::vector<std::pair<int, int>>> &posAggreEdge,
        const std::vector<std::vector<int>> &childEdgeType,
        const std::vector<std::vector<int>> &aggreEdgeType,
        const std::vector<std::pair<int, int>> &nodeTriPos,
        const std::vector<int> &nonLeafIDs,
        const std::vector<int> &triEdgeType,
        const std::vector<int> &triEndType,
        VertexID **candidate,
        ui *candCount,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
);

void executeConNode(
        const ConNode &cn,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
);

void executeShared(
        ExeParam &param,
        std::vector<ui *> &keyPoses,
        std::vector<ui> &keyPosSize,
        const std::vector<ui> &sizeBound,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
);

void executeForest(Forest &f, std::vector<HashTable> &result, const DataGraph &din,
                   const DataGraph &dout, const DataGraph &dun, bool useTriangle, const Triangle &tri,
                   std::vector<HashTable> &allocatedHashTable, EdgeID *outID, EdgeID *unID, EdgeID *reverseID,
                   EdgeID *startOffset, VertexID *patternV, VertexID *dataV, bool *visited, ui *candPos,
                   VertexID *&tmp, VertexID *allV, specialsparse *sg, VertexID *cliqueVertices);


class ExecutePartitionWorker {
public:
        ParallelProcessingMeta *pMeta;
        const Tree &t;
        const std::vector<std::vector<VertexID>> &globalOrder;
        const std::vector<std::vector<std::vector<VertexID>>> &nodesAtStep;
        const std::vector<VertexID> &partitionOrder;
        const std::vector<std::vector<VertexID>> &child;
        const std::vector<VertexID> &postOrder;
        const std::vector<int> &partitionPos;
        const std::vector<std::vector<int>> &partitionInPos;
        const std::vector<std::vector<int>> &partitionOutPos;
        const std::vector<std::vector<int>> &partitionUnPos;
        const std::vector<bool> &partitionInterPos;
        const std::vector<std::vector<int>> &greaterPos;
        const std::vector<std::vector<int>> &lessPos;
        const std::vector<bool> &partitionCandPos;
        const std::vector<std::pair<int, int>> &partitionTriPos;
        const std::vector<int> &triEdgeType;
        const std::vector<int> &triEndType;
        EdgeID *inOffset;
        VertexID *inNbors;
        EdgeID *outOffset;
        VertexID *outNbors;
        EdgeID *unOffset;
        EdgeID *unNbors;
        const DataGraph &din;
        const DataGraph &dout;
        const DataGraph &dun;
        bool useTriangle;
        const Triangle &tri;
        EdgeID *outID;
        EdgeID *unID;
        EdgeID *reverseID;
        VertexID pID;
        const Pattern &p;
        const std::vector<std::vector<VertexID>> &allChild;
        int endPos;
        bool isRoot;
        VertexID *allV;
        bool matchOnly;

        ExecutePartitionWorker(
                ParallelProcessingMeta *pMeta,
                const Tree &t,
                const std::vector<std::vector<VertexID>> &globalOrder,
                const std::vector<std::vector<std::vector<VertexID>>> &nodesAtStep,
                const std::vector<VertexID> &partitionOrder,
                const std::vector<std::vector<VertexID>> &child,
                const std::vector<VertexID> &postOrder,
                const std::vector<int> &partitionPos,
                const std::vector<std::vector<int>> &partitionInPos,
                const std::vector<std::vector<int>> &partitionOutPos,
                const std::vector<std::vector<int>> &partitionUnPos,
                const std::vector<bool> &partitionInterPos,
                const std::vector<std::vector<int>> &greaterPos,
                const std::vector<std::vector<int>> &lessPos,
                const std::vector<bool> &partitionCandPos,
                const std::vector<std::pair<int, int>> &partitionTriPos,
                const std::vector<int> &triEdgeType,
                const std::vector<int> &triEndType,
                EdgeID *inOffset,
                VertexID *inNbors,
                EdgeID *outOffset,
                VertexID *outNbors,
                EdgeID *unOffset,
                EdgeID *unNbors,
                const DataGraph &din,
                const DataGraph &dout,
                const DataGraph &dun,
                bool useTriangle,
                const Triangle &tri,
                EdgeID *outID,
                EdgeID *unID,
                EdgeID *reverseID,
                VertexID pID,
                const Pattern &p,
                const std::vector<std::vector<VertexID>> &allChild,
                int endPos,
                bool isRoot,
                VertexID *allV,
                bool matchOnly = false
        );

        ~ExecutePartitionWorker();

        void operator()(ui start, ui end, int depth);
};

class ExecuteMultiJoinWorker {
public:
        ParallelProcessingMeta *pMeta;
        const Tree &t;
        const DataGraph &din;
        const DataGraph &dout;
        const DataGraph &dun;
        const bool useTriangle;
        const Triangle &tri;
        const Pattern &p;
        EdgeID *outID;
        EdgeID *unID;
        EdgeID *reverseID;
        VertexID *allV;
        bool matchOnly;

        ExecuteMultiJoinWorker(
                ParallelProcessingMeta *pMeta,
                const Tree &t,
                const DataGraph &din,
                const DataGraph &dout,
                const DataGraph &dun,
                const bool useTriangle,
                const Triangle &tri,
                const Pattern &p,
                EdgeID *outID,
                EdgeID *unID,
                EdgeID *reverseID,
                VertexID *allV,
                bool matchOnly = false
        );
        ~ExecuteMultiJoinWorker();
        void operator()(ui start, ui end);
};

#endif //SCOPE_EXECUTION_H
