//
// Created by anonymous author on 2022/8/30.
//

#ifndef SCOPE_COMMAND_H
#define SCOPE_COMMAND_H

#include "command_parser.h"
#include <iostream>
#include <map>

enum OptionKeyword {
    QueryGraphPath = 1,      // -q, the query graph file path
    DataGraphPath = 2,       // -d, the data graph file path
    TrianglePath = 3,        // -t, the triangle binary file path
    ResultPath = 4,          // -r, the result file path
    BatchQuery = 5,          // -b, batch query or single query
    ShareNode = 6,           // -share, enable sharing nodes or not
    HashtableSizeFactor = 7, // -ratio, the factor of hash table size
    ProbLimit = 8,           // -prob, the limit of probability
    MemoryPool = 9,          // -mem, the size of memory pool in GB
    ExecutionMemoryPool = 10, // -exec-mem, execution memory budget in GB
    ProfileReset = 11,        // -profile-reset, profile GPU table reset time
    OccupancyProfile = 12,    // -occupancy-profile, profile hash-table occupancy
    RecordBatchSchedule = 13, // -record-batch-schedule, write adaptive GPU batch schedule
    ReplayBatchSchedule = 14, // -replay-batch-schedule, replay adaptive GPU batch schedule
    MatchOnly = 15            // -match-only, run without table reset and join aggregation
};

class Command : public CommandParser {
private:
    std::map<OptionKeyword, std::string> optionsKey;
    std::map<OptionKeyword, std::string> optionsValue;
    std::map<OptionKeyword, bool> booleanOptionValue;
    std::map<OptionKeyword, int> intOptionValue;
    std::map<OptionKeyword, float> floatOptionValue;

private:
    void processOptions();

public:
    Command(int argc, char **argv);

    std::string getQueryGraphPath() {
        return optionsValue[OptionKeyword::QueryGraphPath];
    }

    std::string getDataGraphPath() {
        return optionsValue[OptionKeyword::DataGraphPath];
    }

    std::string getTrianglePath() {
        return optionsValue[OptionKeyword::TrianglePath];
    }

    std::string getResultPath() {
        return optionsValue[OptionKeyword::ResultPath];
    }

    bool getBatchQuery() {
        return booleanOptionValue[OptionKeyword::BatchQuery];
    }

    bool getShareNode() {
        return booleanOptionValue[OptionKeyword::ShareNode];
    }

    float getRatio() {
        return floatOptionValue[OptionKeyword::HashtableSizeFactor];
    }

    uint32_t getProbLimit() {
        return intOptionValue[OptionKeyword::ProbLimit];
    }

    uint32_t getMemoryPoolSize() {
        return intOptionValue[OptionKeyword::MemoryPool];
    }

    float getExecutionMemoryPoolSize() {
        return floatOptionValue[OptionKeyword::ExecutionMemoryPool];
    }

    bool getProfileReset() {
        return booleanOptionValue[OptionKeyword::ProfileReset];
    }

    bool getOccupancyProfile() {
        return booleanOptionValue[OptionKeyword::OccupancyProfile];
    }

    std::string getRecordBatchSchedulePath() {
        return optionsValue[OptionKeyword::RecordBatchSchedule];
    }

    std::string getReplayBatchSchedulePath() {
        return optionsValue[OptionKeyword::ReplayBatchSchedule];
    }

    bool getMatchOnly() {
        return booleanOptionValue[OptionKeyword::MatchOnly];
    }
};


#endif //SCOPE_COMMAND_H
