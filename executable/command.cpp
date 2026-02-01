//
// Created by Qiyan LI on 2022/8/30.
//

#include "command.h"

Command::Command(int argc, char **argv) : CommandParser(argc, argv){
    optionsKey[OptionKeyword::QueryGraphPath] = "-q";
    optionsKey[OptionKeyword::DataGraphPath] = "-d";
    optionsKey[OptionKeyword::BatchQuery] = "-b";
    optionsKey[OptionKeyword::ResultPath] = "-r";
    optionsKey[OptionKeyword::ShareNode] = "-share";
    optionsKey[OptionKeyword::TrianglePath] = "-t";
    optionsKey[OptionKeyword::HashtableSizeFactor] = "-ratio";
    optionsKey[OptionKeyword::ProbLimit] = "-prob";
    optionsKey[OptionKeyword::MemoryPool] = "-mem";
    floatOptionValue[OptionKeyword::HashtableSizeFactor] = 1.0;
    intOptionValue[OptionKeyword::ProbLimit] = 64;
    intOptionValue[OptionKeyword::MemoryPool] = 0;
    booleanOptionValue[OptionKeyword::BatchQuery] = false;
    booleanOptionValue[OptionKeyword::ShareNode] = false;
    processOptions();
}

void Command::processOptions() {
    optionsValue[OptionKeyword::QueryGraphPath] = getCommandOption(optionsKey[OptionKeyword::QueryGraphPath]);
    optionsValue[OptionKeyword::DataGraphPath] = getCommandOption(optionsKey[OptionKeyword::DataGraphPath]);
    optionsValue[OptionKeyword::TrianglePath] = getCommandOption(optionsKey[OptionKeyword::TrianglePath]);
    optionsValue[OptionKeyword::ResultPath] = getCommandOption(optionsKey[OptionKeyword::ResultPath]);
    if (commandOptionExists(optionsKey[OptionKeyword::HashtableSizeFactor])) {
        floatOptionValue[OptionKeyword::HashtableSizeFactor] = std::stof(getCommandOption(optionsKey[OptionKeyword::HashtableSizeFactor]));
    }
    if (commandOptionExists(optionsKey[OptionKeyword::ProbLimit])) {
        intOptionValue[OptionKeyword::ProbLimit] = std::stoi(getCommandOption(optionsKey[OptionKeyword::ProbLimit]));
    }
    if (commandOptionExists(optionsKey[OptionKeyword::MemoryPool])) {
        intOptionValue[OptionKeyword::MemoryPool] = std::stoi(getCommandOption(optionsKey[OptionKeyword::MemoryPool]));
    }
    booleanOptionValue[OptionKeyword::BatchQuery] = commandOptionExists(optionsKey[OptionKeyword::BatchQuery]);
    booleanOptionValue[OptionKeyword::ShareNode] = commandOptionExists(optionsKey[OptionKeyword::ShareNode]);
}