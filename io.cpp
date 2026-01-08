// This file is part of openpbso, an open-source library for physics-based sound
//
// Copyright (C) 2018 Jui-Hsien Wang <juiwang@alumni.stanford.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/. */
#include "io.h"
#include <string>
// #include <dirent.h>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>
//#include <unistd.h>
#include <iostream>
namespace fs = std::filesystem;
//##############################################################################
namespace Gpu_Wavesolver {
//##############################################################################
void ListDirFiles(const char *dirname, std::vector<std::string> &names,
    const char *contains) {
    try {
        for (const auto& entry : fs::directory_iterator(dirname)) {
            if (!entry.is_regular_file()) continue;
            std::string f = entry.path().string();
            std::string fname = entry.path().filename().string();
            if (fname.empty() || fname[0] == '.') continue;
            if (contains && std::string(f).find(std::string(contains)) == std::string::npos) continue;
            names.push_back(f);
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error opening directory: " << e.what() << std::endl;
    }
}
//##############################################################################
bool IsFile(const char *path) {
    // seems to fail sometimes..
    //struct stat path_stat;
    //stat(path, &path_stat);
    //return S_ISREG(path_stat.st_mode);
    // ref: https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
    struct stat buffer;
    return (stat (path, &buffer) == 0);
}
//##############################################################################
std::string Basename(const std::string &path) {
    auto start = path.find_last_of("/");
    return path.substr(start+1);
}
//##############################################################################
};
//##############################################################################
