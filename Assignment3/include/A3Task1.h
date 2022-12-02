#pragma once

#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"

class A3Task1Solution {
public:
    virtual void prepare(const std::vector<float> &input, const std::vector<float> ker, uint32_t w, uint32_t h) = 0;
    virtual void compute() = 0;
    virtual std::vector<float> result() const = 0;
};

class A3Task1 {
public:
    A3Task1(int w, int h);
    A3Task1(float* input, float k[3][3], uint32_t w, uint32_t h);
    A3Task1(std::vector<float> input, float k[3][3], uint32_t w, uint32_t h);

    bool evaluateSolution(A3Task1Solution& solution);

private:
    void computeReference();
    std::vector<float> input;
    uint32_t w, h;
    std::vector<float> reference;
    float kernel[3][3];
};