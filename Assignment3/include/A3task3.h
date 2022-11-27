#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"

class A3Task3Solution {
public:
    virtual void prepare(std::vector<float> &input, const uint32_t nBins) = 0;
    virtual void compute() = 0;
    virtual std::vector<int> result() const = 0;
};


class A3Task3 {
public:
    A3Task3(float* input, uint32_t size, uint32_t numBins);

    bool evaluateSolution(A3Task3Solution& solution);
    std::vector<int> reference;

private:
    void computeReference();
    std::vector<float> input; 
    uint32_t numBins;
};
  
void print_histogram(const std::vector<int> &h);