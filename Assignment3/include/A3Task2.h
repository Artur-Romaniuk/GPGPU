#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"

class A3Task2Solution {
public:
    virtual void prepare(const std::vector<float> &input, 
        const std::vector<float> horizKer, const std::vector<float> vertKer, 
        uint32_t width, uint32_t height, 
        uint32_t H_ELEMENTS, uint32_t V_ELEMENTS) = 0;
    virtual void compute() = 0;
    virtual std::vector<float> result() const = 0;
};

class A3Task2 {
public:
    A3Task2(float* input, 
    std::vector<float> kernelH, std::vector<float> kernelV, 
    uint32_t w, uint32_t h, uint32_t h_elements, uint32_t v_elements);

    bool evaluateSolution(A3Task2Solution& solution);

private:
    void computeReference();
    std::vector<float> input;
    uint32_t w, h, nH, nV; //nH: number of elements treated per thread
    std::vector<float> reference;
    std::vector<float> kernelH, kernelV;
};
  