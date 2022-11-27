#pragma once

#include "A3Task2.h"

class SeparableA3Task2Solution : public A3Task2Solution{
public:
    SeparableA3Task2Solution(AppResources &app, uint32_t workGrpSize_H[2], uint32_t workGrpSize_V[2]);
    void cleanup();

    void prepare(const std::vector<float> &input, const std::vector<float> horizKer, const std::vector<float> vertKer, uint32_t width, uint32_t height, uint32_t H_ELEMENTS, uint32_t V_ELEMENTS) override;
    void compute() override;
    std::vector<float> result() const;

    float mstime, insideTime;

private:
    struct PushConstant
    {
        uint32_t width;
        uint32_t height;
        uint32_t pitch;
        float kernelWeight;
        float kernel[32];
    };
    PushConstant pushHoriz;
    PushConstant pushVert;
    AppResources &app;
    uint32_t workGroupSize_H[2], workGroupSize_V[2];
    int w, h, p, H_elements, V_elements;
    std::vector<float> imgInput;
    std::vector<float> kernelHoriz, kernelVert;

    Buffer inOutBuf, interBuf;

    // Descriptor & Pipeline Layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayoutHoriz;
    vk::PipelineLayout pipelineLayoutVert;
    

    // Local PPS Pipeline
    vk::ShaderModule shaderModuleHoriz;
    vk::ShaderModule shaderModuleVert;
    vk::Pipeline pipelineHoriz;
    vk::Pipeline pipelineVert;

    // Descriptor Pool
    vk::DescriptorPool descriptorPool;

    // Per-dispatch data
    vk::DescriptorSet descriptorSet;
};
