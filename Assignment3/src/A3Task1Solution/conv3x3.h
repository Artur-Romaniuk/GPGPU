#pragma once

#include "A3Task1.h"

class ConvolutionA3Task1Solution : public A3Task1Solution{
public:
    ConvolutionA3Task1Solution(AppResources &app, uint32_t workGroupSize_x, uint32_t workGroupSize_y);
    void cleanup();

    void prepare(const std::vector<float> &input, const std::vector<float> ker, uint32_t w, uint32_t h) override;
    void compute() override;
    std::vector<float> result() const override;

    float mstime, insideTime;

private:
    struct PushConstant
    {
        uint32_t width;
        uint32_t height;
        uint32_t pitch;
        float kernelWeight;
        float kernel[9];
    } push;

    AppResources &app;
    uint32_t workGroupSize_x, workGroupSize_y;
    int w, h, p;
    std::vector<float> imgInput;
    std::vector<float> kernel;

    Buffer inBuf, outBuf;

    // Descriptor & Pipeline Layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;

    // Local PPS Pipeline
    vk::ShaderModule shaderModule;
    vk::Pipeline pipeline;

    // Descriptor Pool
    vk::DescriptorPool descriptorPool;

    // Per-dispatch data
    vk::DescriptorSet descriptorSet;
};
