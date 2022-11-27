#pragma once

#include "A3Task3.h"

class HistogramA3Task3Solution : public A3Task3Solution{
public:
    HistogramA3Task3Solution(AppResources &app, int workGroupSize);
    void cleanup();

    void prepare(std::vector<float> &input, const uint32_t nBins) override;
    void compute() override;
    std::vector<int> result() const override;

    float mstime, insideTime;
    std::string shaderName = "./shaders/histogram.comp.spv";
    void useNaive(){
        shaderName="./shaders/histogram_naive.comp.spv";
    }
    void useLocalOpti(){
        shaderName="./shaders/histogram.comp.spv";
    }
private:
    struct PushConstant
    {
        int size;
    } push;

    AppResources &app;
    uint32_t workGroupSize, nBins;

    std::vector<float> mpInput;

    Buffer matrixBuffer, histoBuffer;

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
