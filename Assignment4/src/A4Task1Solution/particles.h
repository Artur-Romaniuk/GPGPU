#pragma once

#include "A4Task1.h"

class ParticlesA4Task1Solution : public A4Task1Solution{
public:
    ParticlesA4Task1Solution(AppResources &app, A4Task1Data &datad, uint32_t workGroupSize_x, uint32_t triangleCacheSize);
    
    void cleanup();

    void prepare() override;
    void compute() override;
    std::vector<float> result() const override;

    float mstime, insideTime;

private:

    void make3DTexture();
    void makeSampler();
    void initParticles();

    AppResources &app;
    A4Task1Data &data;
    uint32_t workGroupSize_x;
    uint32_t triangleCacheSize;
    // Local PPS Pipeline
    vk::ShaderModule particleShader, scanShader, reorgShader;
    vk::Pipeline particlePipeline, scanPipeline, reorgPipeline;

    std::vector<vk::DescriptorSetLayoutBinding> scanBindings, reorgBindings;
    vk::DescriptorSetLayout scanDescriptorSetLayout, reorgDescriptorSetLayout;
    vk::PipelineLayout scanLayout, reorgLayout;

    vk::CommandBuffer cb;
};
