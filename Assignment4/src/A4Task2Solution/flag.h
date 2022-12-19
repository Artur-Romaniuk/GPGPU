#pragma once

#include "A4Task2.h"

class FlagA4Task2Solution : public A4Task2Solution{
public:
    FlagA4Task2Solution(AppResources &app, A4Task2Data &datad, uint32_t workGroupSize_x, uint32_t workGroupSize_y);
    
    void cleanup();

    void prepare() override;
    void compute() override;
    std::vector<float> result() const override;

    float mstime, insideTime;

private:

    AppResources &app;
    A4Task2Data &data;
    uint32_t workGroupSize_x, workGroupSize_y;
    // Local PPS Pipeline
    vk::ShaderModule integrateShader, collisionShader, normalsShader, constraintsShader;
    vk::Pipeline integratePipeline, collisionPipeline, normalsPipeline, constraintsPipeline;
};
