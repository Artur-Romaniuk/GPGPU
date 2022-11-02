#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"
#include "A1task2.h"

/* requires to have called prepare() because we need the buffers to be correctly created*/
std::vector<int> A1_Task2::incArray()
{
    // === prepare data ===
    std::vector<int> inputVec(workloadSize, 0u);
    for (size_t i = 0; i < inputVec.size(); i++)
    {
        inputVec[i] = i;
    }
    return inputVec;
}
void A1_Task2::defaultValues()
{
    std::vector<int> inputVec = incArray();
    // === fill buffer ===   
}

void A1_Task2::checkDefaultValues()
{
    // ### gather the output data after having called compute() ###
    std::vector<int> result(workloadSize, 1u);

    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, outBuffer, result);
    std::vector<int> input = incArray();

    std::vector<int> rotate = rotateCPU(input, workloadSize_w, workloadSize_h);
    int errors = 0;
    for(int i = 0 ; i < rotate.size(); i++)
        if(rotate[i] != result[i])
            errors++;
    if(errors>0)
        std::cout<<std::endl<<"=== There were " << errors << " error(s). ===" << std::endl;

/*
    std::cout << std::endl
              << std::endl
              << "========================" << std::endl
              << std::endl;
    // ### CHECK RESULT IS VALID ###
    for (unsigned int i = 0; i < workloadSize; i++)
    {
        if (i % workloadSize_h == 0 && i != 0)
            std::cout << std::endl;
        std::cout << result[i] << " ";
    }*/
}

void A1_Task2::prepare(unsigned int size_w, unsigned int size_h)
{
    this->workloadSize_w = size_w;
    this->workloadSize_h = size_h;
    this->workloadSize = size_h * size_w;

    // ### this fills the descriptorLayoutBindings  ###
    Cmn::addStorage(task.bindings, 0);
    Cmn::addStorage(task.bindings, 1);
    

    Cmn::createDescriptorSetLayout(app.device, task.bindings, task.descriptorSetLayout);
    Cmn::createDescriptorPool(app.device, task.bindings, task.descriptorPool);
    Cmn::allocateDescriptorSet(app.device, task.descriptorSet, task.descriptorPool, task.descriptorSetLayout);

    // ### create buffers, get their index in the task.buffers[] array ###
    using BFlag = vk::BufferUsageFlagBits;
    auto makeDLocalBuffer = [this](vk::BufferUsageFlags usage, vk::DeviceSize size, std::string name) -> Buffer
    {// this lambda just fills the createBuffer with a more human readable set of parameters
        Buffer b;
        createBuffer(app.pDevice, app.device, size, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, name, b.buf, b.mem);
        return b;
    };

    this->inBuffer = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, workloadSize * sizeof(unsigned int), "buffer_in");
    this->outBuffer = makeDLocalBuffer(BFlag::eTransferSrc | BFlag::eStorageBuffer, workloadSize * sizeof(unsigned int), "buffer_out");
    /*
    Todo : create descriptorsetlayout pool set...
    */

    /*
    TODO: create the storage input and output buffers
    */
    this->defaultValues();

    // ### Bind buffers to descriptor set ### (calls update several times)
    Cmn::bindBuffers(app.device, inBuffer.buf, task.descriptorSet, 0);
    Cmn::bindBuffers(app.device, outBuffer.buf, task.descriptorSet, 1);

    // ### Create Pipeline Layout ###
    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct));
    vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &task.descriptorSetLayout, 1U, &pcr);
    task.pipelineLayout = app.device.createPipelineLayout(pipInfo);
}

void A1_Task2::compute(uint32_t dx, uint32_t dy, uint32_t dz, std::string file)
{
    int numGroupsX, numGroupsY;
    PushStruct push;
    
    // ### Create ShaderModule ###
    // Same as in task1

    // ### Create Pipeline ###
    // create the specialization entries for a 2D array
    vk::SpecializationInfo specInfo;
    // in case a pipeline was already created, destroy it
    
    vk::PipelineShaderStageCreateInfo stageInfo(vk::PipelineShaderStageCreateFlags(),
                                            vk::ShaderStageFlagBits::eCompute, task.cShader,
                                            "main", &specInfo);
    app.device.destroyPipeline( task.pipeline );
    vk::ComputePipelineCreateInfo computeInfo(vk::PipelineCreateFlags(), stageInfo, task.pipelineLayout);
    task.pipeline = app.device.createComputePipeline(nullptr, computeInfo, nullptr).value;

    // ### finally do the compute ###
    this->dispatchWork(numGroupsX, numGroupsY, 1U, push);

    // ### gather the output data ###
    std::vector<unsigned int> result(workloadSize, 1u);

    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, outBuffer, result);

    // ### END ###
}

void A1_Task2::dispatchWork(uint32_t dx, uint32_t dy, uint32_t dz, PushStruct &pushConstant)
{
    /* ### You can copy/paste the function of A1_Task1 here ### */

}