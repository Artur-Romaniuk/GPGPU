#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include <algorithm>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"
#include "A1task1.h"

void defaultVectors(std::vector<int> &in1, std::vector<int> &in2, size_t size)
{
    //Prepare data
    in1 = std::vector<int>(size, 0u);
    for (size_t i = 0; i < in1.size(); i++)
        in1[i] = static_cast<int>(i);
    in2 = std::vector<int>(in1);
    std::reverse(in2.begin(), in2.end());
}
   //Requires to have called prepare() because we need the buffers to be correctly created
void A1_Task1::defaultValues()
{
    std::vector<int> inputVec, inputVec2;
    defaultVectors(inputVec, inputVec2, this->workloadSize);
    //Fill buffers
    //std::cout << "Filling buffers..." << std::endl;
    fillDeviceBuffer(app.device, inBuffer1.mem, inputVec);
    fillDeviceBuffer(app.device, inBuffer2.mem, inputVec2);
    //fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inBuffer1, inputVec);
    //fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inBuffer2, inputVec2);
}

void A1_Task1::checkDefaultValues()
{
    //Gather the output data after having called compute()
    std::vector<unsigned int> result(this->workloadSize, 1u);

    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, outBuffer, result);

    std::vector<int> inputVec, inputVec2;
    defaultVectors(inputVec, inputVec2, this->workloadSize);
    std::vector<int> outputVec(this->workloadSize, 0u);
    std::transform(inputVec.begin(), inputVec.end(), inputVec2.begin(), outputVec.begin(), std::plus<int>());

    if (std::equal(result.begin(), result.end(), outputVec.begin()))
        std::cout << "A1_Task1, TEST PASSED with size:" << this->workloadSize<< std::endl;
    else
        std::cout << " Oh no! We found errors! Good luck :)" << std::endl;
}

void A1_Task1::prepare(unsigned int size)
{
    this->workloadSize = size;
    
    // ### Fill the descriptorLayoutBindings  ###

    //DescriptorSetLayout holds the shape and usage of buffers but not the buffers themselves
    Cmn::addStorage(task.bindings, 0);
    Cmn::addStorage(task.bindings, 1);
    Cmn::addStorage(task.bindings, 2);

    Cmn::createDescriptorSetLayout(app.device, task.bindings, task.descriptorSetLayout);
    Cmn::createDescriptorPool(app.device, task.bindings, task.descriptorPool);
    Cmn::allocateDescriptorSet(app.device, task.descriptorSet, task.descriptorPool, task.descriptorSetLayout);

    // ### Push Constant ###

    // ### Create Pipeline Layout ###
    //task.pipelineLayout = ...

    // ### create buffers ###
    createBuffer(app.pDevice, app.device, workloadSize * sizeof(unsigned int), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostCoherent, "inBuffer1", this->inBuffer1.buf,  this->inBuffer1.mem);
    createBuffer(app.pDevice, app.device, workloadSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostCoherent, "inBuffer2", this->inBuffer2.buf,  this->inBuffer2.mem);

    createBuffer(app.pDevice, app.device, workloadSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostCoherent, "outBuffer", this->outBuffer.buf,  this->outBuffer.mem);

    // ### Fills inBuffer1 and inBuffer2 ###

    // This creates default values 
    this->defaultValues();

    // ### Create  structures ###
    // ### DescriptorSet is created but not filled yet ###
    // ### Bind buffers to descriptor set ### (calls update several times)

    Cmn::bindBuffers(app.device, inBuffer1.buf, task.descriptorSet, 0);
    Cmn::bindBuffers(app.device, inBuffer2.buf, task.descriptorSet, 1);
    Cmn::bindBuffers(app.device, outBuffer.buf, task.descriptorSet, 2);

    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0U, sizeof(PushStruct));
    vk::PipelineLayoutCreateInfo pipLayoutInfo(vk::PipelineLayoutCreateFlags(), 1U, &task.descriptorSetLayout, 1U, &pcr);
    task.pipelineLayout = app.device.createPipelineLayout(pipLayoutInfo);

    // ### Preparation work done! ###
}

void A1_Task1::compute(uint32_t dx, uint32_t dy, uint32_t dz, std::string file)
{
    uint32_t groupCount = (workloadSize+dx-1) / dx;
    PushStruct push{workloadSize}; 
    // ### Create ShaderModule ###
    std::string compute = "./build/shaders/"+file+"comp.spv";
    app.device.destroyShaderModule(task.cShader);
    Cmn::createShader(app.device, task.cShader, compute);

    // ### Specialization constants
    // constantID, offset, sizeof(type)
    std::array<vk::SpecializationMapEntry , 1> specEntries =  std::array<vk::SpecializationMapEntry, 1U>{{{0U, 0U, sizeof(int)}}} ;
    std::array<int , 1U> specValues = { int( dx ) } ;
    vk::SpecializationInfo specInfo = vk::SpecializationInfo( CAST(specEntries), specEntries.data(), CAST(specValues) * sizeof(int) , specValues.data());

    vk::PipelineShaderStageCreateInfo  stageInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, task.cShader, "main", &specInfo);

    // ### Create Pipeline ###
    app.device.destroyPipeline(task.pipeline);
    vk::ComputePipelineCreateInfo  computeInfo(vk::PipelineCreateFlags(), stageInfo, task.pipelineLayout);
    task.pipeline = app.device.createComputePipeline(nullptr, computeInfo, nullptr).value;

    // ### finally do the compute ###
    this->dispatchWork(groupCount, 1U, 1U, push);
}

void A1_Task1::dispatchWork(uint32_t dx, uint32_t dy, uint32_t dz, PushStruct &pushConstant)
{
    /* ### Create Command Buffer ### */
    vk::CommandBufferAllocateInfo allocInfo(app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers(allocInfo)[0U];

    /* ### Call Begin and register commands ### */
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cb.begin(beginInfo);

    cb.resetQueryPool(app.queryPool, 0, 2);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 0);
    cb.pushConstants(task.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct), &pushConstant);
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, task.pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, task.pipelineLayout, 0U, 1U, &task.descriptorSet, 0U, nullptr);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 1);

    /* ### End of Command Buffer, enqueue it and use a Fence ### */
    cb.end();
    
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &cb);
    vk::Fence fence = app.device.createFence(vk::FenceCreateInfo());
    app.computeQueue.submit({submitInfo}, fence);
    vk::Result haveIWaited = app.device.waitForFences({fence}, true, UINT64_MAX);
    app.device.destroyFence(fence);

    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);

    /* ### Collect data from the query Pool ### */

    /* Uncomment this once you've finished this function: */
    uint64_t timestamps[2];
    vk::Result result = app.device.getQueryPoolResults(app.queryPool, 0, 2, sizeof(timestamps), &timestamps, sizeof(timestamps[0]), vk::QueryResultFlagBits::e64);
    assert(result == vk::Result::eSuccess);
    uint64_t timediff = timestamps[1] - timestamps[0];
    vk::PhysicalDeviceProperties properties = app.pDevice.getProperties();
    uint64_t nanoseconds = properties.limits.timestampPeriod * timediff;

    mstime = nanoseconds / 1000000.f;
}