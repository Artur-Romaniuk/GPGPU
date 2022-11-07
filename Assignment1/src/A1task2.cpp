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
    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inBuffer, inputVec);
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

    // std::cout << std::endl
    //           << std::endl
    //           << "========================" << std::endl
    //           << std::endl;
    // ### CHECK RESULT IS VALID ###
    // std::ofstream outFile("out.data");
    // for (unsigned int i = 0; i < workloadSize; i++)
    // {
    //     if (i % workloadSize_h == 0 && i != 0)
    //         outFile << "\n";
    //     outFile << result[i] << " ";
    // }
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
    this->outBuffer = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eTransferSrc | BFlag::eStorageBuffer, workloadSize * sizeof(unsigned int), "buffer_out");
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
    int numGroupsX = (this->workloadSize_w + dx -1)/dx;
    int numGroupsY = (this->workloadSize_h + dy -1)/dy;
    PushStruct push{this->workloadSize_w, this->workloadSize_h};
    
    // ### Create ShaderModule ###
    // Same as in task1

    // ### Create Pipeline ###
    // create the specialization entries for a 2D array
    std::array<vk::SpecializationMapEntry , 2U> specEntries =  std::array<vk::SpecializationMapEntry, 2U>{vk::SpecializationMapEntry{0U, 0U, sizeof(int)}, vk::SpecializationMapEntry{1U, sizeof(int), sizeof(int)}} ;
    std::array<int , 2U> specValues = { int( dx ), int(dy) } ;
    vk::SpecializationInfo specInfo = vk::SpecializationInfo( CAST(specEntries), specEntries.data(), CAST(specValues) * sizeof(int) , specValues.data());
    // in case a pipeline was already created, destroy it
    
    std::string compute = "../shaders/"+file+".comp.spv";
    app.device.destroyShaderModule(task.cShader);
    Cmn::createShader(app.device, task.cShader, compute);
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
    cb.dispatch(dx, dy, dz);
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