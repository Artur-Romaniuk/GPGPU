#include "Naive.h"

#include "host_timer.h"

A2Task2SolutioNaive::A2Task2SolutioNaive(
    AppResources &app, uint workGroupSize):
    app(app), workGroupSize(workGroupSize) {}

void A2Task2SolutioNaive::prepare(const std::vector<uint> &input) {
    workSize = input.size();

    // Descriptor & Pipeline Layout
    Cmn::addStorage(bindings, 0);
    Cmn::addStorage(bindings, 1);
    Cmn::createDescriptorSetLayout(app.device, bindings, descriptorSetLayout);
    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct));
    vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &descriptorSetLayout, 1U, &pcr);
    pipelineLayout = app.device.createPipelineLayout(pipInfo);

    // Specialization constant for workgroup size
    std::array<vk::SpecializationMapEntry, 1> specEntries = std::array<vk::SpecializationMapEntry, 1>{ 
        {{0U, 0U, sizeof(workGroupSize)}},
    }; 
    std::array<uint32_t, 2> specValues = {workGroupSize}; //for workgroup sizes
    vk::SpecializationInfo specInfo = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                    CAST(specValues) * sizeof(int), specValues.data());

    // Local PPS Offset Pipeline
    Cmn::createShader(app.device, cShader, "./shaders/A2Task2Naive.comp.spv");
    Cmn::createPipeline(app.device, pipeline, pipelineLayout, specInfo, cShader);

    // ### create buffers, get their index in the task.buffers[] array ###
    using BFlag = vk::BufferUsageFlagBits;
    auto makeDLocalBuffer = [ this ](vk::BufferUsageFlags usage, vk::DeviceSize size, std::string name) -> Buffer
    {
        Buffer b;
        createBuffer(app.pDevice, app.device, size, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, name, b.buf, b.mem);
        return b;
    };

    for (int i = 0; i < 2; i++)
        createBuffer(app.pDevice, app.device, input.size() * sizeof(uint32_t), BFlag::eTransferDst | BFlag::eTransferSrc | BFlag::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, "buffer_" + std::to_string(i), buffers[i]);

    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, buffers[0], input);

    Cmn::createDescriptorPool(app.device, bindings, descriptorPool, 2);

    for (uint i = 0; i < 2; i++)
        Cmn::allocateDescriptorSet(app.device, descriptorSets[i], descriptorPool, descriptorSetLayout);
    Cmn::bindBuffers(app.device, buffers[0].buf, descriptorSets[0], 0);
    Cmn::bindBuffers(app.device, buffers[1].buf, descriptorSets[0], 1);
    Cmn::bindBuffers(app.device, buffers[1].buf, descriptorSets[1], 0);
    Cmn::bindBuffers(app.device, buffers[0].buf, descriptorSets[1], 1);

    activeBuffer = 0;
}

void A2Task2SolutioNaive::compute() {
    vk::CommandBufferAllocateInfo allocInfo(
        app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers( allocInfo )[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    uint32_t groupCount = ( workSize + workGroupSize -1) / workGroupSize;
    PushStruct pushConstant{
        .size = static_cast<uint32_t>(workSize),
        .offset = 1
    };

    cb.begin(beginInfo);

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

    // TO DO: Implement naive scan
    // NOTE: make sure that activeBuffer points to the buffer with the final result in the end
    // That buffer is read back for the correctness check
    // (A2Task2SolutioNaive::result())
    // HINT: You can alternate between the two provided descriptor sets to implement ping-pong

    while(pushConstant.offset<pushConstant.size)
    {
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0U, 1U, &descriptorSets[activeBuffer], 0U, nullptr);
        cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct), &pushConstant);
        cb.dispatch(groupCount, 1, 1);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(), {vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderWrite)},{},{});
        pushConstant.offset*=2;
        activeBuffer = ((activeBuffer==0) ? 1 : 0); 
    }

    cb.end();

    vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb);

    HostTimer timer;

    app.computeQueue.submit({submitInfo});
    app.device.waitIdle();

    mstime = timer.elapsed() * 1000;

    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}

std::vector<uint> A2Task2SolutioNaive::result() const {
    std::vector<uint> result(workSize, 0);
    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, buffers[activeBuffer], result);
    return result;
}

void A2Task2SolutioNaive::cleanup() {
    app.device.destroyDescriptorPool(descriptorPool);

    app.device.destroyPipeline(pipeline);
    app.device.destroyShaderModule(cShader);

    app.device.destroyPipelineLayout(pipelineLayout);
    app.device.destroyDescriptorSetLayout(descriptorSetLayout);
    bindings.clear();

    auto Bclean = [&](Buffer &b){
        app.device.destroyBuffer(b.buf);
        app.device.freeMemory(b.mem);};

    for (auto buffer : buffers)
        destroyBuffer(app.device, buffer);
}