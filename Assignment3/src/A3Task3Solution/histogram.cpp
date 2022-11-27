#include "histogram.h"

#include "host_timer.h"

template <typename T, typename V>
T ceilDiv(T x, V y)
{
    return x / y + (x % y != 0);
}

HistogramA3Task3Solution::HistogramA3Task3Solution(AppResources &app, int workGroupSize) : app(app), workGroupSize(workGroupSize)
{
}

void HistogramA3Task3Solution::prepare(std::vector<float> &input, const uint32_t numBins)
{
    mpInput = input;
    nBins = numBins;
    cleanup();

    Cmn::addStorage(bindings, 0);
    Cmn::addStorage(bindings, 1);

    Cmn::createDescriptorSetLayout(app.device, bindings, descriptorSetLayout);
    // Define push constant values (no 'create' needed!)
    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,            // Shader stage push constant will go to
        0,                                            // Offset into given data to pass to push constant
        sizeof(PushConstant)                          // Size of data being passed
    );
    vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &descriptorSetLayout, 1U, &pcr);
    pipelineLayout = app.device.createPipelineLayout(pipInfo);

    // Specialization constant for workgroup size
    std::array<vk::SpecializationMapEntry, 2> specEntries = std::array<vk::SpecializationMapEntry, 2>{
        {{0U, 0U, sizeof(workGroupSize)},
        {1U, sizeof(workGroupSize), sizeof(nBins)}},
    };
    std::array<uint32_t, 2> specValues = {workGroupSize, nBins}; 
    vk::SpecializationInfo specInfo = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                                             CAST(specValues) * sizeof(int), specValues.data());

    Cmn::createShader(app.device, shaderModule, shaderName);
    Cmn::createPipeline(app.device, pipeline, pipelineLayout, specInfo, shaderModule);

    createBuffer(app.pDevice, app.device, mpInput.size() * sizeof(mpInput[0]),
                 vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, "matrixBuffer", matrixBuffer.buf, matrixBuffer.mem);
    createBuffer(app.pDevice, app.device, nBins * sizeof(uint32_t),
                vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, "histoBuffer", histoBuffer.buf, histoBuffer.mem);
    std::vector<uint32_t> histEmpty(nBins,0);
    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, matrixBuffer, mpInput);
    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, histoBuffer, histEmpty);

    Cmn::createDescriptorPool(app.device, bindings, descriptorPool);
    Cmn::allocateDescriptorSet(app.device, descriptorSet, descriptorPool, descriptorSetLayout);
    Cmn::bindBuffers(app.device, matrixBuffer.buf, descriptorSet, 0);
    Cmn::bindBuffers(app.device, histoBuffer.buf, descriptorSet, 1);
}

void HistogramA3Task3Solution::compute()
{
    push.size=mpInput.size();
    uint32_t dx = (mpInput.size()+workGroupSize-1) / workGroupSize;
    vk::CommandBufferAllocateInfo allocInfo(
        app.computeCommandPool,                    // Commandpool 
        vk::CommandBufferLevel::ePrimary,          // ePrimary - Buffer you submit directly to queue. Cant be called by other buffers.
        1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cb.begin(beginInfo);
    cb.resetQueryPool(app.queryPool, 0, 2);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 0);
    cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute,
                     0, sizeof(PushConstant), &push);
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout,
                          0U, 1U, &descriptorSet, 0U, nullptr);
    cb.dispatch(dx, 1, 1);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 1);
    cb.end();

    // Submit the command buffer to the queue and set up a fence.
    vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb); // Submit a single command buffer
    vk::Fence fence = app.device.createFence(vk::FenceCreateInfo());         // Fence makes sure the control is not returned to CPU till command buffer is depleted

    app.computeQueue.submit({submitInfo}, fence);
    HostTimer timer;
    vk::Result haveIWaited = app.device.waitForFences({fence}, true, uint64_t(-1)); // Wait for the fence indefinitely
    app.device.destroyFence(fence);

    mstime = timer.elapsed() * 1000;

    uint64_t timestamps[2];
    vk::Result result = app.device.getQueryPoolResults(app.queryPool, 0, 2, sizeof(timestamps), &timestamps, sizeof(timestamps[0]), vk::QueryResultFlagBits::e64);
    assert(result == vk::Result::eSuccess);
    uint64_t timediff = timestamps[1] - timestamps[0];
    vk::PhysicalDeviceProperties properties = app.pDevice.getProperties();
    uint64_t nanoseconds = properties.limits.timestampPeriod * timediff;

    insideTime = nanoseconds / 1000000.f;
    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}

std::vector<int> HistogramA3Task3Solution::result() const
{
    std::vector<int> result(nBins, 0);
    fillHostWithStagingBuffer<int>(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, histoBuffer, result);
    return result;
}

void HistogramA3Task3Solution::cleanup()
{
    app.device.destroyDescriptorPool(descriptorPool);
    descriptorPool=VK_NULL_HANDLE;
    app.device.destroyPipeline(pipeline);
    pipeline=VK_NULL_HANDLE;
    app.device.destroyShaderModule(shaderModule);
    shaderModule=VK_NULL_HANDLE;
    app.device.destroyPipelineLayout(pipelineLayout);
    pipelineLayout=VK_NULL_HANDLE;
    app.device.destroyDescriptorSetLayout(descriptorSetLayout);
    descriptorSetLayout=VK_NULL_HANDLE;
    bindings.clear();

    auto Bclean = [&](Buffer &b)
    {
        app.device.destroyBuffer(b.buf);
        app.device.freeMemory(b.mem);
        b.buf=VK_NULL_HANDLE;
        b.mem=VK_NULL_HANDLE;
    };

    Bclean(matrixBuffer);
    Bclean(histoBuffer);
}