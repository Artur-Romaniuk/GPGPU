#include "KernelDecomposition.h"

#include "host_timer.h"

A2Task1SolutionKernelDecomposition::A2Task1SolutionKernelDecomposition(AppResources &app, uint workGroupSize, std::string shaderFileName) :
    app(app), workGroupSize(workGroupSize), shaderFileName(shaderFileName) {}

void A2Task1SolutionKernelDecomposition::prepare(const std::vector<uint> &input)
{
    mpInput = &input;

    Cmn::addStorage(bindings, 0);
    Cmn::addStorage(bindings, 1);
    Cmn::createDescriptorSetLayout(app.device, bindings, descriptorSetLayout);
    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstant));
    vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &descriptorSetLayout, 1U, &pcr);
    pipelineLayout = app.device.createPipelineLayout(pipInfo);

    // Specialization constant for workgroup size
    std::array<vk::SpecializationMapEntry, 1> specEntries = std::array<vk::SpecializationMapEntry, 1>{ 
        {{0U, 0U, sizeof(workGroupSize)}},
    }; 
    std::array<uint32_t, 1> specValues = {workGroupSize}; //for workgroup sizes
    vk::SpecializationInfo specInfo = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                    CAST(specValues) * sizeof(int), specValues.data());

    Cmn::createShader(app.device, shaderModule, shaderFileName);
    Cmn::createPipeline(app.device, pipeline, pipelineLayout, specInfo, shaderModule);

    for (int i = 0; i < 2; i++) {
        createBuffer(app.pDevice, app.device, mpInput->size() * sizeof((*mpInput)[0]),
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, "buffer_" + std::to_string(i), buffers[i].buf, buffers[i].mem);
    }

    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, buffers[0], input);
    
    Cmn::createDescriptorPool(app.device, bindings, descriptorPool, 2);
    for (int i = 0; i < 2; i++)
        Cmn::allocateDescriptorSet(app.device, descriptorSets[i], descriptorPool, descriptorSetLayout);
    Cmn::bindBuffers(app.device, buffers[0].buf, descriptorSets[0], 0);
    Cmn::bindBuffers(app.device, buffers[1].buf, descriptorSets[0], 1);
    Cmn::bindBuffers(app.device, buffers[1].buf, descriptorSets[1], 0);
    Cmn::bindBuffers(app.device, buffers[0].buf, descriptorSets[1], 1);
}

void A2Task1SolutionKernelDecomposition::compute()
{
    vk::CommandBufferAllocateInfo allocInfo(
        app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers( allocInfo )[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cb.begin(beginInfo);

    // TO DO: Implement reduction with kernel decomposition
    // NOTE: make sure that activeBuffer points to the buffer with the final result in the end
    // That buffer is read back for the correctness check
    // (A2Task1SolutionKernelDecomposition::result())
    // HINT: You can alternate between the two provided descriptor sets to implement ping-pong

    cb.end();

    vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb);

    HostTimer timer;

    app.computeQueue.submit({submitInfo});
    app.device.waitIdle();

    mstime = timer.elapsed() * 1000;

    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}

uint A2Task1SolutionKernelDecomposition::result() const
{
    std::vector<uint> result(1, 0);
    fillHostWithStagingBuffer<uint>(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, buffers[activeBuffer], result);
    return result[0];
}

void A2Task1SolutionKernelDecomposition::cleanup()
{
    app.device.destroyDescriptorPool(descriptorPool);

    app.device.destroyPipeline(pipeline);
    app.device.destroyShaderModule(shaderModule);

    app.device.destroyPipelineLayout(pipelineLayout);
    app.device.destroyDescriptorSetLayout(descriptorSetLayout);
    bindings.clear();

    for (int i = 0; i < 2; i++)
        destroyBuffer(app.device, buffers[i]);
}