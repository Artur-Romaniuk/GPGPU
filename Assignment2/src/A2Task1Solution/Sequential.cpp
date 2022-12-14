#include "Sequential.h"

#include "host_timer.h"

A2Task1SolutionSequential::A2Task1SolutionSequential(AppResources &app, uint workGroupSize) :
    app(app), workGroupSize(workGroupSize) {}

void A2Task1SolutionSequential::prepare(const std::vector<uint> &input)
{
    mpInput = &input;

    Cmn::addStorage(bindings, 0);
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

    Cmn::createShader(app.device, shaderModule, "./shaders/A2Task1Sequential.comp.spv");
    Cmn::createPipeline(app.device, pipeline, pipelineLayout, specInfo, shaderModule);

    createBuffer(app.pDevice, app.device, mpInput->size() * sizeof((*mpInput)[0]),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal, "inoutBuffer", inoutBuffer.buf, inoutBuffer.mem);

    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inoutBuffer, input);
    
    Cmn::createDescriptorPool(app.device, bindings, descriptorPool);
    Cmn::allocateDescriptorSet(app.device, descriptorSet, descriptorPool, descriptorSetLayout);
    Cmn::bindBuffers(app.device, inoutBuffer.buf, descriptorSet, 0);
}

void A2Task1SolutionSequential::compute()
{
    vk::CommandBufferAllocateInfo allocInfo(
        app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers( allocInfo )[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    uint32_t groupCount = (  mpInput->size()/2 + workGroupSize -1) / workGroupSize;
    PushConstant pushConstant{
        .size = static_cast<uint32_t>(mpInput->size()/2),
        .offset = pushConstant.size
    };

    cb.begin(beginInfo);

    // TO DO: Implement reduction with sequential addressing
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0U, 1U, &descriptorSet, 0U, nullptr);


    while(pushConstant.size>=1U){
        cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstant), &pushConstant);
        cb.dispatch(groupCount, 1, 1);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(), {vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderWrite)},{},{});
        pushConstant.size/=2U;
        pushConstant.offset/=2U;
        groupCount = (pushConstant.size + workGroupSize -1) / workGroupSize; 
    }

    cb.end();

    vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb);

    HostTimer timer;

    app.computeQueue.submit({submitInfo});
    app.device.waitIdle();

    mstime = timer.elapsed() * 1000;

    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}

uint A2Task1SolutionSequential::result() const
{
    std::vector<uint> result(1, 0);
    fillHostWithStagingBuffer<uint>(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inoutBuffer, result);
    return result[0];
}

void A2Task1SolutionSequential::cleanup()
{
    app.device.destroyDescriptorPool(descriptorPool);

    app.device.destroyPipeline(pipeline);
    app.device.destroyShaderModule(shaderModule);

    app.device.destroyPipelineLayout(pipelineLayout);
    app.device.destroyDescriptorSetLayout(descriptorSetLayout);
    bindings.clear();

    auto Bclean = [&](Buffer &b){
        app.device.destroyBuffer(b.buf);
        app.device.freeMemory(b.mem);
    };

    Bclean(inoutBuffer);
}