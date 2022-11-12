#include "KernelDecomposition.h"

#include "host_timer.h"

A2Task2SolutionKernelDecomposition::A2Task2SolutionKernelDecomposition(
    AppResources &app, uint workGroupSize):
    app(app), workGroupSize(workGroupSize) {}

void A2Task2SolutionKernelDecomposition::prepare(const std::vector<uint> &input) {
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
    std::array<uint32_t, 1> specValues = {workGroupSize}; //for workgroup sizes
    vk::SpecializationInfo specInfo = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                    CAST(specValues) * sizeof(int), specValues.data());

    // Local PPS Pipeline
    Cmn::createShader(app.device, cShaderLocalPPS, "./shaders/A2Task2KernelDecomposition.comp.spv");
    Cmn::createPipeline(app.device, pipelineLocalPPS, pipelineLayout, specInfo, cShaderLocalPPS);
    
    // Local PPS Offset Pipeline
    Cmn::createShader(app.device, cShaderLocalPPSOffset, "./shaders/A2Task2KernelDecompositionOffset.comp.spv");
    Cmn::createPipeline(app.device, pipelineLocalPPSOffset, pipelineLayout, specInfo, cShaderLocalPPSOffset);

    // ### create buffers, get their index in the task.buffers[] array ###
    using BFlag = vk::BufferUsageFlagBits;
    auto makeDLocalBuffer = [ this ](vk::BufferUsageFlags usage, vk::DeviceSize size, std::string name) -> Buffer
    {
        Buffer b;
        createBuffer(app.pDevice, app.device, size, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, name, b.buf, b.mem);
        return b;
    };

    inoutBuffers.push_back(makeDLocalBuffer(BFlag::eTransferDst | BFlag::eTransferSrc | BFlag::eStorageBuffer, input.size() * sizeof(uint32_t), "buffer_inout_0"));

    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inoutBuffers[0], input);

    // TO DO create additional buffers (by pushing into inoutBuffers) and descriptors (by pushing into descriptorSets)
    // You need to create an appropriately-sized DescriptorPool first
}

void A2Task2SolutionKernelDecomposition::compute() {
        vk::CommandBufferAllocateInfo allocInfo(
            app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
        vk::CommandBuffer cb = app.device.allocateCommandBuffers( allocInfo )[0];

        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        cb.begin(beginInfo);

        // TO DO: Implement efficient version of scan

        // Make sure that the local prefix sum works before you start experimenting with large arrays
        
        cb.end();

        vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb);

        HostTimer timer;

        app.computeQueue.submit({submitInfo});
        app.device.waitIdle();

        mstime = timer.elapsed() * 1000;

        app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}

std::vector<uint> A2Task2SolutionKernelDecomposition::result() const {
    std::vector<uint> result(workSize, 0);
    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inoutBuffers[0], result);
    return result;
}


void A2Task2SolutionKernelDecomposition::cleanup() {
    app.device.destroyDescriptorPool(descriptorPool);

    app.device.destroyPipeline(pipelineLocalPPSOffset);
    app.device.destroyShaderModule(cShaderLocalPPSOffset);

    app.device.destroyPipeline(pipelineLocalPPS);
    app.device.destroyShaderModule(cShaderLocalPPS);

    app.device.destroyPipelineLayout(pipelineLayout);
    app.device.destroyDescriptorSetLayout(descriptorSetLayout);
    bindings.clear();

    auto Bclean = [&](Buffer &b){
        app.device.destroyBuffer(b.buf);
        app.device.freeMemory(b.mem);};

    for (auto inoutBuffer : inoutBuffers) {
        Bclean(inoutBuffer);
    }
}
