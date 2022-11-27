#include "convsep.h"
#include "host_timer.h"
#include <math.h> 
#include <numeric> //std::accumulate

template<typename T, typename V>
T ceilDiv(T x, V y) {
    return x / y + (x % y != 0);
}

SeparableA3Task2Solution::SeparableA3Task2Solution(AppResources &app, uint32_t workGrpSize_H[2], uint32_t workGrpSize_V[2]):
    app(app) {
    workGroupSize_H[0] = workGrpSize_H[0];
    workGroupSize_H[1] = workGrpSize_H[1];
    workGroupSize_V[0] = workGrpSize_V[0];
    workGroupSize_V[1] = workGrpSize_V[1];
}

void SeparableA3Task2Solution::prepare(
    const std::vector<float> &input, 
    const std::vector<float> horizKer, const std::vector<float> vertKer, 
    uint32_t width, uint32_t height,
    uint32_t H_ELEMENTS, uint32_t V_ELEMENTS)
{
    H_elements = H_ELEMENTS;
    V_elements = V_ELEMENTS;
    kernelHoriz=horizKer;
    kernelVert=vertKer;
    float kSumH = std::accumulate(kernelHoriz.begin(), kernelHoriz.end(), 0.f);
    float kSumV = std::accumulate(kernelVert.begin(), kernelVert.end(), 0.f);
    float kernelWeight = 1.f / sqrt(kSumH * kSumV);
    std::transform(kernelHoriz.begin(), kernelHoriz.end(), kernelHoriz.begin(), [kernelWeight](float& v) {return v*kernelWeight;});
    std::transform(kernelVert.begin(), kernelVert.end(), kernelVert.begin(), [kernelWeight](float& v) {return v*kernelWeight;});

    // std::cout<< "sumH = " <<kWeightH << "\tsumV = "<<kWeightV <<"\tkernelWeights = "<<kernelWeight <<std::endl;

    uint32_t radH = (horizKer.size()-1)/2;
    uint32_t radV = (vertKer.size()-1)/2;
    w=p=width;
    h=height;
    // if width is not a multiple of the transaction size (32 floats)
    // -> resize
    if(w%32 != 0)
        p = w + 32 - (w%32);

    imgInput.resize(p*h,0.f);
    for (int j = 0 ; j < h; j++)
        memcpy(imgInput.data()+p*j, input.data()+w*j, w*sizeof(float));
    
    cleanup();

    Cmn::addStorage(bindings, 0); 
    Cmn::addStorage(bindings, 1);
    Cmn::createDescriptorSetLayout(app.device, bindings, descriptorSetLayout);
    
    vk::PushConstantRange pcrH(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstant));
    vk::PushConstantRange pcrV(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstant));

    vk::PipelineLayoutCreateInfo pipInfoH(vk::PipelineLayoutCreateFlags(), 1U, &descriptorSetLayout, 1U, &pcrH);
    pipelineLayoutHoriz = app.device.createPipelineLayout(pipInfoH);
    vk::PipelineLayoutCreateInfo pipInfoV(vk::PipelineLayoutCreateFlags(), 1U, &descriptorSetLayout, 1U, &pcrV);
    pipelineLayoutVert = app.device.createPipelineLayout(pipInfoV);

    Cmn::createShader(app.device, shaderModuleHoriz, "./shaders/conv_horizontal.comp.spv");
    Cmn::createShader(app.device, shaderModuleVert, "./shaders/conv_vertical.comp.spv");

    // Specialization constant HORIZONTAL 
    std::array<vk::SpecializationMapEntry, 4> specEntries = std::array<vk::SpecializationMapEntry, 4>{ 
        {{0U, 0U, sizeof(uint32_t)}, // workgroup size x
        {1U, 1*sizeof(uint32_t), sizeof(uint32_t)}, // workgroup size y
        {2U, 2*sizeof(uint32_t), sizeof(uint32_t)}, // kernel radius H
        {3U, 3*sizeof(uint32_t), sizeof(uint32_t)}}, // n elements treated per thread
    }; 

    std::array<uint32_t, 4> specValuesH = {workGroupSize_H[0], workGroupSize_H[1], radH, H_ELEMENTS}; //for workgroup sizes

    vk::SpecializationInfo specInfoH = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                    CAST(specValuesH) * sizeof(int), specValuesH.data());
                                    
    Cmn::createPipeline(app.device, pipelineHoriz, pipelineLayoutHoriz, specInfoH, shaderModuleHoriz);
    
    // Specialization constant VERTICAL
    std::cout << "grps H = {" << workGroupSize_H[0 ] <<","<<workGroupSize_H[1]<<"}"<<std::endl;
    std::cout << "grps V = {" << workGroupSize_V[0 ] <<","<<workGroupSize_V[1]<<"}"<<std::endl;
    std::array<uint32_t, 4> specValuesV = {workGroupSize_V[0], workGroupSize_V[1], radV, V_ELEMENTS}; //for workgroup sizes
    vk::SpecializationInfo specInfoV = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                    CAST(specValuesV) * sizeof(int), specValuesV.data());
                                    
    Cmn::createPipeline(app.device, pipelineVert, pipelineLayoutVert, specInfoV, shaderModuleVert);

    //input buffer will be the output of the second shader
    createBuffer(app.pDevice, app.device, imgInput.size() * sizeof(imgInput[0]),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal, "inOutBuf", inOutBuf.buf, inOutBuf.mem);

    createBuffer(app.pDevice, app.device, imgInput.size() * sizeof(imgInput[0]),
        vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, 
        "interBuf", interBuf.buf, interBuf.mem);

    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inOutBuf, input);
    
    Cmn::createDescriptorPool(app.device, bindings, descriptorPool);
    Cmn::allocateDescriptorSet(app.device, descriptorSet, descriptorPool, descriptorSetLayout);
    Cmn::bindBuffers(app.device, inOutBuf.buf, descriptorSet, 0);
    Cmn::bindBuffers(app.device, interBuf.buf, descriptorSet, 1);
    pushHoriz.width=w;
    pushHoriz.height=h;
    pushHoriz.pitch=p;
    pushHoriz.kernelWeight=kernelWeight;

    memcpy(pushHoriz.kernel, kernelHoriz.data(), sizeof(float)*kernelHoriz.size());

    pushVert.width=w;
    pushVert.height=h;
    pushVert.pitch=p;
    pushVert.kernelWeight=kernelWeight;
    memcpy(pushVert.kernel, kernelVert.data(), sizeof(float)*kernelVert.size());
}

void SeparableA3Task2Solution::compute()
{
    // We are computing the dispatch sizes.
    // Can you explain these computations?
    uint32_t dxh = (w + workGroupSize_H[0]*H_elements - 1) / (workGroupSize_H[0]*H_elements);
    uint32_t dyh = (h + workGroupSize_H[1] - 1) / workGroupSize_H[1];
    uint32_t dxv = (w + workGroupSize_V[0] - 1) / (workGroupSize_V[0]);
    uint32_t dyv = (h + workGroupSize_V[1]*V_elements - 1) / (workGroupSize_V[1]*V_elements);
    vk::CommandBufferAllocateInfo allocInfo(
        app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers( allocInfo )[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cb.begin(beginInfo);
    cb.resetQueryPool(app.queryPool, 0, 2);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 0);
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineHoriz);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayoutHoriz,
                        0U, 1U, &descriptorSet, 0U, nullptr);
    cb.pushConstants(pipelineLayoutVert, vk::ShaderStageFlagBits::eCompute,
                     0, sizeof(PushConstant), &pushHoriz);
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags(),
        {vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead)},
        {},
        {}
    );
    cb.dispatch(dxh, dyh, 1);
    
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineVert);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayoutVert,
                        0U, 1U, &descriptorSet, 0U, nullptr);
    cb.pushConstants(pipelineLayoutVert, vk::ShaderStageFlagBits::eCompute,
                     0, sizeof(PushConstant), &pushVert);
    cb.dispatch(dxv, dyv, 1);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 1);
    cb.end();

    // submit the command buffer to the queue and set up a fence.
    vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb); // submit a single command buffer

    HostTimer timer;

    app.computeQueue.submit({submitInfo});
    app.device.waitIdle();

    mstime = timer.elapsed() * 1000;

    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);

    uint64_t timestamps[2];
    vk::Result result = app.device.getQueryPoolResults(app.queryPool, 0, 2, sizeof(timestamps), &timestamps, sizeof(timestamps[0]), vk::QueryResultFlagBits::e64);
    assert(result == vk::Result::eSuccess);
    uint64_t timediff = timestamps[1] - timestamps[0];
    vk::PhysicalDeviceProperties properties = app.pDevice.getProperties();
    uint64_t nanoseconds = properties.limits.timestampPeriod * timediff;
    insideTime = nanoseconds / 1000000.f;
}

std::vector<float> SeparableA3Task2Solution::result() const
{   
    std::cout<<w<<", "<<h<<std::endl;
    std::vector<float> resultPitched(p*h, 0.);
    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inOutBuf, resultPitched);

    std::vector<float> result(w*h, 0.);
    // re-align to remove the pitch
    for(int i = 0; i<h; i++)
        memcpy(result.data()+w*i,resultPitched.data()+p*i,w*sizeof(float));
    return result;
}

void SeparableA3Task2Solution::cleanup()
{
    app.device.destroyDescriptorPool(descriptorPool);

    app.device.destroyPipeline(pipelineHoriz);
    app.device.destroyPipeline(pipelineVert);

    app.device.destroyShaderModule(shaderModuleVert);
    app.device.destroyShaderModule(shaderModuleHoriz);

    app.device.destroyPipelineLayout(pipelineLayoutVert);
    app.device.destroyPipelineLayout(pipelineLayoutHoriz);
    app.device.destroyDescriptorSetLayout(descriptorSetLayout);
    bindings.clear();

    auto Bclean = [&](Buffer &b){
        app.device.destroyBuffer(b.buf);
        app.device.freeMemory(b.mem);
    };

    Bclean(inOutBuf);
    Bclean(interBuf);
}