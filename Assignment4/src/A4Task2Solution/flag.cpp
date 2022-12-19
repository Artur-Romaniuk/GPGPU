#include "flag.h"
#include <math.h>
#include "A4Task1.h"
#include "host_timer.h"
#define HALOSIZE 2

template <typename T, typename V>
T ceilDiv(T x, V y)
{
    return x / y + (x % y != 0);
}

FlagA4Task2Solution::FlagA4Task2Solution(AppResources &app, A4Task2Data &datad, uint32_t workGroupSize_x, uint32_t workGroupSize_y ) : 
    app(app), data(datad), workGroupSize_x(workGroupSize_x), workGroupSize_y(workGroupSize_y)
{

    // ########## Setup Pipelines ############ 
    // (nothing to do here)
    auto preparePipeline = [&](std::string shaderFile, vk::ShaderModule &sm, vk::Pipeline &pip, std::string name){
    app.device.destroyShaderModule(sm);

    Cmn::createShader(app.device, sm, shaderFile);
    setObjectName(app.device, sm, name+"_comp");

    // ### Create Pipeline ###
    std::array<vk::SpecializationMapEntry, 4> specEntries = std::array<vk::SpecializationMapEntry, 4>{
        {{0U, 0U, sizeof(uint32_t)},
        {1U, sizeof(uint32_t), sizeof(uint32_t)},
        {2U, 2*sizeof(uint32_t), sizeof(uint32_t)},
        {3U, 3*sizeof(uint32_t), sizeof(uint32_t)},
        } };
    std::array<uint32_t, 4> specValues = {workGroupSize_x, workGroupSize_y, HALOSIZE*2+workGroupSize_x, HALOSIZE*2+workGroupSize_y}; //for workgroup sizes

    vk::SpecializationInfo specInfo = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                                             CAST(specValues) * sizeof(uint32_t), specValues.data());
    // in case a pipeline was already created, destroy it
    app.device.destroyPipeline( pip );
    vk::PipelineShaderStageCreateInfo stageInfo(vk::PipelineShaderStageCreateFlags(),
                                            vk::ShaderStageFlagBits::eCompute, sm,
                                            "main", &specInfo);
    
    vk::ComputePipelineCreateInfo computeInfo(vk::PipelineCreateFlags(), stageInfo, data.layout);
    pip = app.device.createComputePipeline(nullptr, computeInfo, nullptr).value;
    setObjectName(app.device, pip, name+"_pip");
    };

    preparePipeline("./shaders/cloth_integrate.comp.spv", integrateShader, integratePipeline, "integrate");
    preparePipeline("./shaders/cloth_constraints.comp.spv", constraintsShader, constraintsPipeline, "constraints");
    preparePipeline("./shaders/cloth_collision.comp.spv", collisionShader, collisionPipeline, "collision");
    preparePipeline("./shaders/cloth_normals.comp.spv", normalsShader, normalsPipeline, "normals  ");

}

void FlagA4Task2Solution::compute()
{
    // ########### PREPARE DATA #############
    uint32_t dx = (data.WIDTH + workGroupSize_x - 1) / workGroupSize_x;
    uint32_t dy = (data.HEIGHT + workGroupSize_y - 1) / workGroupSize_y;
    data.currentSet = 0;
    static HostTimer ht; // this timer controls the speed of the simulation.
    // change the multiplier to adjust the speed
    float it_time= std::clamp(float(ht.elapsed()) , 0.f, 0.05f);
    
    data.push.dTs_rest = {
        it_time, // current iteration time
        data.push.dTs_rest[0], // old iteration time
        it_time + data.push.dTs_rest[2], // accumulate total time
        data.push.dTs_rest[3] // don't change this one
    };
    
    ht.reset();
    // ########### PREPARE COMMAND BUFFER #############
    vk::CommandBufferAllocateInfo allocInfo(
        app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    auto pipFence = [](vk::CommandBuffer &cb){
        cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags(),
        {vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead)},
        {},
        {}
    );};
    auto runKernel = [&](vk::Pipeline &p, vk::DescriptorSet &ds){   
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, p);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, data.layout,
                            0U, 1U, &(ds), 0U, nullptr);
        cb.pushConstants(data.layout, vk::ShaderStageFlagBits::eAll,
                        0, sizeof(A4Task2Data::PushConstant), &data.push);
        cb.dispatch(dx, dy, 1);
    };

    // ########### FILL COMMANDS #############
    cb.begin(beginInfo);
    cb.resetQueryPool(app.queryPool, 0, 2);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 0);
    // integrate
    runKernel(integratePipeline, data.computeDescriptorSet[0]);
    pipFence(cb);

    // ADD YOUR CODE HERE

	// Check for collisions
	
	// Constraint relaxation: use the ping-pong technique and perform the relaxation in several iterations
	//for (unsigned int i = 0; i < [enough iterations? twice the cloth size is an idea]; i++){
	//
	//	 Execute the constraint relaxation kernel
	//
	//	 if(i % 3 == 0)
	//		 Occasionally check for collisions
	//
	//	 Swap the ping pong buffers
    //  
	//}
    //
	// You can check for collisions here again, to make sure there is no intersection with the cloth in the end
    //
    // NOTE: two descriptor sets are already prepared with first two buffers inverted
    // to swap between them, you can use the data.currentSet value, changing its value everytime it is necessary
    // >> you need to use the out buffer of the constraints kernel in the collision kernel
    // make sure that after all is said and done, the correct computation is in gPos, binding 0 (for rendering and normals)
    data.currentSet = 0;
    for(uint i=0; i<2*dx*dy; i++){
        runKernel(constraintsPipeline, data.computeDescriptorSet[data.currentSet]);
        if(i%3==0){
            runKernel(collisionPipeline, data.computeDescriptorSet[data.currentSet]);
        }
        data.currentSet = (data.currentSet == 0) ? 1 : 0;
    }
    runKernel(collisionPipeline, data.computeDescriptorSet[data.currentSet]);
    
    // compute normals
    runKernel(normalsPipeline, data.computeDescriptorSet[data.currentSet]);

    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 1);
    cb.end();

    // submit the command buffer to the queue and set up a fence.

    vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb); // submit a single command buffer
    vk::Fence fence = app.device.createFence(vk::FenceCreateInfo());         // fence makes sure the control is not returned to CPU till command buffer is depleted

    app.computeQueue.submit({submitInfo}, fence);
    
    HostTimer timer; // this is a timer for the time taken by the computation

    vk::Result haveIWaited = app.device.waitForFences({fence}, true, uint64_t(-1)); // wait for the fence indefinitely
    app.device.destroyFence(fence);
    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);

    mstime = timer.elapsed() * 1000;
}

void FlagA4Task2Solution::prepare(){}

std::vector<float> FlagA4Task2Solution::result() const
{
    //std::cout<<w<<", "<<h<<std::endl;
    std::vector<float> resultPitched(1, 0.f);
    return resultPitched;
}

void FlagA4Task2Solution::cleanup()
{

    app.device.destroyPipeline(integratePipeline);
    app.device.destroyShaderModule(integrateShader);
    app.device.destroyPipeline(normalsPipeline);
    app.device.destroyShaderModule(normalsShader);
    app.device.destroyPipeline(collisionPipeline);
    app.device.destroyShaderModule(collisionShader);
    app.device.destroyPipeline(constraintsPipeline);
    app.device.destroyShaderModule(constraintsShader);
}