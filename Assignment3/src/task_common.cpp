#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "task_common.h"
#include "initialization.h"
#include "utils.h"

namespace Cmn{
    //We have a binding vector ready to become a descriptorSetLayout
void createDescriptorSetLayout(vk::Device &device,
                               std::vector<vk::DescriptorSetLayoutBinding> &bindings, vk::DescriptorSetLayout &descLayout)
{
    vk::DescriptorSetLayoutCreateInfo layoutInfo(
        {},
        CAST(bindings),     // Number of binding infos
        bindings.data()     // Array of binding infos
    );
    descLayout = device.createDescriptorSetLayout(layoutInfo);
}

void addStorage(std::vector<vk::DescriptorSetLayoutBinding> &bindings, uint32_t binding)
{
    //Bindings needed for DescriptorSetLayout
    //The DescriptorType eStorageBuffer is used in our case as storage buffer for compute shader
    //The ID binding(argument) is needed in the shader
    //DescriptorCount is set to 1U
    bindings.push_back(vk::DescriptorSetLayoutBinding(
        binding,                                          // The binding number of this entry
        vk::DescriptorType::eStorageBuffer,               // Type of resource descriptors used for this binding
        1U,                                               // Number of descriptors contained in the binding
        vk::ShaderStageFlagBits::eCompute)                // All defined shader stages can access the resource
    );
}

void allocateDescriptorSet(vk::Device &device, vk::DescriptorSet &descSet, vk::DescriptorPool &descPool,
                         vk::DescriptorSetLayout &descLayout)
{
    // You can technically allocate multiple layouts at once, we don't need that (so we put 1)
    vk::DescriptorSetAllocateInfo descAllocInfo(descPool, 1U, &descLayout);
    // Therefore the vector is length one, we want to take its (only) element    
    descSet = device.allocateDescriptorSets(descAllocInfo)[0];
}


//Binding our DescriptorSet to Buffer
//VK_WHOLE_SIZE is specified to bind the entire Buffer
//DescriptorType eStorageBuffer in our case should be coherant with DescriptorSetLayout
//WriteDescriptorSets(creates array) and updateDescriptorSets can be used only once
void bindBuffers(vk::Device &device, vk::Buffer &b, vk::DescriptorSet &set, uint32_t binding)
{
    // Buffer info and data offset info
    vk::DescriptorBufferInfo descInfo(
        b,                        // Buffer to get data from
        0ULL,                     // Position of start of data                  
        VK_WHOLE_SIZE             // Size of data
    );

    //     Binding index in the shader    V
    vk::WriteDescriptorSet write(set, binding, 0U, 1U,
                                 vk::DescriptorType::eStorageBuffer, nullptr, &descInfo);
    device.updateDescriptorSets(1U, &write, 0U, nullptr);
}

void createPipeline(vk::Device &device, vk::Pipeline &pipeline,
                    vk::PipelineLayout &pipLayout, vk::SpecializationInfo &specInfo, 
                    vk::ShaderModule &sModule)
{
    vk::PipelineShaderStageCreateInfo stageInfo(vk::PipelineShaderStageCreateFlags(),
                                                vk::ShaderStageFlagBits::eCompute, sModule,
                                                "main", &specInfo);

    vk::ComputePipelineCreateInfo computeInfo(vk::PipelineCreateFlags(), stageInfo, pipLayout);

    // This is a workaround: ideally there should not be a ".value"
    // This should be fixed in later releases of the SDK
    pipeline = device.createComputePipeline(nullptr, computeInfo, nullptr).value;
}

//Number of DescriptorSets is one by default
void createDescriptorPool(vk::Device &device,
                            std::vector<vk::DescriptorSetLayoutBinding> &bindings, vk::DescriptorPool &descPool, uint32_t numDescriptorSets)
{
    vk::DescriptorPoolSize descriptorPoolSize = vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, bindings.size() * numDescriptorSets);
    vk::DescriptorPoolCreateInfo descriptorPoolCI = vk::DescriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlags(), numDescriptorSets, 1U, &descriptorPoolSize);

    descPool = device.createDescriptorPool(descriptorPoolCI);
}


void createShader(vk::Device &device, vk::ShaderModule &shaderModule, const std::string &filename){    
    std::vector<char> cshader = readFile(filename);
    // Shader Module creation information
    vk::ShaderModuleCreateInfo smi(
        {},
        static_cast<uint32_t>(cshader.size()),                   // Size of code
        reinterpret_cast<const uint32_t*>( cshader.data() ));    // Pointer to code (of uint32_t pointer type)
    shaderModule = device.createShaderModule(smi);
}
}

void TaskResources::destroy(vk::Device &device)
{
    //Destroy all the resources we created in reverse order
   //Pipeline Should be destroyed before PipelineLayout
    device.destroyPipeline(this->pipeline);
    //PipelineLayout should be destroyed before DescriptorPool
    device.destroyPipelineLayout(this->pipelineLayout);
    //DescriptorPool should be destroyed before the DescriptorSetLayout
    device.destroyDescriptorPool(this->descriptorPool);
    device.destroyDescriptorSetLayout(this->descriptorSetLayout);
    device.destroyShaderModule(this->cShader);
    //The DescriptorSet does not need to be destroyed, It is managed by DescriptorPool.

    std::cout << std::endl
              << "destroyed everything successfully in task" << std::endl;
}

