#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "task_common.h"
#include "initialization.h"
#include "utils.h"

namespace Cmn
{
    void createDescriptorSetLayout(vk::Device &device,
                                   std::vector<vk::DescriptorSetLayoutBinding> &bindings, vk::DescriptorSetLayout &descLayout)
    {
        //Uncomment for debugging
        //std::cout << "Creating DescriptorSetLayout..." << std::endl << std::endl;
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, CAST(bindings), bindings.data());
        descLayout = device.createDescriptorSetLayout(layoutInfo);
    }

    void addStorage(std::vector<vk::DescriptorSetLayoutBinding> &bindings, uint32_t binding)
    {
        bindings.push_back(vk::DescriptorSetLayoutBinding(
            binding, vk::DescriptorType::eStorageBuffer,
            1U, vk::ShaderStageFlagBits::eCompute));

    }

    void allocateDescriptorSet(vk::Device &device, vk::DescriptorSet &descSet, vk::DescriptorPool &descPool,
                               vk::DescriptorSetLayout &descLayout)
    {
        vk::DescriptorSetAllocateInfo descAllocInfo(descPool, 1U, &descLayout);
        descSet = device.allocateDescriptorSets(descAllocInfo)[0U];
    }

    void bindBuffers(vk::Device &device, vk::Buffer &b, vk::DescriptorSet &set, uint32_t binding)
    {
        vk::DescriptorBufferInfo descBuffInfo(b, 0UL, VK_WHOLE_SIZE);
        vk::WriteDescriptorSet write(set, binding, 0U, 1U, vk::DescriptorType::eStorageBuffer, nullptr, &descBuffInfo);
        device.updateDescriptorSets(1U, &write, 0U, nullptr);
    }

    void createDescriptorPool(vk::Device &device,
                              std::vector<vk::DescriptorSetLayoutBinding> &bindings, vk::DescriptorPool &descPool, uint32_t numDescriptorSets)
    {
        //std::cout << "DescriptorPool of size " << bindings.size() * numDescriptorSets <<" is created"<< std::endl << std::endl;
        vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer, bindings.size() * numDescriptorSets);
        vk::DescriptorPoolCreateInfo descriptorPoolCI(vk::DescriptorPoolCreateFlags(), numDescriptorSets, 1U, &descriptorPoolSize);
        descPool = device.createDescriptorPool(descriptorPoolCI);
    }

    void createPipeline(vk::Device &device, vk::Pipeline &pipeline,
                        vk::PipelineLayout &pipLayout, vk::SpecializationInfo &specInfo,
                        vk::ShaderModule &sModule)
    {
        std::cout << "createPipeline has not been defined yet, line: " << __LINE__ << " file " << __FILE__ << std::endl;
    }

    void createShader(vk::Device &device, vk::ShaderModule &shaderModule, const std::string &filename)
    {
        std::vector<char> cshader = readFile(filename);
        vk::ShaderModuleCreateInfo smi({}, static_cast<uint32_t>(cshader.size()), reinterpret_cast<const uint32_t*>(cshader.data()));
        shaderModule = device.createShaderModule(smi);
    }
}

void TaskResources::destroy(vk::Device &device)
{
   device.destroyPipeline(this->pipeline);
   device.destroyPipelineLayout(this->pipelineLayout);
   device.destroyDescriptorPool(this->descriptorPool);
   device.destroyDescriptorSetLayout(this->descriptorSetLayout);
   device.destroyShaderModule(this->cShader);
}
