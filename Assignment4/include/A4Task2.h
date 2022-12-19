#pragma once

#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include "stb_image.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include "tiny_obj_loader.h"
#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"
#include "render.h"

struct A4Task2Data {

    struct PushConstant {
        glm::mat4 mvp;
        glm::uvec4 parts_w_h; // nParticles, width, height
        glm::vec4 dTs_rest; // timestep, previous timestep, total time, resting distance
        glm::vec4 ballPos; // x,y,z, size
    };
    
    uint32_t WIDTH=64;
    uint32_t HEIGHT=64;
    PushConstant push;
    uint32_t particleCount, triangleCount, sphereTriangleCount;

    glm::vec4 spherePos; // w contains size

    vk::Sampler textureSampler;
    vk::Image textureImage;
    vk::ImageView textureView;
    vk::DeviceMemory textureImageMemory;

    Buffer gPosArray, gOldPosArray, gAuxPosArray, 
           gNormals, gTex, gIndices, gTriangleSoup, gSoupNormals, gSpherePositions, gSphereNormals;

    // Descriptor & Pipeline Layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout layout;

    // Descriptor Pool
    vk::DescriptorPool descriptorPool;

    // Per-dispatch data
    uint32_t currentSet;
    vk::DescriptorSet descriptorSet, computeDescriptorSet[2];
};  

class A4Task2Solution {
public:
    virtual void prepare() = 0;
    virtual void compute() = 0;
    virtual std::vector<float> result() const = 0;
};

class A4Task2Render;


class A4Task2Render {
public:
    AppResources &app;
    Render &render;

    vk::Pipeline opaquePipeline;
    vk::Pipeline spherePipeline;
    vk::Pipeline clothPipeline;
    
    A4Task2Render(AppResources &app, Render &render): app(app), render(render) {};
    void prepare(A4Task2Data &data) {
        { // Opaque
            vk::ShaderModule vertexM, fragmentM;
            // Create Shader Modules
            Cmn::createShader(app.device, vertexM, "./shaders/soup.vert.spv");
            Cmn::createShader(app.device, fragmentM, "./shaders/phong2.frag.spv");

            // Put shader stage creation info in to array
            // Graphics Pipeline creation info requires array of shader stage creates

            vk::PipelineShaderStageCreateInfo shaderStageCI[] = {
                {{}, vk::ShaderStageFlagBits::eVertex, vertexM, "main", nullptr},
                {{}, vk::ShaderStageFlagBits::eFragment, fragmentM, "main", nullptr},
            };

            // Vertex input
            vk::PipelineVertexInputStateCreateInfo vertexInputSCI = {
                {},
                0,                       // Vertex binding description  count
                nullptr,                 // List of Vertex Binding Descriptions (data spacing/stride information)
                0,                       // Vertex attribute description count
                nullptr                  // List of Vertex Attribute Descriptions (data format and where to bind to/from)
            };
            // Input Assembly
            vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI = {
                {}, 
                vk::PrimitiveTopology::eTriangleList,    // Primitive type to assemble vertices as
                false                                    // Allow overriding of "strip" topology to start new
            };
            // Viewport & Scissor 

            vk::Viewport viewport = {
                0.f,                                // x start coordinate
                (float)app.extent.height,           // y start coordinate
                (float)app.extent.width,            // Width of viewport
                -(float)app.extent.height,          // Height of viewport
                0.f,                                // Min framebuffer depth,
                1.f                                 // Max framebuffer depth
            };
            vk::Rect2D scissor = {
                {0, 0},                             // Offset to use region from
                app.extent                          // Extent to describe region to use, starting at offset
            };
            // Viewport create info
            vk::PipelineViewportStateCreateInfo viewportSCI = {
                {},
                1,              // Viewport count
                &viewport,      // Viewport used
                1,              // Scissor count
                &scissor        // Scissor used
            };

            // Rasterizer
            vk::PipelineRasterizationStateCreateInfo rasterizationSCI = {
                {},
                false,                                // Change if fragments beyond near/far planes are clipped (default) or clamped to plane
                false,                                // Whether to discard data and skip rasterizer. Never creates fragments, only suitable for pipeline without framebuffer output
                vk::PolygonMode::eFill,               // How to handle filling points between vertices
                vk::CullModeFlagBits::eNone,          // Which face of a tri to cull
                vk::FrontFace::eCounterClockwise,     // Winding to determine which side is front
                false,                                // Whether to add depth bias to fragments (good for stopping "shadow acne" in shadow mapping)
                0.f,
                0.f,
                0.f,
                1.f                                   // How thick lines should be when drawn
            };
            //  Multisampling
            vk::PipelineMultisampleStateCreateInfo multisampleSCI = {
                {},
                vk::SampleCountFlagBits::e1,      // Number of samples to use per fragment
                false,                            // Enable multisample shading or not
                0.f,
                nullptr,
                false,
                false
            };

            // Depth stencil creation
            vk::PipelineDepthStencilStateCreateInfo depthStencilSCI = {
                {}, true, true, vk::CompareOp::eLess, false, false, {}, {}, 0.f, 0.f
            };


            // -- BLENDING --
            // Blending decides how to blend a new colour being written to a fragment, with the old value
            // Blend Attachment State (how blending is handled)
            vk::PipelineColorBlendAttachmentState colorBlendAttachmentState = { false };
            // Colours to apply blending to
            colorBlendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
            vk::PipelineColorBlendStateCreateInfo colorBlendSCI = {
                {}, 
                false,                                     //logicOpEnable - Alternative to calculations is to use logical operations
                {}, 
                1,
                &colorBlendAttachmentState,                //pAttachments
                {}
            };
            vk::GraphicsPipelineCreateInfo cI = {
                {},
                2,                               // Number of shader stages
                &shaderStageCI[0],               // List of shader stages
                &vertexInputSCI,                 // All the fixed function pipeline states
                &inputAssemblySCI,
                nullptr,
                &viewportSCI,
                &rasterizationSCI, 
                &multisampleSCI,
                &depthStencilSCI,
                &colorBlendSCI,
                nullptr,
                data.layout,                     // Pipeline Layout pipeline should use                    
                render.renderPass,               // Render pass description the pipeline is compatible with
                0,                               // Subpass of render pass to use with pipeline
                // Pipeline Derivatives : Can create multiple pipelines that derive from one another for optimisation
                {},                              //basePipelineHandle - Existing pipeline to derive from...
                0                                //basePipelineIndex - or index of pipeline being created to derive from (in case creating multiple at once)
            };
            // Create Graphics Pipeline
            auto pipelines = app.device.createGraphicsPipelines(VK_NULL_HANDLE, {cI});
            if (pipelines.result != vk::Result::eSuccess)
                throw std::runtime_error("Pipeline creation failed");
            opaquePipeline = pipelines.value[0];

            app.device.destroyShaderModule(vertexM);
            app.device.destroyShaderModule(fragmentM);
        }
        { // Sphere
            vk::ShaderModule vertexM, fragmentM;
            // Read in SPIR-V code of shaders
            Cmn::createShader(app.device, vertexM, "./shaders/sphere.vert.spv");
            Cmn::createShader(app.device, fragmentM, "./shaders/phong2.frag.spv");

            // Put shader stage creation info in to array
            // Graphics Pipeline creation info requires array of shader stage creates
            vk::PipelineShaderStageCreateInfo shaderStageCI[] = {
                {{}, vk::ShaderStageFlagBits::eVertex, vertexM, "main", nullptr},
                {{}, vk::ShaderStageFlagBits::eFragment, fragmentM, "main", nullptr},
            };

            // Vertex input
            vk::PipelineVertexInputStateCreateInfo vertexInputSCI = {
                {},
                0,                       // Vertex binding description  count
                nullptr,                 // List of Vertex Binding Descriptions (data spacing/stride information)
                0,                       // Vertex attribute description count
                nullptr                  // List of Vertex Attribute Descriptions (data format and where to bind to/from)
            };

            // Input Assembly
            vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI = {
                {}, vk::PrimitiveTopology::eTriangleList, false
            };
            // Viewport & Scissor 
            vk::Viewport viewport = {
                0.f,                                // x start coordinate
                (float)app.extent.height,           // y start coordinate
                (float)app.extent.width,            // Width of viewport
                -(float)app.extent.height,          // Height of viewport
                0.f,                                // Min framebuffer depth,
                1.f                                 // Max framebuffer depth
            };
            vk::Rect2D scissor = {
                {0, 0},                             // Offset to use region from
                app.extent                          // Extent to describe region to use, starting at offset
            };
            // Viewport create info
            vk::PipelineViewportStateCreateInfo viewportSCI = {
                {},
                1,              // Viewport count
                &viewport,      // Viewport used
                1,              // Scissor count
                &scissor        // Scissor used
            };
            // Rasterizer
            vk::PipelineRasterizationStateCreateInfo rasterizationSCI = {
                {},
                false,                                // Change if fragments beyond near/far planes are clipped (default) or clamped to plane
                false,                                // Whether to discard data and skip rasterizer. Never creates fragments, only suitable for pipeline without framebuffer output
                vk::PolygonMode::eFill,               // How to handle filling points between vertices
                vk::CullModeFlagBits::eNone,          // Which face of a tri to cull
                vk::FrontFace::eCounterClockwise,     // Winding to determine which side is front
                false,                                // Whether to add depth bias to fragments (good for stopping "shadow acne" in shadow mapping)
                0.f,
                0.f,
                0.f,
                1.f                                   // How thick lines should be when drawn
            };
            vk::PipelineMultisampleStateCreateInfo multisampleSCI = {
                {},
                vk::SampleCountFlagBits::e1,      // Number of samples to use per fragment
                false,                            // Enable multisample shading or not
                0.f,
                nullptr,
                false,
                false,
            };
            // Depth stencil creation
            vk::PipelineDepthStencilStateCreateInfo depthStencilSCI = {
                {}, true, true, vk::CompareOp::eLess, false, false, {}, {}, 0.f, 0.f
            };

            // -- BLENDING --
            // Blending decides how to blend a new colour being written to a fragment, with the old value
            // Blend Attachment State (how blending is handled)
            vk::PipelineColorBlendAttachmentState colorBlendAttachmentState = { false };
            // Colours to apply blending to
            colorBlendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
            vk::PipelineColorBlendStateCreateInfo colorBlendSCI = {
                {},
                false,                                     //logicOpEnable - Alternative to calculations is to use logical operations
                {},
                1,
                &colorBlendAttachmentState,                //pAttachments
                {}
            };
            // Graphics Pipeline Creation
            vk::GraphicsPipelineCreateInfo cI = {
                {},
                2,                               // Number of shader stages
                &shaderStageCI[0],               // List of shader stages
                &vertexInputSCI,                 // All the fixed function pipeline states
                &inputAssemblySCI,
                nullptr,
                &viewportSCI,
                &rasterizationSCI,
                &multisampleSCI,
                &depthStencilSCI,
                &colorBlendSCI,
                nullptr,
                data.layout,                     // Pipeline Layout pipeline should use                    
                render.renderPass,               // Render pass description the pipeline is compatible with
                0,                               // Subpass of render pass to use with pipeline
                // Pipeline Derivatives : Can create multiple pipelines that derive from one another for optimisation
                {},                              //basePipelineHandle - Existing pipeline to derive from...
                0                                //basePipelineIndex - or index of pipeline being created to derive from (in case creating multiple at once)
            };

            // Create Graphics Pipeline
            auto pipelines = app.device.createGraphicsPipelines(VK_NULL_HANDLE, {cI});
            if (pipelines.result != vk::Result::eSuccess)
                throw std::runtime_error("Pipeline creation failed");
            spherePipeline = pipelines.value[0];

            // Destroy Shader Modules, no longer needed after Pipeline created 
            app.device.destroyShaderModule(vertexM);
            app.device.destroyShaderModule(fragmentM);
        }
        { // Cloth
            vk::ShaderModule vertexM, fragmentM;
            Cmn::createShader(app.device, vertexM, "./shaders/cloth.vert.spv");
            Cmn::createShader(app.device, fragmentM, "./shaders/cloth.frag.spv");

            // Put shader stage creation info in to array
            // Graphics Pipeline creation info requires array of shader stage creates
            vk::PipelineShaderStageCreateInfo shaderStageCI[] = {
                {{}, vk::ShaderStageFlagBits::eVertex, vertexM, "main", nullptr},
                {{}, vk::ShaderStageFlagBits::eFragment, fragmentM, "main", nullptr},
            };

            // Vertex input
            vk::PipelineVertexInputStateCreateInfo vertexInputSCI = {
                {},
                0,                       // Vertex binding description  count
                nullptr,                 // List of Vertex Binding Descriptions (data spacing/stride information)
                0,                       // Vertex attribute description count
                nullptr                  // List of Vertex Attribute Descriptions (data format and where to bind to/from)
            };

            // Input Assembly
            vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI = {
                {}, vk::PrimitiveTopology::eTriangleList, false
            };
            // Viewport & Scissor 
            vk::Viewport viewport = {
                0.f,                                // x start coordinate
                (float)app.extent.height,           // y start coordinate
                (float)app.extent.width,            // Width of viewport
                -(float)app.extent.height,          // Height of viewport
                0.f,                                // Min framebuffer depth,
                1.f                                 // Max framebuffer depth
            };
            vk::Rect2D scissor = {
                {0, 0},                             // Offset to use region from
                app.extent                          // Extent to describe region to use, starting at offset
            };
            // Viewport create info
            vk::PipelineViewportStateCreateInfo viewportSCI = {
                {},
                1,              // Viewport count
                &viewport,      // Viewport used
                1,              // Scissor count
                &scissor        // Scissor used
            };
            // Rasterizer
            vk::PipelineRasterizationStateCreateInfo rasterizationSCI = {
                {},
                false,                                // Change if fragments beyond near/far planes are clipped (default) or clamped to plane
                false,                                // Whether to discard data and skip rasterizer. Never creates fragments, only suitable for pipeline without framebuffer output
                vk::PolygonMode::eFill,               // How to handle filling points between vertices
                vk::CullModeFlagBits::eNone,          // Which face of a tri to cull
                vk::FrontFace::eCounterClockwise,     // Winding to determine which side is front
                false,                                // Whether to add depth bias to fragments (good for stopping "shadow acne" in shadow mapping)
                0.f,
                0.f,
                0.f,
                1.f                                   // How thick lines should be when drawn
            };
            vk::PipelineMultisampleStateCreateInfo multisampleSCI = {
                {},
                vk::SampleCountFlagBits::e1,      // Number of samples to use per fragment
                false,                            // Enable multisample shading or not
                0.f,
                nullptr,
                false,
                false,
            };
            // Depth stencil creation
            vk::PipelineDepthStencilStateCreateInfo depthStencilSCI = {
                {}, true, true, vk::CompareOp::eLess, false, false, {}, {}, 0.f, 0.f
            };

            // -- BLENDING --
            // Blending decides how to blend a new colour being written to a fragment, with the old value
            // Blend Attachment State (how blending is handled)
            vk::PipelineColorBlendAttachmentState colorBlendAttachmentState = { false };
            // Colours to apply blending to
            colorBlendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
            vk::PipelineColorBlendStateCreateInfo colorBlendSCI = {
                {},
                false,                                     //logicOpEnable - Alternative to calculations is to use logical operations
                {},
                1,
                &colorBlendAttachmentState,                //pAttachments
                {}
            };
            vk::GraphicsPipelineCreateInfo cI = {
                {},
                2,                               // Number of shader stages
                &shaderStageCI[0],               // List of shader stages
                &vertexInputSCI,                 // All the fixed function pipeline states
                &inputAssemblySCI,
                nullptr,
                &viewportSCI,
                &rasterizationSCI,
                &multisampleSCI,
                &depthStencilSCI,
                &colorBlendSCI,
                nullptr,
                data.layout,                     // Pipeline Layout pipeline should use                    
                render.renderPass,               // Render pass description the pipeline is compatible with
                0,                               // Subpass of render pass to use with pipeline
                // Pipeline Derivatives : Can create multiple pipelines that derive from one another for optimisation
                {},                              //basePipelineHandle - Existing pipeline to derive from...
                0                                //basePipelineIndex - or index of pipeline being created to derive from (in case creating multiple at once)
            };

            // Create Graphics Pipeline
            auto pipelines = app.device.createGraphicsPipelines(VK_NULL_HANDLE, {cI});
            if (pipelines.result != vk::Result::eSuccess)
                throw std::runtime_error("Pipeline creation failed");
            clothPipeline = pipelines.value[0];

            app.device.destroyShaderModule(vertexM);
            app.device.destroyShaderModule(fragmentM);
        }
    }

    void destroy() {
        app.device.destroyPipeline(opaquePipeline);
        app.device.destroyPipeline(spherePipeline);
        app.device.destroyPipeline(clothPipeline);
    }

    void renderFrame(A4Task2Data &data) {
        render.renderFrame(
        [&](vk::CommandBuffer &cb) {
            {
                cb.bindPipeline(vk::PipelineBindPoint::eGraphics, opaquePipeline);
                A4Task2Data::PushConstant pC;
                pC.mvp = render.camera.viewProjectionMatrix();
                cb.pushConstants(data.layout, vk::ShaderStageFlagBits::eAll, 0, sizeof(pC), &pC);
                cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, data.layout, 0, 1, &data.computeDescriptorSet[0], 0, nullptr);
                cb.draw(3 * data.triangleCount, 1, 0, 0);
            }
            {
                cb.bindPipeline(vk::PipelineBindPoint::eGraphics, spherePipeline);
                A4Task2Data::PushConstant pC = data.push;
                pC.mvp = render.camera.viewProjectionMatrix();
                pC.ballPos = data.push.ballPos;
                cb.pushConstants(data.layout, vk::ShaderStageFlagBits::eAll, 0, sizeof(pC), &pC);
                // Bind Descriptor Sets
                cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, data.layout, 0, 1, &data.computeDescriptorSet[0], 0, nullptr);
                // Execute pipeline
                cb.draw(3 * data.sphereTriangleCount, 1, 0, 0);
            }
            {
                cb.bindPipeline(vk::PipelineBindPoint::eGraphics, clothPipeline);
                A4Task2Data::PushConstant pC;
                pC.mvp = render.camera.viewProjectionMatrix();
                cb.pushConstants(data.layout, vk::ShaderStageFlagBits::eAll, 0, sizeof(pC), &pC);
                // Bind Descriptor Sets
                cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, data.layout, 0, 1, &data.computeDescriptorSet[0], 0, nullptr);
                // Execute pipeline
                cb.draw(3 * 2 * data.WIDTH * data.HEIGHT, 1, 0, 0);
            }
        },
        [&](vk::CommandBuffer &cb) { });
    }

};

class A4Task2 {
    using vec4 = std::array<float,4>;
    using arrayVec4 = std::vector<vec4> ;
    AppResources &app;

   
    A4Task2Render render;

public:
    A4Task2Data data;


    A4Task2(AppResources &app, Render &render) : app(app), render(app, render)
    {
        make2DTexture(); // for the texture on the flag
        makeSampler(); // same

        // this is for the pole
        
        arrayVec4 vertices, normals, soupNormals;
        std::vector<uint32_t> indices;
        std::vector<std::array<float,2>> tex;
        arrayVec4 tSoup = getTriangles("../Assets/clothscene.obj", soupNormals); 
        data.triangleCount = tSoup.size() / 3; 
        createPlane(data.WIDTH, data.HEIGHT, indices, vertices, normals, tex);
        data.particleCount=data.WIDTH*data.HEIGHT;

        arrayVec4 sphereNormals;
        arrayVec4 spherePositions = getTriangles("../Assets/sphere.obj", sphereNormals);
        data.sphereTriangleCount = spherePositions.size() / 3;

        using BFlag = vk::BufferUsageFlagBits;
        auto makeDLocalBuffer = [&](vk::BufferUsageFlags usage, vk::DeviceSize size, std::string name) -> Buffer {
            Buffer b;
            createBuffer(app.pDevice, app.device, size, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, name, b.buf, b.mem);
            return b;
        };

        data.gPosArray = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, vertices.size() * sizeof(float)*4, "gPosArray");
        data.gOldPosArray = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, vertices.size() * sizeof(float)*4, "gOldPosArray");
        data.gAuxPosArray = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, vertices.size() * sizeof(float)*4, "gAuxPosArray");
        data.gNormals = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, normals.size() * sizeof(float)*4, "gNormalArray");
        data.gTex = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, tex.size() * sizeof(float)*2, "gTex");
        data.gIndices = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, indices.size() * sizeof(unsigned int), "gIndices");
        data.gTriangleSoup = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, tSoup.size() * sizeof(float)*4, "gTriangleSoup");
        data.gSoupNormals = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, soupNormals.size() * sizeof(float)*4, "gSoupNormals");
        data.gSpherePositions = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, spherePositions.size() * sizeof(float)*4, "gSpherePositions");
        data.gSphereNormals = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, sphereNormals.size() * sizeof(float)*4, "gSphereNormals");

        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gTriangleSoup, tSoup);
        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gPosArray, vertices);
        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gOldPosArray, vertices);
        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gNormals, normals);
        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gTex, tex);
        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gIndices, indices);
        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gSoupNormals, soupNormals);
        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gSpherePositions, spherePositions);
        fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,app.transferQueue, data.gSphereNormals, sphereNormals);

        
        Cmn::addStorage(data.bindings, 0); // gPosArray
        Cmn::addStorage(data.bindings, 1); // gAuxPosArray
        Cmn::addStorage(data.bindings, 2); // gOldPosArray
        Cmn::addStorage(data.bindings, 3); // triangleSoup
        Cmn::addStorage(data.bindings, 4); // cloth tex
        Cmn::addStorage(data.bindings, 5); // soupnormals
        Cmn::addStorage(data.bindings, 6); // cloth indices
        Cmn::addStorage(data.bindings, 7); // cloth normals
        Cmn::addCombinedImageSampler(data.bindings, 8); // texture sampler
        Cmn::addStorage(data.bindings, 9); //sphere positions
        Cmn::addStorage(data.bindings, 10); // sphere normals

        Cmn::createDescriptorSetLayout(app.device, data.bindings, data.descriptorSetLayout);

        Cmn::createDescriptorPool(app.device, data.bindings, data.descriptorPool, 2);
        Cmn::allocateDescriptorSet(app.device, data.computeDescriptorSet[0], data.descriptorPool, data.descriptorSetLayout);
        Cmn::allocateDescriptorSet(app.device, data.computeDescriptorSet[1], data.descriptorPool, data.descriptorSetLayout);

        Cmn::bindBuffers(app.device, data.gPosArray.buf, data.computeDescriptorSet[0], 0);
        Cmn::bindBuffers(app.device, data.gAuxPosArray.buf, data.computeDescriptorSet[0], 1);
        Cmn::bindBuffers(app.device, data.gOldPosArray.buf, data.computeDescriptorSet[0], 2);
        Cmn::bindBuffers(app.device, data.gTriangleSoup.buf, data.computeDescriptorSet[0], 3);
        Cmn::bindBuffers(app.device, data.gTex.buf, data.computeDescriptorSet[0], 4);
        Cmn::bindBuffers(app.device, data.gSoupNormals.buf, data.computeDescriptorSet[0], 5);
        Cmn::bindBuffers(app.device, data.gIndices.buf, data.computeDescriptorSet[0], 6);
        Cmn::bindBuffers(app.device, data.gNormals.buf, data.computeDescriptorSet[0], 7);
        Cmn::bindCombinedImageSampler(app.device, data.textureView, data.textureSampler, data.computeDescriptorSet[0], 8);
        Cmn::bindBuffers(app.device, data.gSpherePositions.buf, data.computeDescriptorSet[0], 9);
        Cmn::bindBuffers(app.device, data.gSphereNormals.buf, data.computeDescriptorSet[0], 10);

        Cmn::bindBuffers(app.device, data.gPosArray.buf, data.computeDescriptorSet[1], 1); // These two
        Cmn::bindBuffers(app.device, data.gAuxPosArray.buf, data.computeDescriptorSet[1], 0); // Are inverted
        Cmn::bindBuffers(app.device, data.gOldPosArray.buf, data.computeDescriptorSet[1], 2);
        Cmn::bindBuffers(app.device, data.gTriangleSoup.buf, data.computeDescriptorSet[1], 3);
        Cmn::bindBuffers(app.device, data.gTex.buf, data.computeDescriptorSet[1], 4);
        Cmn::bindBuffers(app.device, data.gSoupNormals.buf, data.computeDescriptorSet[1], 5);
        Cmn::bindBuffers(app.device, data.gIndices.buf, data.computeDescriptorSet[1], 6);
        Cmn::bindBuffers(app.device, data.gNormals.buf, data.computeDescriptorSet[1], 7);
        Cmn::bindCombinedImageSampler(app.device, data.textureView, data.textureSampler, data.computeDescriptorSet[1], 8);
        Cmn::bindBuffers(app.device, data.gSpherePositions.buf, data.computeDescriptorSet[1], 9);
        Cmn::bindBuffers(app.device, data.gSphereNormals.buf, data.computeDescriptorSet[1], 10);


        vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eAll, 0, sizeof(A4Task2Data::PushConstant));

        vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &data.descriptorSetLayout, 1U, &pcr);
        data.layout = app.device.createPipelineLayout(pipInfo);
        data.push.parts_w_h = {data.particleCount, data.WIDTH, data.HEIGHT, 0U};
        data.push.dTs_rest = {0.f, 0.f, 0.f, 1.f/data.WIDTH*0.9f};
        data.spherePos = {3.f,0.f,1.f,0.3f};
        data.push.ballPos = {3.f,0.f,1.f,0.3f};
        this->render.prepare(data);
    }

    void loop(A4Task2Solution &solution);
    void prepare(A4Task2Solution &solution);
    
    void cleanup()
    {

        app.device.destroyImage(data.textureImage);
        app.device.freeMemory(data.textureImageMemory);
        app.device.destroyImageView(data.textureView);
        app.device.destroySampler(data.textureSampler);

        app.device.destroyDescriptorPool(data.descriptorPool);

        app.device.destroyPipelineLayout(data.layout);
        app.device.destroyDescriptorSetLayout(data.descriptorSetLayout);
        data.bindings.clear();

        auto Bclean = [&](Buffer &b)
        {
            app.device.destroyBuffer(b.buf);
            app.device.freeMemory(b.mem);
        };

        Bclean( data.gPosArray );
        Bclean( data.gOldPosArray );
        Bclean( data.gAuxPosArray );
        Bclean( data.gTriangleSoup );
        Bclean( data.gTex );
        Bclean( data.gIndices );
        Bclean( data.gNormals );
        Bclean( data.gSoupNormals );
        Bclean( data.gSpherePositions );
        Bclean( data.gSphereNormals );

        render.destroy();
    }
private:
    /*  1- read, modify and setup the data for the texture as an std::vector<float>
        2- fill a staging buffer with data in 1
        3- Create an Image, allocate its memory and bind them
        4- Change the format of the image so that we can copy the data to it (eTransferDstOptimal)
        5- copy data from 2 to 3 with new format of 4
        6- Change the format again to make it accessible to the shader (eShaderReadOnlyOptimal)
        7- Destroy the staging buffer and free its memory
        ===================================================
        members modified:
        vk::Image textureImage
        vk::DeviceMemory textureImageMemory
    */
  void make2DTexture()
    {

        Buffer staging;
        int w,h,n;
        { // #### SCOPE : Read and Fill a vector ####
            
            unsigned char* datad = stbi_load("../Assets/clothTexture.tga"
                , &w, &h, &n, 4);
            n = 4;
            if (datad==NULL){
                std::cout<<"error!"<<std::endl;
                return;
            }
            std::vector<unsigned char> data(datad, datad+w*h*n);

            int size = w*h;
            createBuffer( app.pDevice,app.device, w*h*n*sizeof(unsigned char),
                        vk::BufferUsageFlagBits::eTransferSrc,
                        vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible, 
                        "staging_tex", staging.buf, staging.mem);

            fillDeviceBuffer(app.device, staging.mem, data);
        }

        auto makeImage = [&](vk::Image &image, vk::ImageView &imageView, vk::DeviceMemory &img_mem)
        {
            vk::ImageCreateInfo imgInfo(vk::ImageCreateFlags{}, vk::ImageType::e2D,             // VkImageCreateFlags, VkImageType
                                        vk::Format::eR8G8B8A8Srgb,                              // VkImageFormat
                                        vk::Extent3D(w, h, 1),                                  // w,h,depth
                                        1, 1,                                                   // mipLevels, arrayLayers,
                                        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, // VkSampleCountFlagBits, VkImageTiling
                                        vk::ImageUsageFlagBits::eSampled |                      // VkImageUsageFlags
                                            vk::ImageUsageFlagBits::eTransferDst,
                                        vk::SharingMode::eExclusive,                            // VkSharingMode
                                        0, nullptr,                                             // queueFamilyIndexCount, *pQueueFamilyIndices
                                        vk::ImageLayout::eUndefined                             // VkImageLayout
            );

            // Create memory for image


            vk::MemoryRequirements img_mem_req;
            if (app.device.createImage(
                &imgInfo,  //Pointer to image create info
                nullptr,    // Allocation callbacks
                &image         //Pointer to image
            ) != vk::Result::eSuccess)
            {
                throw std::runtime_error("failed to create image!");
            }
            // Get memory requirements for a type of image
            app.device.getImageMemoryRequirements(image, &img_mem_req);
            // Allocate memory using image requirements and user defined properties
            vk::MemoryAllocateInfo allocInfo(img_mem_req.size,
                                            findMemoryType(img_mem_req.memoryTypeBits,
                                            vk::MemoryPropertyFlagBits::eDeviceLocal, app.pDevice));

            img_mem = app.device.allocateMemory(allocInfo, nullptr);
        // Connect memory to image
        app.device.bindImageMemory(image, img_mem, 0);

        vk::ImageViewCreateInfo viewInfo(
                {},
            image,                                     // Image to create view for
            vk::ImageViewType::e2D,                    // Type of image (1D, 2D, 3D, Cube, etc)
            vk::Format::eR8G8B8A8Srgb,                 // Format of image data
            {{}, {}, {}, {}},                          // Allows remapping of rgba components to other rgba values
            // Subresources allow the view to view only a part of an image
            {vk::ImageAspectFlagBits::eColor,          // Which aspect of image to view (e.g. COLOR_BIT for viewing colour)
            0,                                         // baseMipLevel - Start mipmap level to view from
            1,                                         // levelCount   - Number of mipmap levels to view 
            0,                                         // baseArrayLayer - Start array level to view from
            1                                          // layerCount -  Number of array levels to view
            });
        // Create image view
            imageView = app.device.createImageView(viewInfo);
        };

        makeImage(data.textureImage, data.textureView, data.textureImageMemory);
        setObjectName(app.device, data.textureImage,         "2DTexImage");
        setObjectName(app.device, data.textureView,          "2DTexView");
        setObjectName(app.device, data.textureImageMemory,   "2DTexImageMemory");
        
        transitionImageLayout(
            app.device, app.transferCommandPool, app.transferQueue,
            data.textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(app.device, app.transferCommandPool, app.transferQueue, staging.buf, data.textureImage, w, h, 1);
        ownershipTransfer(app.device, app.transferCommandPool, app.transferQueue, app.tQ, app.graphicsCommandPool, app.graphicsQueue, app.gQ, data.textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        app.device.destroyBuffer(staging.buf);
        app.device.freeMemory(staging.mem);
    }

    std::vector<std::array<float,4> > getTriangles(std::string dataFile, std::vector<std::array<float,4>> &normals){
        std::vector<std::array<float,4> > triangleSoup;
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(dataFile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
        }

        if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
        }
        
        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();
        triangleSoup.reserve(attrib.GetVertices().size());
        normals.reserve(attrib.GetVertices().size());
        // Loop over shapes
        for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            float vec[4];
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) { // * for us, fv = 3 (triangles)
            // Access to vertex
            tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
            
            float vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
            float vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
            float vz = attrib.vertices[3*size_t(idx.vertex_index)+2];
            
            triangleSoup.push_back( {vx,vy,vz,0} );

            // Check if `normal_index` is zero or positive. negative = no normal data
            if (idx.normal_index >= 0) {
                tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
                normals.push_back({nx,ny,nz,0.f});
            }
    /*
            // Check if `texcoord_index` is zero or positive. negative = no texcoord data
            if (idx.texcoord_index >= 0) {
                tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
            }
    */
            }        
            index_offset += fv;

            // per-face material
            // shapes[s].mesh.material_ids[f];
        }
        }
        return triangleSoup;
    }

    void makeSampler(){
        
        vk::PhysicalDeviceProperties properties = app.pDevice.getProperties();
        vk::SamplerCreateInfo samplerInfo(
            vk::SamplerCreateFlags{},
            vk::Filter::eLinear,vk::Filter::eLinear,
            vk::SamplerMipmapMode::eLinear,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat,
            0.f,
            vk::Bool32(false),
            properties.limits.maxSamplerAnisotropy,
            vk::Bool32(false),
            vk::CompareOp::eAlways, 0.f, 0.f,
            vk::BorderColor::eIntOpaqueBlack,
            vk::Bool32(false)
        );
        data.textureSampler = app.device.createSampler(samplerInfo);  
        setObjectName(app.device, data.textureSampler,       "2DTexSampler");
  
    }
    template <typename T>
    void makeTriangles(T& dst,T &src, unsigned int temp, unsigned int nVertsX){
                dst.push_back(src[temp]);
                dst.push_back(src[temp+nVertsX]);
                dst.push_back(src[temp+1]);

                dst.push_back(src[temp+nVertsX]);
                dst.push_back(src[temp+1+nVertsX]);
                dst.push_back(src[temp+1]);
    }

    void createPlane(unsigned int nVertsX, unsigned int nVertsY, std::vector<uint32_t>& indices, arrayVec4& vertices, arrayVec4 &normals, 
                    std::vector<std::array<float,2>> &tex)
    {
	if(nVertsX < 2 || nVertsY < 2)
		throw std::runtime_error("Invalid plane resolution");
		

	//vertex data
	unsigned int xSegments = nVertsX - 1;
	unsigned int ySegments = nVertsY - 1;

	float deltaX = 1.0f / xSegments;
	float deltaY = 1.0f / ySegments;

    vertices.reserve(nVertsX*nVertsY);
    normals.reserve(nVertsX*nVertsY);
    tex.reserve(nVertsX*nVertsY);

	for(unsigned int y = 0; y < nVertsY; y++)
		for(unsigned int x = 0; x < nVertsX; x++)
		{
			vertices.push_back({0, deltaX * x - 0.5f, -deltaY * y + 0.6f, 0.f});
			normals.push_back({0.f, 0.f, 1.0f, 0.f});
			tex.push_back({deltaX * x, deltaY * y});
		}

	//create indices
    indices.reserve(nVertsX*nVertsY*3*2);


	for(unsigned int x = 0; x < xSegments; x++){
		for(unsigned int y = 0; y < ySegments; y++)
		{
            unsigned int temp = y * nVertsX + x;
                indices.push_back(temp);
                indices.push_back(temp+nVertsX);
                indices.push_back(temp+1);
                indices.push_back(temp+nVertsX);
                indices.push_back(temp+1+nVertsX);
                indices.push_back(temp+1);
            /*
            makeTriangles(triSoup, vertices, temp, nVertsX);
            makeTriangles(normals, tempNormals, temp, nVertsX);
            makeTriangles(tex, tempTex, temp, nVertsX);
            */
		}
    }
    }
    void computeReference();

    
};
