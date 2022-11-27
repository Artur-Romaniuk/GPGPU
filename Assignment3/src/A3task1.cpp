#include "A3Task1.h"

#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"
#include "host_timer.h"
#include "stb_image_write.h"

A3Task1::A3Task1(int w, int h) : input(w * h, 0), kernel(), w(w), h(h)
{
    //todo: fill with arbitrary values
    //fill kernel and input
    computeReference();
}
A3Task1::A3Task1(float *in, float k[3][3], uint32_t w, uint32_t h) : w(w), h(h)
{
    input.assign(in, in + w * h);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            kernel[i][j] = k[i][j];
            std::cout << kernel[i][j] << " ";
        }
        std::cout << std::endl;
    }
    computeReference();
}
A3Task1::A3Task1(std::vector<float> input, float k[3][3], uint32_t w, uint32_t h) : input(input), w(w), h(h)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            kernel[i][j] = k[i][j];

    computeReference();
}

bool A3Task1::evaluateSolution(A3Task1Solution &solution)
{
    std::vector<float> ker(9);
    for(int j=0; j<3; j++)
        for(int i=0; i<3; i++)
            ker[i+3*j]=kernel[i][j];
    
    solution.prepare(input, ker, w, h);
    solution.compute();
    std::vector<float> result = solution.result();

    const float EPS = 1e-6f;
    std::vector<float> diff(result.size(),0.f);
    bool pass = true;
    for(int i = 0; i< diff.size(); i++){
        diff[i] = abs(result[i] - reference[i]);
        if( diff[i]>EPS)
        {
            pass = false;
            break;
        }
    }
    if(!pass)
        std::cout << "error A3T1: check 3x3diff.jpg" << std::endl;
    writeFloatJpg("images/3x3result.jpg", result, w, h);
    writeFloatJpg("images/3x3diff.jpg", diff, w, h);
    writeFloatJpg("images/3x3ref.jpg", reference, w, h);
    return pass;
}

void A3Task1::computeReference()
{
    reference.resize(w * h);
    int pitch = w; 
    float kernelWeight = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            kernelWeight += kernel[i][j];

    if (kernelWeight > 0)
        kernelWeight = 1.0f / kernelWeight; // Use this for normalization
    else
        kernelWeight = 1.0f;
    float max = 0.f;
    for (unsigned int y = 0; y < h; y++)
    {
        for (unsigned int x = 0; x < w; x++)
        {
            float value = 0;
            //Apply convolution kernel
            for (int offsetY = -1; offsetY < 2; offsetY++)
            {
                int sy = y + offsetY;
                if (sy >= 0 && sy < int(h))
                    for (int offsetX = -1; offsetX < 2; offsetX++)
                    {
                        int sx = x + offsetX;
                        if (sx >= 0 && sx < int(w))
                            value += input[sy * pitch + sx] * kernel[1 + offsetY][1 + offsetX];
                    }
            }
            reference[y * pitch + x] = value * kernelWeight;
            if (value * kernelWeight > max)
                max = value * kernelWeight;
        }
    }
    std::cout<<"max value: "<< max << std::endl;
    std::vector<uint8_t> refINT(reference.size());
    for (int i = 0; i < reference.size(); i++)
        refINT[i] = static_cast<uint8_t>(reference[i] * 255);
}