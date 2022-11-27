#include "A3Task2.h"

#include <iostream>
#include <numeric> // for std::accumulate
#include <cstdlib>
#include <functional>
#include <math.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"
#include "host_timer.h"
#include "stb_image_write.h"

A3Task2::A3Task2(float *in, 
    std::vector<float> kernelH, std::vector<float> kernelV, 
    uint32_t w, uint32_t h, uint32_t h_elements, uint32_t v_elements)
    :  kernelH(kernelH), kernelV(kernelV), w(w), h(h), nH(h_elements), nV(v_elements)
{
    input.assign(in, in + w * h);
    
    computeReference();
}

bool A3Task2::evaluateSolution(A3Task2Solution &solution)
{
    solution.prepare(input, kernelH, kernelV, w, h, nH, nV);
    solution.compute();
    std::vector<float> result = solution.result();

    const float EPS = 1e-6f;
    std::vector<float> diff(result.size(), 0.f);
    bool pass = true;
    for (int i = 0; i < diff.size(); i++)
    {
        diff[i] = abs(result[i] - reference[i]);
        if (diff[i] > EPS)
        {
            pass = false;
            std::cout<<i%w << ", " << i/w <<"\t";
            break;
        }
    }

    if(!pass)
        std::cout << "error in A3T2: check sepdiff.jpg" << std::endl;
    
    writeFloatJpg("images/sep_result.jpg", result, w, h);
    writeFloatJpg("images/sep_diff.jpg", diff, w, h);
    writeFloatJpg("images/sep_ref.jpg", reference, w, h);
    return pass;
}

void A3Task2::computeReference()
{
    reference.resize(w * h);
    int p = w;

    float kWeightH = std::accumulate(kernelH.begin(), kernelH.end(), 0.f);
    float kWeightV = std::accumulate(kernelV.begin(), kernelV.end(), 0.f);
    float kernelWeight = 1.f / sqrt(kWeightH * kWeightV);
    // std::cout<< "sumH = " <<kWeightH << "\tsumV = "<<kWeightV <<"\tkernelWeights = "<<kernelWeight <<std::endl;
    std::transform(kernelH.begin(), kernelH.end(), kernelH.begin(), [kernelWeight](float& v) {return v*kernelWeight;});
    std::transform(kernelV.begin(), kernelV.end(), kernelV.begin(), [kernelWeight](float& v) {return v*kernelWeight;});


    std::vector<float> tempBuffer(w * h, 0.f);
    //Horizontal pass
    int radiusH = (kernelH.size() - 1) / 2;
    int radiusV = (kernelV.size() - 1) / 2;
    for (int y = 0; y < (int)h; y++)
        for (int x = 0; x < (int)w; x++)
        {
            float value = 0;
            //Apply horizontal kernel
            for (int k = -radiusH; k <= radiusH; k++)
            {
                int sx = x + k;
                if (sx >= 0 && sx < (int)w)
                    value += input[y * p + sx] * kernelH[radiusH - k];
            }
            tempBuffer[y * p + x] = value;
        }

    //Vertical pass
    for (int x = 0; x < (int)w; x++)
        for (int y = 0; y < (int)h; y++)
        {
            float value = 0;
            //Apply vertical kernel
            for (int k = -radiusV; k <= radiusV; k++)
            {
                int sy = y + k;
                if (sy >= 0 && sy < (int)h)
                    value += tempBuffer[sy * p + x] * kernelV[radiusV - k];
            }
            reference[y * p + x] = value;
        }

    
}