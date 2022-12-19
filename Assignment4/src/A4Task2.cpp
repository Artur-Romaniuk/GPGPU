#include "A4Task2.h"

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

void A4Task2::prepare(A4Task2Solution &solution){
    //solution.prepare(data.particleCount,data);
}
void A4Task2::loop(A4Task2Solution &solution){
    solution.compute();
    render.renderFrame(data);
}

void A4Task2::computeReference()
{
    
}