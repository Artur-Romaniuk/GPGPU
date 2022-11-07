#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "A1task1.h"
#include "A1task2.h"
#include "renderdoc.h"
#include <CSVWriter.h>




void run_A1_task1(AppResources app) {
    CSVWriter csv;
    csv.newRow() << "wg" << "vs" << "timing";
    const std::vector<uint32_t> vecSizes{512*512, 1024*1024, 2048*2048, 2048*2048 + 31};
    for(const uint32_t vecsize : vecSizes){
        A1_Task1 task(app);
        task.prepare(vecsize);

        for(int i= 32; i<512;i=i<<1)
        {
            uint64_t avgTime{};
            static constexpr uint8_t numberOfAverageRuns = 10U;
            for(uint8_t index = 0; index<numberOfAverageRuns; index++){
                task.compute(i, 1, 1);
                task.checkDefaultValues();
                avgTime+=task.mstime * 10000;
            }

            csv.newRow() << i << vecsize << avgTime / 10000.f/ static_cast<float>(numberOfAverageRuns);
        }
        task.cleanup();
    }

    std::cout << csv << std::endl;
    csv.writeToFile("vectorAddition.csv", false);
}

void run_A1_task2(AppResources app, bool optimal = false) {
    CSVWriter csv;
    csv.newRow() << "wg" << "vs" << "timing";
    std::vector<std::pair<uint32_t, uint32_t>> matrixSizes{{1600, 2000},{3200, 4000}, {6400, 8000}};
    for(auto matrixSize : matrixSizes){
        A1_Task2 task(app);
        task.prepare(matrixSize.first, matrixSize.second);

        std::vector<std::pair<uint32_t, uint32_t>> workGroupSizes{{4,4},{16,8},{16,16},{32,16}};
        for(auto workGroupSize : workGroupSizes)
        {
            uint64_t avgTime{};
            static constexpr uint8_t numberOfAverageRuns = 10U;
            for(uint8_t index = 0; index<numberOfAverageRuns; index++){
                task.compute(workGroupSize.first, workGroupSize.second, 1, (optimal?"matrixRotOpti":"matrixRotNaive"));
                task.checkDefaultValues();
                avgTime+=task.mstime*100000;
            }
            csv.newRow() << workGroupSize.first * workGroupSize.second << matrixSize.first * matrixSize.second << avgTime/ 100000.f/ static_cast<float>(numberOfAverageRuns);
        }
        task.cleanup();
    }

    std::cout << csv << std::endl;
    std::string fileName = "vectorRotate";
    fileName+= (optimal?"Opt":"Naive");
    fileName+=".csv";
    csv.writeToFile(fileName, false);
}

int main()
{
    try
    {
        //AppResources is a struct = Instance, Physical Device, Queues, CommandPool and QueryPool are some of the members
        AppResources app;
        
        //Initialises the app with instance, Physical device , Compute & Transfer Queues
        initApp(app);

        renderdoc::initialize();
        renderdoc::startCapture();

        std::cout << "\n" << "=========== Task 1 ===========" << "\n";
        run_A1_task1(app);
        std::cout << "\n" << "======== Task 2 Naive ========" << "\n";
        run_A1_task2(app);
        std::cout << "\n" << "======= Task 2 Optimal =======" << "\n";
        run_A1_task2(app, true);
    
        renderdoc::endCapture();

        app.destroy();
    }
    catch (vk::SystemError &err)
    {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        exit(-1);
    }
    catch (std::exception &err)
    {
        std::cout << "std::exception: " << err.what() << std::endl;
        exit(-1);
    }
    catch (...)
    {
        std::cout << "unknown error\n";
        exit(-1);
    }
    return EXIT_SUCCESS;
}