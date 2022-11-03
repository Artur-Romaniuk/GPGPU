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




void run_A1_task1(A1_Task1 task) {
    CSVWriter csv;
    csv.newRow() << "dx" << "dy" << "dz" << "vecsize" << "timing";
    task.prepare(1024 * 1024 + 31);

    for(int i= 32; i<1024;i=i<<1)
    {
        task.compute(i, 1, 1);
        task.checkDefaultValues();
        csv.newRow() << i << 1 << 1 << 1024 * 1024 << task.mstime;
    }
    
    task.cleanup();
    std::cout << csv << std::endl;
    csv.writeToFile("vectorAddition.csv", false);

}

void run_A1_task2(A1_Task2 A1task2) {
    A1task2.prepare(3200, 4000);

    A1task2.compute(32, 16, 1);
    A1task2.checkDefaultValues();

    std::cout << "took: " << A1task2.mstime << " ms" << std::endl;

    A1task2.compute(32, 16, 1, "matrixRotOpti");
    A1task2.checkDefaultValues();
    std::cout << "took: " << A1task2.mstime << " ms" << std::endl;

    A1task2.cleanup();
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
        
        A1_Task1 task(app);
        
        run_A1_task1(task);

        
        A1_Task2 A1task2(app);
        run_A1_task2(A1task2);
    
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