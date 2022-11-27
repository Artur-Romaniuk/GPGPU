#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
//#include "A3Task1.h"
//#include "A3Task2.h"
#include "A3Task2Solution/convsep.h"
#include "A3Task1Solution/conv3x3.h"
#include "A3Task3Solution/histogram.h"
#include "renderdoc.h"
#include <CSVWriter.h>

void run_A3_task1(AppResources &app, float* data, int w, int h){
    int size = w*h;

   /* a different kernel
    float kern[3][3] = {
        { -1.0f / 8.0f, -1.0f / 8.0f, -1.0f / 8.0f },
        { -1.0f / 8.0f,  1.0f,        -1.0f / 8.0f },
        { -1.0f / 8.0f, -1.0f / 8.0f, -1.0f / 8.0f }
    };*/
    float kern[3][3] = {
        { .2, .2, .2 },
        { .2, 1.0f, .2 },
        { .2, .2, .2 }
    };
    std::cout << "==== 3x3 Convolution ===" << std::endl;

    std::cout<<"Input kernel:"<<std::endl;
    std::cout<<"--------------"<<std::endl;

    A3Task1 a3Task1(data, kern, w, h);
    ConvolutionA3Task1Solution convSol(app, 16, 16);

    a3Task1.evaluateSolution(convSol);
    std::cout<<"--------------"<<std::endl;

    convSol.cleanup();
    std::cout << "inside time: " << convSol.insideTime << " ms" << std::endl;

    std::cout << size / (convSol.mstime / 1 / 1000) / 1000000000 << " GE/s" << std::endl <<std::endl;
}

void run_A3_task2(AppResources &app, float* data,int w, int h){
    int size = w*h;
    std::cout << "==== Separable Convolution ===" << std::endl;

    // you can use two different vectors, in this case we use the same, but feel free to play around
    std::vector<float> uv = {0.61f, .242f, .383f, .242f, .061f};
    // you can use a loop to change the workgroup values :)
    uint32_t workGroupSizes_x[2] = {32, 8};
    uint32_t workGroupSizes_y[2] = {16, 16};
    A3Task2 a3Task2(data, uv, uv, w, h, 3u, 3u);

    SeparableA3Task2Solution sepSol(app, workGroupSizes_x, workGroupSizes_y);
    a3Task2.evaluateSolution(sepSol);
    sepSol.cleanup();
    std::cout<<"--------------"<<std::endl;

    std::cout << "inside time: " << sepSol.insideTime << " ms" << std::endl;
    std::cout << size / (sepSol.mstime / 1 / 1000) / 1000000000 << " GE/s" << std::endl << std::endl;
}

void run_A3_task3(AppResources &app, float* data,int w, int h){
    int size = w*h;

    A3Task3 t3(data, w*h, 26);
    HistogramA3Task3Solution histsol(app, 128);
    t3.evaluateSolution(histsol);

    std::cout<<"=== REFERENCE ==="<<std::endl;
    print_histogram(t3.reference);
    std::cout<<"=== RESULT ==="<<std::endl;
    print_histogram(histsol.result());
    std::cout << "==== histogram local atomics ===" << std::endl;
    std::cout << size / (histsol.mstime / 1 / 1000) / 1000000000 << " GE/s" << std::endl;
    std::cout << "inside time: " << histsol.insideTime << " ms\t" << size / (histsol.insideTime / 1 / 1000) / 1000000000 << " GE/s"<< std::endl;

    histsol.cleanup();
    histsol.useNaive();
    t3.evaluateSolution(histsol);

    std::cout<<"=== RESULT ==="<<std::endl;
    print_histogram(histsol.result());
    std::cout << "==== histogram naive atomics ===" << std::endl;
    std::cout << size / (histsol.mstime / 1 / 1000) / 1000000000 << " GE/s" << std::endl;
    std::cout << "inside time: " << histsol.insideTime << " ms\t" << size / (histsol.insideTime / 1 / 1000) / 1000000000 << " GE/s"<< std::endl;
    histsol.cleanup();
}

int main()
{
    try
    {
        AppResources app;
        
        initApp(app);

        renderdoc::initialize();
        renderdoc::startCapture();

        CSVWriter csv;
        
        float mstime = 0;
        float mstimesq = 0;
        int w,h,n;
        mstime = 0;
        float *data = stbi_loadf("../images/the_starry_night.jpg"
            , &w, &h, &n, 1);
        if (data==NULL){
            std::cout<<"error starry night not loaded"<<std::endl;
            return 0;
        }

        run_A3_task1(app, data, w, h);

        run_A3_task2(app, data, w, h);

        run_A3_task3(app, data, w, h);

        renderdoc::endCapture();
        app.destroy();
        free(data);
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
        std::cout << "unknown error/n";
        exit(-1);
    }
    return EXIT_SUCCESS;
}