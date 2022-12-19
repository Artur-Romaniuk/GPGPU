#include <iostream>
#include <cstdlib>
#include <stb_image.h>
#include <stb_image_write.h>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include <GLFW/glfw3.h>

//#include "A4Task2Solution/flag.h"
#include "A4Task1Solution/particles.h"
#include "A4Task2Solution/flag.h"

#include "renderdoc.h"
#include <CSVWriter.h>
#include <render.h>
void Task2()
{
    AppResources app;

    initApp(app, true);

    // Since our application is now frame-based, renderdoc can find frame delimiters on its own

    CSVWriter csv;

    Render render(app, 2);
    render.camera.position = {-0.926159799, -1.42368257, 0.127661496};
    render.camera.theta = 1.56080616;
    render.camera.phi = 0.585675538;
    A4Task2 task2(app, render);
    FlagA4Task2Solution sol1(app, task2.data, 16, 16);
    //sol1.compute();

    // Loop until the user closes the window
    while (true)
    {
        double time = glfwGetTime();
        render.timedelta = time - render.prevtime;
        render.prevtime = time;

        render.preInput();

        // Poll for and process events
        glfwPollEvents();

        render.doRawMouseInput |= glfwGetKey(app.window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
        render.input();
        if (glfwGetKey(app.window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS && glfwGetKey(app.window, GLFW_KEY_LEFT_SHIFT) != GLFW_PRESS)
        {
            task2.data.push.ballPos.x += render.xdiff * 0.001;
            task2.data.push.ballPos.z -= render.ydiff * 0.001;
        }

        if (glfwGetKey(app.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(app.window, 1);

        if (glfwWindowShouldClose(app.window))
            break;

        // Render here //
        task2.loop(sol1);
    }

    app.device.waitIdle();

    sol1.cleanup();
    task2.cleanup();

    render.cleanup();

    app.destroy();
}
void Task1()
{
    AppResources app;

    initApp(app, true);

    // Since our application is now frame-based, renderdoc can find frame delimiters on its own
    renderdoc::initialize();
    renderdoc::startCapture();

    CSVWriter csv;

    Render render(app, 2);
    render.camera.position = glm::vec3(0.5, 2, 0.9);
    render.camera.phi = glm::pi<float>();
    render.camera.theta = 0.4 * glm::pi<float>();
    A4Task1 task1(app, render, 400000, "../Assets/cubeNormal.obj");
    ParticlesA4Task1Solution sol1(app, task1.data, 192, 192);
    task1.prepare(sol1);

    renderdoc::endCapture();

    // Loop until the user closes the window
    while (true)
    {
        double time = glfwGetTime();
        render.timedelta = time - render.prevtime;
        render.prevtime = time;

        render.preInput();

        // Poll for and process events
        glfwPollEvents();

        render.input();

        if (glfwGetKey(app.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(app.window, 1);

        if (glfwWindowShouldClose(app.window))
            break;

        // Render here //
        task1.loop(sol1);
    }

    app.device.waitIdle();

    sol1.cleanup();
    task1.cleanup();

    render.cleanup();

    app.destroy();
}
int main()
{
    try
    {
        Task1();
        //Task2();
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