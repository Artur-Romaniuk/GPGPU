#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"
#include "A3task3.h"
#include "host_timer.h"

A3Task3::A3Task3(float *in, uint32_t size, uint32_t nBins)
    :  numBins(nBins)
{
    input.assign(in, in + size);
    
    computeReference();
}

bool A3Task3::evaluateSolution(A3Task3Solution &solution)
{
    solution.prepare(input, numBins);
    solution.compute();
    std::vector<int> result = solution.result();

    std::vector<int> diff(result.size(), 0);
    bool pass = true;
		const float EPS = 1e-6f;
    for(int i = 0; i< diff.size(); i++){
        diff[i] = abs(result[i] - reference[i]);
        if( diff[i]>EPS)
        {
            pass = false;
            break;
        }
    }

    if(!pass)
        std::cout << "error A3T3: check outHist.jpg" << std::endl;
    return pass;
}

void A3Task3::computeReference()
{
    reference.resize(numBins);

	for(int i = 0; i < input.size(); i++) {
			float p = input[i] * float(numBins);
			int h_idx = std::min<int>(numBins - 1, std::max<int>(0, int(p)));
			reference[h_idx]++;
    }
}

void print_histogram(const std::vector<int> &h)
{
	int max_val = 0;
	for(auto i: h)
		max_val = std::max<int>(max_val, i);

	std::cout << "+";
	for(size_t i = 0; i < h.size(); i++)
		std::cout << "-";
	std::cout << "+\n";
	const int max_height = 8;
	for(int y = max_height - 1; y >= 0; y--) {
		int val = (max_val * y) / max_height;
		std::cout << "|";
		for(auto i: h)
			std::cout << (i >= val ? '#' : ' ');
		std::cout << "|\n";
	}
	std::cout << "+";
	for(size_t i = 0; i < h.size(); i++)
		std::cout << "-";
	std::cout << "+\n";
}