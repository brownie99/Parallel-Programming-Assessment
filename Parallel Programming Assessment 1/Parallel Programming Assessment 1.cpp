// Parallel Programming Assessment 1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//
//This implementation of the assessment implements calculating the mean, min, max, standard deviation, bitonic sort, median, 1st and 3rd quartiles all using real values.
//The mean and standard deviation use the code from the add kernel from the workshops. The min/max kernel using local memory to find partial min and maxes which are then used by the host.
//The bitonic sort uses the code from the lecture to sort individual bitonic sets. This has then been improved using a three stage approach. 
//The initial stage sets up the list into stage 0 where groups of 4 values are in a bitonic set. The next stage repeatedly launches the same kernel on increasingly larger bitonic sequences.
//The final stage sorts the final bitonic sequence
//Every kernel reports the upload, kernel execution and download times for profiling

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "Utils.h"
#include <chrono>
#include <stdint.h>

using namespace std;

void print_help() {
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
	cerr << "  -s : show sorted list (comes before stats)" << endl;
	cerr << "  -g : set work group size (default max size for device)" << endl;
}

//function to load data from provided path
//opens and reads file line by line, splitting and storing only the temp values
//outputs every 100,000th value as progress
vector<float> loadData(string path) {
	vector<float> temps;
	ifstream reader(path);
	string line;
	int counter = 0;
	//read lines
	while (getline(reader, line)) {
		stringstream ss(line);
		string temp;
		//get each token in string and stop at last
		while (ss >> temp) {
		}
		//store temp as float
		temps.push_back(stof(temp));
		counter++;
		if (counter % 100000 == 0)
		{
			cout << counter << endl;
		}
	}
	return temps;
}

//main function
int main(int argc, char** argv)
{
	//load data into vector
	vector<float> temps;
	cout << "Loading Data" << endl;
	temps = loadData("temp_lincolnshire_datasets/temp_lincolnshire.txt");
	cout << "Total size of dataset = " << temps.size() << endl;

	
	

	//initialise option variables
	int platformID = 0;
	int deviceID = 0;
	int work_groups = 0;
	bool show_sorted = false;

	//check command line arguments and set options
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platformID = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { deviceID = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
		else if (strcmp(argv[i], "-s") == 0) { show_sorted = true; }
		else if ((strcmp(argv[i], "-g") == 0) && (i < (argc - 1))) { work_groups = atoi(argv[++i]); }
	}

	try {
		//host operations
		//select computing devices
		cl::Context context = GetContext(platformID, deviceID);

		// get device
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; 

		//set work group size
		size_t local_size;
		if (work_groups > 0) {
			local_size = work_groups;
		}
		else {
			//set workgroup size to max value available for the device - datasets are large so a large value is useful
			local_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		}
		//display the selected device
		cout << "Runinng on " << GetPlatformName(platformID) << ", " << GetDeviceName(platformID, deviceID) << endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			throw err;
		}


		

		//create a padded array so the total size is divisible by the number of workgroups
		vector<float> padded_temps(temps.begin(), temps.end());
		size_t padding_size = local_size - (padded_temps.size() % local_size);
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected

		if (padding_size) {
			//create an extra vector with neutral values
			vector<float> A_ext(padding_size, 0.0f);
			//append that extra vector to our input
			padded_temps.insert(padded_temps.end(), A_ext.begin(), A_ext.end());
		}

		//create a vector to store the output from the mean kernel
		vector<int> output(1);

		//initialise some size variables for use later when creating buffers and kernels
		size_t input_sizef = padded_temps.size() * sizeof(float);//size in bytes
		size_t vector_elementsf = padded_temps.size();//number of elements
		size_t output_size = output.size() * sizeof(int);//size in bytes
		size_t output_sizef = output.size() * sizeof(float);

		//device buffers
		cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_input(context, CL_MEM_READ_WRITE, input_sizef);

		//profiling events for the buffers and kernel
		cl::Event output_event;
		cl::Event input_event;
		cl::Event output_download_event;
		cl::Event prof_event;

		//copy arrays input and output to device memory
		queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, input_sizef, &padded_temps[0], NULL, &input_event);
		queue.enqueueFillBuffer(buffer_output, 0, 0, output_size, NULL, &output_event);//zero output buffer on device memory

		//create kernel and set arguments
		cl::Kernel kernel_mean = cl::Kernel(program, "meanf");
		kernel_mean.setArg(0, buffer_input);
		kernel_mean.setArg(1, buffer_output);
		kernel_mean.setArg(2, cl::Local(local_size * sizeof(float)));//local memory size

		

		//start the kernel
		queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, cl::NDRange(vector_elementsf), cl::NDRange(local_size), NULL, &prof_event);

		//copy the result from device to host
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output_sizef, &output[0], NULL, &output_download_event);

		cout << "Average kernel timings:" << endl;
		cout << "Input upload [ns]: " << input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Output upload [ns]: " << output_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - output_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nKernel started" << endl;
		cout << "Queued time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "Submitted time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() << endl;
		cout << "Kernal execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Total kernel time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "\nOutput download [ns]: " << output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nTotal time [ns]: " << output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		//make sure previous kernel has finished before continuing
		queue.finish();

		//initialise new kernel storage
		output_sizef = (padded_temps.size() / local_size) * sizeof(float);
		vector<float> maxs(output_sizef, 0.0f);
		vector<float> mins(output_sizef, 0.0f);

		//create new kernel
		cl::Kernel kernel_maxmin = cl::Kernel(program, "maxminf");
		
		//initialise new buffers
		cl::Buffer buffer_maxs(context, CL_MEM_READ_WRITE, output_sizef);
		cl::Buffer buffer_mins(context, CL_MEM_READ_WRITE, output_sizef);

		//profiling events for the buffers and kernel
		cl::Event max_output_upload_event;
		cl::Event min_output_upload_event;
		cl::Event max_output_download_event;
		cl::Event min_output_download_event;

		//send buffers to device - both zeroed ready for output
		queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, input_sizef, &padded_temps[0], NULL, &input_event);
		queue.enqueueFillBuffer(buffer_maxs, 0, 0, output_sizef, NULL, &max_output_upload_event);
		queue.enqueueFillBuffer(buffer_mins, 0, 0, output_sizef, NULL, &min_output_upload_event);

		//set kernel arguments
		kernel_maxmin.setArg(0, buffer_input);
		kernel_maxmin.setArg(1, buffer_maxs);
		kernel_maxmin.setArg(2, buffer_mins);
		kernel_maxmin.setArg(3, cl::Local(local_size * sizeof(float)));//local memory size

		//start kernel
		queue.enqueueNDRangeKernel(kernel_maxmin, cl::NullRange, cl::NDRange(vector_elementsf), cl::NDRange(local_size), NULL, &prof_event);

		//retrieve outputs from device
		queue.enqueueReadBuffer(buffer_maxs, CL_TRUE, 0, output_sizef, &maxs[0], NULL, &max_output_download_event);
		queue.enqueueReadBuffer(buffer_mins, CL_TRUE, 0, output_sizef, &mins[0], NULL, &min_output_download_event);


		cout << "\n\nMax and Min kernel timings:" << endl;
		cout << "Input upload [ns]: " << input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Max upload [ns]: " << max_output_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_output_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Min upload [ns]: " << min_output_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_output_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nKernel started" << endl;
		cout << "Queued time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "Submitted time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() << endl;
		cout << "Kernal execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Total kernel time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "\nMax download [ns]: " << max_output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Min download [ns]: " << min_output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nTotal time [ns]: " << min_output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		//make sure previous kernel has finished before continuing
		queue.finish();

		//calculate max and minimum value from partials
		float max_val = maxs[0];
		float min_val = mins[0];
		for (int i = 1; i < maxs.size(); i++) {
			if (maxs[i] > max_val) {
				max_val = maxs[i];
			}
			if (mins[i] < min_val) {
				min_val = mins[i];
			}
		}

		// calculate average
		float x = output[0] / 10.0f;
		float mean_val = x / temps.size();

		//set precision so decimals are shown on large numbers
		cout.precision(10);

		//pad variance vector
		vector<float> padded_temps_variance(temps.begin(), temps.end());
		padding_size = local_size - (padded_temps_variance.size() % local_size);
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			vector<float> var_ext(padding_size, mean_val);
			//append that extra vector to our input
			padded_temps_variance.insert(padded_temps_variance.end(), var_ext.begin(), var_ext.end());
		}
		
		//set kernel input, output and size variables
		vector<float> mean(1);
		vector<int64_t> varsum(1);
		input_sizef = padded_temps_variance.size() * sizeof(float);
		output_size = varsum.size() * sizeof(int64_t);
		output_sizef = 1 * sizeof(float);
		
		varsum[0] = (int64_t)0;
		mean[0] = mean_val;

		//initialise new buffers
		cl::Buffer buffer_mean(context, CL_MEM_READ_ONLY, mean.size() * sizeof(float));
		cl::Buffer buffer_varsum(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_padded_variance(context, CL_MEM_READ_ONLY, input_sizef);
		
		cl::Event mean_value_upload_event;
		cl::Event varsum_output_upload_event;
		cl::Event mean_value_download_event;
		cl::Event varsum_output_download_event;

		//send buffers to device
		queue.enqueueWriteBuffer(buffer_padded_variance, CL_TRUE, 0, input_sizef, &padded_temps_variance[0], NULL, &input_event);
		queue.enqueueWriteBuffer(buffer_mean, CL_TRUE, 0, mean.size() * sizeof(float), &mean[0], NULL, &mean_value_upload_event);
		queue.enqueueWriteBuffer(buffer_varsum, CL_TRUE, 0, output_size, &varsum[0], NULL, &varsum_output_upload_event);
		

		//create kernel
		cl::Kernel kernel_var = cl::Kernel(program, "variance");

		//set kernel arguments
		kernel_var.setArg(0, buffer_padded_variance);
		kernel_var.setArg(1, buffer_varsum);
		kernel_var.setArg(2, buffer_mean);
		kernel_var.setArg(3, cl::Local(local_size * sizeof(float)));//local memory size

		//start kernel
		queue.enqueueNDRangeKernel(kernel_var, cl::NullRange, cl::NDRange(padded_temps_variance.size()), cl::NDRange(local_size), NULL, &prof_event);

		//retrieve output from device
		queue.enqueueReadBuffer(buffer_varsum, CL_TRUE, 0, output_size, &varsum[0], NULL, &varsum_output_download_event);

		cout << "\n\nVariance squared difference kernel timings:" << endl;
		cout << "Input upload [ns]: " << input_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Mean value upload [ns]: " << mean_value_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - mean_value_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Intermidiate squared difference sum upload [ns]: " << varsum_output_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - varsum_output_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nKernel started" << endl;
		cout << "Queued time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "Submitted time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() << endl;
		cout << "Kernal execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Total kernel time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "Intermidiate squared difference sum download [ns]: " << varsum_output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - varsum_output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nTotal time [ns]: " << varsum_output_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - input_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		//calculate variance and standard deviation
		float variance_value = (float)varsum[0] / (100.0f * temps.size());
		float standard_deviation_val = sqrt(varsum[0] / (100.0f * temps.size()));
		
		//pad temperature array for sorting - must be power of 2 to work with bitonic sort
		vector<float> padded_sort_temps(temps.begin(), temps.end());
		float pos = ceil(log2(padded_sort_temps.size()));
		int power = pow(2, pos);
		padding_size = power - padded_sort_temps.size();
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values - 999.9 is greater than any temperature in dataset
			vector<float> A_ext(padding_size, 999.9f);
			//append that extra vector to our input
			padded_sort_temps.insert(padded_sort_temps.end(), A_ext.begin(), A_ext.end());
		}

		//Bitonic sort - start with initial stage to set up stage 0

		//initialise vector to be sorted
		vector<float> sortVec(padded_sort_temps.size());

		//initialise buffer for kernel
		cl::Buffer buffer_sort_temps(context, CL_MEM_READ_WRITE, sortVec.size() * sizeof(float));

		//profiling events
		cl::Event sort_temps_upload_event;
		cl::Event sort_temps_download_event;

		//write buffer to device
		queue.enqueueWriteBuffer(buffer_sort_temps, CL_TRUE, 0, sortVec.size() * sizeof(float), &padded_sort_temps[0], NULL, &sort_temps_upload_event);

		//create kernel
		cl::Kernel kernel_initial = cl::Kernel(program, "bitonic_initial");

		//set kernel arguments
		kernel_initial.setArg(0, buffer_sort_temps);

		//start kernel
		queue.enqueueNDRangeKernel(kernel_initial, cl::NullRange, cl::NDRange(sortVec.size()), cl::NullRange, NULL, &prof_event);

		//retrieve sorted vector stage 0 from device
		queue.enqueueReadBuffer(buffer_sort_temps, CL_TRUE, 0, sortVec.size() * sizeof(float), &sortVec[0], NULL, &sort_temps_download_event);

		cout << "\n\nBitonic stage 0 kernel timings:" << endl;
		cout << "Input upload [ns]: " << sort_temps_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sort_temps_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nKernel started" << endl;
		cout << "Queued time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "Submitted time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() << endl;
		cout << "Kernal execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Total kernel time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "\nStage 0 download [ns]: " << sort_temps_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sort_temps_download_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nTotal time [ns]: " << sort_temps_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sort_temps_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		//Bitonic sort stage N - loop until final stage is reached, sorting larger sizes of bitonic groups

		//initialise buffer for kernel
		cl::Buffer buffer_stage(context, CL_MEM_READ_ONLY, 1 * sizeof(int));

		//profiling events
		cl::Event stage_upload_event;
		cl::Event stage_download_event;

		//create kernel
		cl::Kernel kernel_bitonic_sortn = cl::Kernel(program, "bitonic_nmerge");

		//set kernel arguments
		kernel_bitonic_sortn.setArg(0, buffer_sort_temps);

		
		//loop calling kernel with different stage
		for (int stages = 0; stages < log2(sortVec.size()); stages++)
		{
			//write buffer to device
			queue.enqueueFillBuffer(buffer_stage, stages, 0, 1 * sizeof(int));
			//set kernel argument
			kernel_bitonic_sortn.setArg(1, buffer_stage);
			//start kernel
			queue.enqueueNDRangeKernel(kernel_bitonic_sortn, cl::NullRange, cl::NDRange(sortVec.size()), cl::NullRange, NULL, &prof_event);
			//wait for queue to finish before moving to next stage
			queue.finish();
			cout << "\n\nBitonic stage " << stages+1 <<  " kernel timings:" << endl;
			cout << "\nKernel started" << endl;
			cout << "Queued time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
			cout << "Submitted time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() << endl;
			cout << "Kernal execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
			cout << "Total kernel time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
			cout << "\nTotal time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
			
		}

		//event for profiling
		cl::Event sorted_download_event;

		//create final stage kernel to sort final bitonic sequence
		cl::Kernel kernel_bitonic_finish = cl::Kernel(program, "bitonic_merge_final");

		//set kernel argument
		kernel_bitonic_finish.setArg(0, buffer_sort_temps);

		//start kernel
		queue.enqueueNDRangeKernel(kernel_bitonic_finish, cl::NullRange, cl::NDRange(padded_sort_temps.size()), cl::NullRange, NULL, &prof_event);

		//retrieve sorted vector from device
		queue.enqueueReadBuffer(buffer_sort_temps, CL_TRUE, 0, sortVec.size() * sizeof(float), &sortVec[0], NULL, &sorted_download_event);

		cout << "\n\nFinal bitonic stage kernel timings:" << endl;
		cout << "\nKernel started" << endl;
		cout << "Queued time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "Submitted time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() << endl;
		cout << "Kernal execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Total kernel time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << endl;
		cout << "\nFinal stage download [ns]: " << sorted_download_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sorted_download_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nTotal time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		cout << "\n\nTotal bitonic sort time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sort_temps_upload_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		//remove extra padding from sorted array
		vector<float> final(temps.size());
		for (int i = 0; i < temps.size(); i++) {
			final[i] = sortVec[i];
		}

		//show sorted list if argument is set
		if (show_sorted)
		{
			cout << "Sorted List" << final << endl;
		}

		//calculate median, 1st and 3rd quartiles from sorted list
		float median;
		float upperQ;
		float lowerQ;
		if (final.size() % 2)
		{
			median = (final[ceil(final.size() / 2)] + final[floor(final.size() / 2)]) / 2;
			vector<float> upperSet(final.size() / 2);
			vector<float> lowerSet(final.size() / 2);
			for (int i = 0; i < upperSet.size(); i++) {
				lowerSet[i] = final[i];
				upperSet[i] = final[i + upperSet.size()];
			}
			if (upperSet.size() % 2)
			{
				upperQ = (upperSet[ceil(upperSet.size() / 2)] + upperSet[floor(upperSet.size() / 2)]) / 2;
				lowerQ = (lowerSet[ceil(upperSet.size() / 2)] + lowerSet[floor(lowerSet.size() / 2)]) / 2;


			}
			else
			{
				upperQ = upperSet[(upperSet.size() + 1) / 2];
				lowerQ = lowerSet[(lowerSet.size() + 1) / 2];


			}
			
		}
		else
		{
			median = final[(final.size() + 1) / 2];
			vector<float> upperSet((final.size()-1) / 2);
			vector<float> lowerSet((final.size() -1)/ 2);
			for (int i = 0; i < upperSet.size(); i++) {
				lowerSet[i] = final[i];
				upperSet[i] = final[i + upperSet.size()+1];
			}
			if (upperSet.size() % 2)
			{
				upperQ = (upperSet[ceil(upperSet.size() / 2)] + upperSet[floor(upperSet.size() / 2)]) / 2;
				lowerQ = (lowerSet[ceil(upperSet.size() / 2)] + lowerSet[floor(lowerSet.size() / 2)]) / 2;


			}
			else
			{
				upperQ = upperSet[(upperSet.size() + 1) / 2];
				lowerQ = lowerSet[(lowerSet.size() + 1) / 2];


			}
		}
		
		//output stats
		cout << "\n\nMean = " << mean_val << endl;
		cout << "Max = " << max_val << endl;
		cout << "Min = " << min_val << endl;
		cout << "Varience = " << variance_value << endl;
		cout << "Standard Deviation = " << standard_deviation_val << endl;
		cout << "1st Quartile = " << lowerQ << endl;
		cout <<"Meadian = " << median << endl;
		cout << "3rd Quatile = " << upperQ << endl;

	}
	//catch any errors produced by OpenCL API
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	return 0;
}


