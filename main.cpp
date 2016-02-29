//Main.cpp
//Christopher Bird
//January-March, 2016
//Convolution Layer One
//Fast CNN on FPGA
//Hardware: DE1-SoC FPGA board
//NOTE: Program is designed to work with a variable number of devices, but currently only runs on 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

//OpenCL required libraries
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

//OpenCL runtime config; important variables
cl_platform_id platform = NULL;
unsigned num_devices = 0; 						//For number of different FPGA/CPU/GPU devices, for us: 1
unsigned values = 154587;						//Will change soon

//Arrays for Devices and Command Queues
scoped_array<cl_device_id> device;				//How many elements per device?
scoped_array<cl_command_queue> kernel_queue;	//One entry per device

//Context and Program Variables
cl_context context = NULL;
cl_program program = NULL;

//Arrays for Memory and Kernel elements
scoped_array<cl_kernel> kernel;
scoped_array<cl_mem> input_a_buffer;			//Image elements per device
scoped_array<cl_mem> input_b_buffer;			//Weight elements per device; 	this and the following will
scoped_array<cl_mem> input_c_buffer;			//Bias elements per device; 	soon be destroyed
scoped_array<cl_mem> output_buffer;				//Stores output of each kernel

//Problem Data
unsigned N = 0;									//Command line options for changing problem if needed
scoped_array<scoped_aligned_ptr<float> >	input_a, input_b, input_c;	//Stores image, weight, bias data from input
scoped_array<scoped_aligned_ptr<float> > output;	//Stores output values 
scoped_array<scoped_array<float> > ref_output;	//Unused, but can be used to check values before completion
scoped_array<unsigned> n_per_device;			//Sets number of values used per device based on available

//Function Prototypes
bool init_opencl();								//Initiates context, program, kernels, and command queues
void init_problem();							//Initiates problem data, including reading values from input
void run();										//Enqueue and begin execution of kernels
void cleanup();									//Destroy all traces of program from memory

//------------------ main Function ------------------

int main(int argc, char **argv) {				//Check and use input arguments if present
	Options options(argc, argv);				//Creates options objects to store input parameters
	if(options.has("n")) {
		N = options.get<unsigned>("n");			//If n in params, take value for use later, currently unneeded
	}
	
	if(!init_opencl()) {						//Execute init_opencl and return if it fails
		return -1;
	}
	
	init_problem();								//Read in and distribute data from input
	run();										//Enqueue and begin execution of kernels	
	cleanup();									//Destroy all traces of program from memory
	return 0;
}

//------------------ End main Function ------------------

//------------------ init_opencl Function ------------------

bool init_opencl() {
	cl_int status;								//Status variable for use with OpenCL success commands
	
	printf("Initializing OpenCL\n");
	
	if(!setCwdToExeDir()) {						//Tells compiler that working directory is the same
		return false;							//as the folder with the executable
	}

	platform = findPlatform("Altera");			//Find and set the OpenCL platform
	if(platform == NULL) {
		printf("ERROR: Unable to find Altera OpenCL platform.\n");
		return false;							
	}
	
	device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));	//Fill device array with available devices
	printf("Platform: %s\n", getPlatformName(platform).c_str());
	printf("Using %d device(s)\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) {
		printf("	%s\n", getDeviceName(device[i]).c_str());				//Print to user for verification
	}
	
														//Placing all devices onto the same context
	context = clCreateContext(	NULL, 					//Properties; Can choose to force certain type of context, NULL gives default 
								num_devices, 			//# of Devices; How many devices to put in the context
								&device[0],				//List of Devices
								&oclContextCallback,	//Callback List; allows program to see errors in making context
								NULL, 					//User data added to context
								&status);				//Error code if not successful
	checkError(status, "Failed to create context");
	
	std::string binary_file = getBoardBinaryFile("layer_one_conv", device[0]);	//Sets the .cl file to use for implementation
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(	context, 				//Which context to create in
										binary_file.c_str(), 	//Which .cl to use
										&device[0],				//List of devices to make it on 
										num_devices);			//How many devices to program from list
	
	status = clBuildProgram(	program,	//Specify which program to use
								0, 			//If not NULL, specify which device to build on
								NULL, 		//Give list of devices if needed
								"", 		//Give options if needed
								NULL, 		//If NULL, wait till build is complete to return, otherwise return
								NULL);		//User data if needed
	checkError(status, "Failed to build program.");
	
	kernel_queue.reset(values);					//Reset all scoped arrays to ensure clean data
	kernel.reset(values);
	n_per_device.reset(values);
	input_a_buffer.reset(values);
	input_b_buffer.reset(values);
	input_c_buffer.reset(values);
	output_buffer.reset(values);

	for(unsigned i = 0; i < num_devices; ++i) {
		kernel_queue[i] = clCreateCommandQueue(	context,					//Which context to find device in
												device[i], 					//Which device in that context
												CL_QUEUE_PROFILING_ENABLE, 	//Profile_enable or out_of_order
												&status);					//Error code if not successful
		checkError(status, "Failed to create command queue");
	
		const char *kernel_name = "layer_one";
		kernel[i] = clCreateKernel(	program, 		//Program to execute kernel
									kernel_name, 	//Which kernel to create
									&status);		//Error code if not successful
		checkError(status, "Failed to create kernel");
		
		n_per_device[i] = values;					//How many values this device will use, CHANGE this
		
		input_a_buffer[i] = clCreateBuffer(	context,							//Context to build in
											CL_MEM_READ_ONLY,					//Which type of buffer
											n_per_device[i] * sizeof(float),	//Size of the buffer
											NULL,								//Pointer to buffer if already allocated
											&status);							//Error cored if not successful
		checkError(status, "Failed to create buffer for input A");
		
		input_b_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, n_per_device[i] * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for input B");
		
		input_c_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, n_per_device[i] * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for input C");
		
		output_buffer[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_per_device[i] * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for output data");
	}
	
	return true;
}

//------------------ End init_opencl Function ------------------

//------------------ init_problem Function ------------------

void init_problem() {
	if(num_devices == 0) {								//Check for devices
		checkError(-1, "No Devices");
	}
	
	input_a.reset(values);								//Reset input buffers
	input_b.reset(values);
	input_c.reset(values);
	output.reset(values);
	
	std::ifstream myfile, myfile2, myfile3;				//Open input data files for reading
	myfile.open("bin/imagedata.txt");
	myfile2.open("bin/imageBias.txt");
	myfile3.open("bin/imageKernels.txt");
	
	float data, bias, weight;
	
	for(unsigned i = 0; i < num_devices; ++i) {			//Prepare input arrays for data
		input_a[i].reset(n_per_device[i]);
		input_b[i].reset(n_per_device[i]);
		input_c[i].reset(n_per_device[i]);
		output[i].reset(n_per_device[i]);
		
		for(unsigned j = 0; j < n_per_device[i]; ++j) {	//Read in all data
			myfile >> data;
			myfile2 >> bias;
			myfile3 >> weight;
			input_a[i][j] = data;
			input_b[i][j] = bias;
			input_c[i][j] = weight;
		}
	}
}

//------------------ End init_problem Function ------------------

//------------------ run Function ------------------

void run() {
	cl_int status;
	const double start_time = getCurrentTimestamp();
	
	scoped_array<cl_event> kernel_event(values);			//List of kernel events
	scoped_array<cl_event> kernel_finish_event(values);		//List of finished kernel events
	scoped_array<cl_event> write_event(3);					//Overall event waitlist
	
	for(unsigned i = 0; i < num_devices; ++i) {
		status = clEnqueueWriteBuffer(	kernel_queue[i],					//Command queue to look in
										input_a_buffer[i],					//Buffer to write to
										CL_FALSE,							//Blocking?
										0,									//Offset in the buffer
										n_per_device[i] * sizeof(float),	//Size of bytes to write
										input_a[i],							//Pointer to host data 
										0,									//Events in waitlist 
										NULL,								//Wait list pointer 
										&write_event[0]);					//This Event 
		checkError(status, "Failed to transfer Input A");
		
		status = clEnqueueWriteBuffer(kernel_queue[i], input_b_buffer[i], CL_FALSE, 0, 
			n_per_device[i] * sizeof(float), input_b[i], 0, NULL, &write_event[1]);
		checkError(status, "Failed to transfer input B");
		
		status = clEnqueueWriteBuffer(kernel_queue[i], input_c_buffer[i], CL_FALSE, 0,
			n_per_device[i] * sizeof(float), input_c[i], 0, NULL, &write_event [2]);
		checkError(status, "Failed to transfer input C");
		
		for(unsigned j = 0; j < num_devices; ++j) {			//Wait for all previous events to finish before continuing
			clFinish(kernel_queue[j]);
		}
		
		unsigned argi = 0;									//Begin setting parameters for the kernel
		
		status = clSetKernelArg(	kernel[i],				//Which kernel to set argument
									argi++, 				//Which index of argument to set
									sizeof(cl_mem), 		//Size of argument data
									&input_a_buffer[i]);	//What to pass as argument
		checkError(status, "Failed to set argument %d", argi - 1);
		
		status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buffer[i]);
		checkError(status, "Failed to set argument %d", argi - 1);
		
		status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_c_buffer[i]);
		checkError(status, "Failed to set argument %d", argi - 1);
		
		status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buffer[i]);
		checkError(status, "Failed to set argument %d", argi - 1);
		
		const size_t global_work_size = n_per_device[i];	//Setting global worksize for current kernel
		const size_t one = 1;
		printf("Launching for device %d (%d elements)\n", i, global_work_size);
		
		status = clEnqueueNDRangeKernel(kernel_queue[i],	//Which command queue
										kernel[i],			//Which kernel 
										1,					//Work dimensions: 3D map of kernels
										NULL,				//Global Work Offset: Not used in OCL 1.2 
										&one,				//Global Work Size: How many kernel copies
										NULL,				//Local Work Size: Let compiler choose
										3,					//Events in waitlist: writing to buffers 
										write_event,		//Link to event list 
										&write_event[i]);	//This Event
		checkError(status, "Failed to launch kernel");

		status = clEnqueueReadBuffer(	kernel_queue[i],				//Command Queue
										output_buffer[i],				//Buffer to read 
										CL_FALSE,						//Blocking?
										0,								//Offset 
										n_per_device[i] *sizeof(float),	//Bytes to read
										output[i],						//Memory to write too
										1,								//Events in waitlist 
										&write_event[i],				//Pointer to waitlist
										&kernel_finish_event[i]);		//This event
		checkError(status, "Failed to read output buffer");
		
		clReleaseEvent(write_event[0]);
		clReleaseEvent(write_event[1]);
		clReleaseEvent(write_event[2]);
	}
	
	clWaitForEvents(num_devices, kernel_finish_event);
	clWaitForEvents(num_devices, write_event);
	
	printf("Timing Finish.");
	
	const double end_time = getCurrentTimestamp();
	
	printf("\nFinal Time: %0.3f ms\n", (end_time - start_time) * 1e3);
	
	cl_ulong time_start, time_end;
	double total_time;
	for(int i = 0; i < num_devices; ++i) {
		clGetEventProfilingInfo(kernel_event[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(kernel_event[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = time_end - time_start;
		printf("\nKernel %d, Execution time in miliseconds = %0.3f ms\n", i, (total_time / 1000000.0));
	}
	
	for(unsigned i = 0; i < num_devices; ++i) {
		clReleaseEvent(kernel_event[i]);
		clReleaseEvent(kernel_finish_event[i]);
	}
	
	for(unsigned i = 0; i < num_devices; ++i) {
		for(unsigned j = 0; j < 20; ++j) {
			printf("%3.4f ", output[i][j]);
		}
		printf("\n");
	}
}

//------------------ End run Function ------------------

//------------------ cleanup Function ------------------

void cleanup() {
	for(unsigned i = 0; i < num_devices; ++i) {
		if(kernel && kernel[i]) {						//Destroy kernels
			clReleaseKernel(kernel[i]);
		}
		if(kernel_queue && kernel_queue[i]) {			//Destroy kernel queue
			clReleaseCommandQueue(kernel_queue[i]);
		}
		if(input_a_buffer && input_a_buffer[i]) {		//Destroy all memory buffers
			clReleaseMemObject(input_a_buffer[i]);
		}
		if(input_b_buffer && input_b_buffer[i]) {
			clReleaseMemObject(input_b_buffer[i]);
		}
		if(input_c_buffer && input_c_buffer[i]) {
			clReleaseMemObject(input_c_buffer[i]);
		}
		if(output_buffer && output_buffer[i]) {
			clReleaseMemObject(output_buffer[i]);
		}
	}
	
	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}
}

//------------------ End cleanup Function ------------------