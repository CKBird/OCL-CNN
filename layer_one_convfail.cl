//layer_one_conv.cl
//Christopher Bird
//January-March, 2016
//Convolution Layer One
//Fast CNN on FPGA
//Hardware: DE1-SoC FPGA board
//NOTE: Program is designed to work with a variable number of devices, but currently only runs on 1

//ACL Kernel for computing first convolution layer of AlexNet CNN

#define imageSize 154587 	//227 * 227 * 3
#define weightSize 34848 	//11 * 11 * 3 * 96
#define biasSize 3			//1 * 1 * 1 * 3
#define windowSize 100		//
#define shiftSize 100		//Size needed for shift register to function properly

__kernel void layer_one_conv (	__global const float *restrict g_in,
								__global const float *restrict g_weight,
								__global const float *restrict g_bias,
								__global float *restrict g_out) 
{
	//in and out are matricies the size of the image, 227 x 227 x 3
	//Weight is a 4D matrix sized 11 x 11 x 3 x 96
	//Bias is a 4D matrix sized 1 x 1 x 1 x 3
	
	const int Win = 227, Hin = 227, N = 3, M = 96, K = 11, S = 4, pad = 0;
	int Wout = (Win + 2 * pad - K) / S + 1;
	int	Hout = (Hin + 2 * pad - K) / S + 1;
	int w = 0, h = 0, m, n, i = 0, j = 0; 
	int wIndex = 0;
	int iIndex = 0; 
	
	float in[imageSize];
	float weight[weightSize];
	float bias[biasSize];
	float out[imageSize];
	
	/*for(iIndex = 0; iIndex < imageSize; iIndex++) {
		in[iIndex] = g_in[iIndex];
	}
	for(wIndex = 0; wIndex < weightSize; wIndex++) {
		weight[wIndex] = g_weight[wIndex];
	}*/
	
	/*for(w = 0; w < Wout; w++) {
		for(h = 0; h < Hout; h++) {
			for(m = 0; m < M; m++) {
				for(n = 0; n < N; n++) {
					in[w + h*Wout + Wout*Hout*m] = g_in[w + h*Wout + Wout*Hout*m];
					weight[wIndex] = g_weight[wIndex];
					wIndex++;
					//bias[] = g_bias[];
				}
			}
		}
	}*/
	
	/*for(wIndex = 0; wIndex < weightSize; wIndex++) {	//Moving from global to local memory
		weight[wIndex] = g_weight[wIndex];
	}
	
	#pragma unroll
	for(iIndex = 0; iIndex < imageSize; iIndex++) {		//Setting Shift Register Values to 0
		in[iIndex] = 0;
	}
	
	while(j < Hout) {	
		#pragma unroll
		for(iIndex = 1; iIndex < shiftSize; iIndex++) {	//Filling shift register with values
			in[iIndex - 1] = in[iIndex];
		}
		in[imageSize - 1] = g_in[]
	}*/
	

	for(w = 0; w < Win; w++) {
		for(h = 0; h < Hin; h++) {
			for(m = 0; m < M; m++) {
				for(n = 0; n < N; n++) {
					
					
					
					for(i = 0; i < K; i++) {							//Since i and j are generally small (between 0-11)
						for(j = 0; j < K; j++) {						//They do not need to be unrolled
							for(w = 0; w < Wout; w++) {	
								for(h = 0; h < Hout; h++) {
									//#pragma unroll
									for(m = 0; m < M; m++) {
										#pragma unroll
										for(n = 0; n < N; n++) {
											#pragma unroll 
												out[(w) + (Wout * h) + (Wout * Hout * m)] += 							//output_fm
												in[(w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n)] *		//input_fm
												weight[(i) + (K * j) + (K * K * n) + (K * K * N * m)];					//weights
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
		
	
	
	
	for(int oIndex = imageSize; oIndex > 0; oIndex--) {
		g_out[oIndex] = out[oIndex];
	}
}

/*const int Win = 227, Hin = 227, N = 3, M = 96, K = 11, S = 4, pad = 0;
int Wout = (Win + 2 * pad - K) / S + 1;
int	Hout = (Hin + 2 * pad - K) / S + 1;
int w = 0, h = 0, m, n, i, j; 

for(w = 0; w < Wout; w++ ) { //Wout				//row
	for(h = 0; h < Hout; h++ ) { //Hout			//col
		for(m = 0; m < M; m++ ) { //M			//to
			for(n = 0; n < N; n++ ) {			//ti
				for(i = 0; i < K; i++ ) { 		//i	
					for(j = 0; j < K; j++){ 	//j
						out[(w)+(Wout*h)+(Wout*Hout*m)] += 
						in[(w*S+i-pad)+Win*(h*S+j-pad)+(Win*Hin*n)] *
						weight[(i)+(K*j)+(K*K*n)+(K*K*N*m)];
					}
				}
			}
		}
	}
}

for(w = 0; w < Wout; w++) {
	for( h = 0; h < Hout; h++) {
		for( m = 0; m < M; m++) {
			out[(w)+(Wout*h)+(Wout*Hout*m)]+=bias[m];
		}
	}
}*/
