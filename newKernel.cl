//layer_one_conv.cl
//Christopher Bird
//January-March, 2016
//Convolution Layer One
//Fast CNN on FPGA
//Hardware: DE1-SoC FPGA board
//NOTE: Program is designed to work with a variable number of devices, but currently only runs on 1

//ACL Kernel for computing first convolution layer of AlexNet CNN

__kernel void layer_one_conv (	__global const float *in,
								__global const float *weight,
								__global const float *bias,
								__global float *restrict out) 
{
	//in and out are matricies the size of the image, 227 x 227 x 3
	//Weight is a 4D matrix sized 11 x 11 x 3 x 96
	//Bias is a 4D matrix sized 1 x 1 x 1 x 3
	
	const int Win = 227, Hin = 227, N = 3, M = 96, K = 11, S = 4, pad = 0;
	int Wout = (Win + 2 * pad - K) / S + 1;
	int	Hout = (Hin + 2 * pad - K) / S + 1;
	int w = 0, h = 0, m, n, i, j; 
	
	for(i = 0; i < K; i++) {							//Since i and j are generally small (between 0-11)
		for(j = 0; j < K; j++) {						//They do not need to be unrolled
			for(w = 0; w < Wout; w++) {	
				for(h = 0; h < Hout; h++) {
					#pragma unroll
					for(m = 0; m < M, m++) {
						#pragma unroll
						for(n = 0; n < N; n++) {
							#pragma unroll 
							{
								out[(w)+(Wout*h)+(Wout*Hout*m)] += 					//output_fm
								in[(w*S+i-pad)+Win*(h*S+j-pad)+(Win*Hin*n)] *		//input_fm
								weight[(i)+(K*j)+(K*K*n)+(K*K*N*m)];				//weights
							}
						}
					}
				}
			}
		}
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
