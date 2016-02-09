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
	
	//How big will the pixel buffer need to be in order to process a frame at a time?
	int rows[(K*(K-1)) + K];
	int count = -((K*(K-1)) + K);
	
	while(count != 227*227*3) {
		//What will happen in this while loop
		//1. Shift new pixel into the buffer based on stride (1 for now)
		//2. Once enough have been shifted in, calculate output with weights on the frame
		//3. Repeat
		
		#pragma unroll
		for(int i = 227; i > 0; --i) {
			rows[i] = rows[i-1];
		}
		rows[0] = count >= 0 ? in[count] : 0;
		
		//Initialize any variables needed within unroll 
		
		#pragma unroll
		for(int i = 0; i < (width of weight); ++i) {
			#pragma unroll
			for(int j = 0; j < (height of weight); ++j) {
				//read data from in buffer and work with it
				//Add/multiply weights etc
			}
		}
		
		//Apply bias here?
	}
	
}

/*const int Win = 227, Hin = 227, N = 3, M = 96, K = 11, S = 4, pad = 0;
int Wout = (Win + 2 * pad - K) / S + 1;
int	Hout = (Hin + 2 * pad - K) / S + 1;
int w = 0, h = 0, m, n, i, j; 

for(w = 0; w < Wout; w++ ) { //Wout
	for(h = 0; h < Hout; h++ ) { //Hout
		for(m = 0; m < M; m++ ) { //M
			for(n = 0; n < N; n++ ) {
				for(i = 0; i < K; i++ ) {
					for(j = 0; j < K; j++){
						if((w*S+i-pad<0) || (w*S+i-pad >= Win) ||
							(h*S+j-pad < 0) || (h*S+j-pad >= Hin)) continue;
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