//enable use of long for large dataset
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

//calculates the sum of the input
kernel void meanf(global const float* A, global int* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	//multiply by 10 to turn all decimals in ints so no data is lossed when using atomic_add
	scratch[lid] = A[id]*10.0f;
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	//partial sum for work group
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (lid == 0) {
		int intVal = convert_int(scratch[lid]);
		atomic_add(&B[0],intVal);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
}

//calculates partial maxes and mins of array
kernel void maxminf(global const float* A, global float* B, global float* C, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	
	//cache all N values from global memory to local memory
	scratch[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	//calculate partial max and min
	if (lid == 0){
		float max = scratch[lid];
		float min = scratch[lid];
	
		for (int i = 1; i < N; i++)
		{
			if (scratch[i] > max){
				max = scratch[i];
			}
			if (scratch[i] < min){
				min = scratch[i];
			}
		}
		B[get_group_id(0)] = max;
		C[get_group_id(0)] = min;
	}

}


//calculate squared difference
kernel void variance(global const float* A, global long* B, global float* mean, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	////calculate each squared difference and store in local memory
	float diff = A[id] - mean[0];
	float power = convert_float(pow(diff, 2.0f));
	scratch[lid] = power * 100.0f;//multiply by 100 to preserve decimals when converting to int
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	//calculate partial sums
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	////we add results from all local groups to the first element of the array
	////serial operation! but works for any group size
	////copy the cache to output array
	if (lid == 0) {
		long intVal = convert_long(scratch[lid]);
		atom_add(&B[0],intVal);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

//compare and exchange values 
void cmpxchg(global float* A, global float* B, bool dir) {
	if ((!dir && *A > *B) || (dir && *A < *B)) {
		float t = *A;
		*A = *B;
		*B = t;
	}
}



//create bitonic stage 0
kernel void bitonic_initial(global float* A){
	int id = get_global_id(0);
	//compare pairs of values and alternate direction 
	if (id/2 % 2 == 0){
		cmpxchg(&A[id],&A[id+1],false);
	}
	else if (id % 2 == 0){
		cmpxchg(&A[id],&A[id+1],true);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	
}

//calculate intermidiate stage n
kernel void bitonic_nmerge(global float* A, global int* stage){
	int gs = get_global_size(0);
	//calculate the number of bitonic sequences for current stage
	float base = 2.0;
	int stride = convert_int(pown(base, stage[0]));
	int numBitonicGroups = gs / (stride * 4);
	int N = gs / numBitonicGroups;
	
	int id = get_global_id(0);
	bool dir = false;
	barrier(CLK_GLOBAL_MEM_FENCE);
	//sort each bitonic sequence in the stage into either ascending or descending or to create larger bitonic sequnces
	if (get_global_id(0)/(gs/(numBitonicGroups)) % 2 == 0 && get_global_id(0) % (gs/(numBitonicGroups)) == 0){
		for (int i = N/2; i > 0; i/=2) {
			for (int j = 0; j < N-i; j++){
				if ((id % (i*2)) < i)
				{
					cmpxchg(&A[j+id],&A[j+i+id],dir);
				}
			}
		}
	}
	else if (get_global_id(0) % (gs/(numBitonicGroups)) == 0){
		bool dir = true;
		for (int i = N/2; i > 0; i/=2) {
			for (int j = 0; j < N-i; j++){
				if ((id % (i*2)) < i)
				{
					cmpxchg(&A[j+id],&A[j+i+id],dir);
				}
			}
		}
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
}

//produce final sorted list
kernel void bitonic_merge_final(global float* A){
	int id = get_global_id(0);
	int N = get_global_size(0);
	//perform the final bitonic merge to sort the list
	if (id == 0){
		bool dir = false;
		barrier(CLK_GLOBAL_MEM_FENCE);
		for (int i = N/2; i > 0; i/=2) {
			for (int j = 0; j < get_global_size(0)-i; j++){
				barrier(CLK_GLOBAL_MEM_FENCE);
				if ((id % (i*2)) < i)
				{
					cmpxchg(&A[j],&A[j+i],dir);
				}
				barrier(CLK_GLOBAL_MEM_FENCE);
			}
		}
	}
}