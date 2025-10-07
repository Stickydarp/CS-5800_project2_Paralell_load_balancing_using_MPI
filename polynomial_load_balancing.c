#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define MAX 50000
#define COEFFICIENT 1
#define WORK_TAG 1
#define TERMINATE_TAG 2



double power(double x, int degree)
{     
      if(degree == 0)  return 1;
      
      if(degree == 1)  return x;

      return x * power(x, degree - 1);
}

void run_master(int nprocs, int*coeffArr, int N ,double x, int chunk_size){
	int next = 0;
	int num_workers = nprocs - 1;
    	int workers_done = 0;
	double global_sum = 0.0;

    	// 1. seed initial work (2 items per worker or chunk_size)
    	for (int w = 1; w <= num_workers && next < N; w++) {
        	int count = ( (N - next) < chunk_size ? (N - next) : chunk_size );
        	MPI_Send(&count, 1, MPI_INT, w, WORK_TAG, MPI_COMM_WORLD);
        	MPI_Send(&next, 1, MPI_INT, w, WORK_TAG, MPI_COMM_WORLD);
       
		MPI_Send(coeffArr + next, count, MPI_INT, w, WORK_TAG, MPI_COMM_WORLD);
        	next += count;
    	}

    	// 2. dynamic handing-out loop
    	while (workers_done < num_workers) {
        	MPI_Status status;
        	double partial;
        	MPI_Recv(&partial, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                MPI_COMM_WORLD, &status);
        	int worker = status.MPI_SOURCE;

        	global_sum += partial;

        	if (next < N) {
            		int count = ((N - next) < chunk_size ? (N - next) : chunk_size);
            		MPI_Send(&count, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
            		MPI_Send(&next, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
            		MPI_Send(coeffArr + next, count, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
            		next += count;
        	} else {
            	int zero = 0;
            	MPI_Send(&zero, 1, MPI_INT, worker, TERMINATE_TAG, MPI_COMM_WORLD);
            	workers_done++;
        	}
    	}

    	// After loop, all workers have been terminated, global_sum is final result
    	printf("Master: final result = %f\n", global_sum);
	
	
	
}

void run_worker(int rank, double x){
	while(1){
		MPI_Status status;
		int count;
		MPI_Recv(&count, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == TERMINATE_TAG) {
            		// termination signal
            		break;
		}else{
			// got work
			int start;
			MPI_Recv(&start, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, &status);//recieve start
			int *buf = malloc(sizeof(int) * count);
			MPI_Recv(buf, count, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, &status);//recieve buffer

			double partial = 0.0;
			int end = start + count;
			for (int i = start; i < end; i++) {
				partial += buf[i-start] * power(x,i);
			}
			free(buf);
			MPI_Send(&partial, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);	
		}
	}	





}




double sequential(int coeffArr[], double x)
{
   int maxDegree = MAX - 1;
   int i;
   double  answer = 0;
   
   for( i = 0; i < maxDegree;  i++)
   {
      
      double powerX = power(x, i);

      //printf("%f ", powerX);
      answer = answer + coeffArr[i] * powerX;
   }
   return answer;
 }




void initialize(int coeffArr[])
{
   int maxDegree = MAX - 1;
   int i;
   for( i = 0; i < maxDegree; i++)
   {
      coeffArr[i] = COEFFICIENT;
   }
}

// Driver Code
int main(int argc, char **argv){

	MPI_Init(&argc, &argv);
    	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    	int N = MAX;  // or pass N via argv
    	double x = 0.99;
    	int chunk_size = 5;  // seed count, or configurable

    	int *coeffArr = NULL;
    	if (rank == 0) {
        	// master allocates and initializes
        	coeffArr = malloc(sizeof(int) * N);
        	for (int i = 0; i < N; i++) {
            		coeffArr[i] = COEFFICIENT;
        	}
    	}

    	MPI_Barrier(MPI_COMM_WORLD);
    	double t0 = MPI_Wtime();

    	if (rank == 0) {
        	run_master(nprocs, coeffArr, N, x, chunk_size);
   	} else {
        	run_worker(rank, x);
    	}

    	MPI_Barrier(MPI_COMM_WORLD);
    	double t1 = MPI_Wtime();
    	if (rank == 0) {
        	printf("Total parallel time = %f seconds\n", t1 - t0);
    	}

    	if (coeffArr) free(coeffArr);








	MPI_Finalize();
    
	return 0;
}
