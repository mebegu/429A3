/*
 * noise_remover.cpp
 *
 * This program removes noise from an image based on Speckle Reducing Anisotropic Diffusion
 * Y. Yu, S. Acton, Speckle reducing anisotropic diffusion,
 * IEEE Transactions on Image Processing 11(11)(2002) 1260-1270 <http://people.virginia.edu/~sc5nf/01097762.pdf>
 * Original implementation is Modified by Burak BASTEM and Nufail Farooqi
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "mpi.h"

#define MATCH(s) (!strcmp(argv[ac], (s)))

// returns the current time
static const double kMicro = 1.0e-6;
double get_time() {
	struct timeval TV;
	struct timezone TZ;
	const int RC = gettimeofday(&TV, &TZ);
	if(RC == -1) {
		printf("ERROR: Bad call to gettimeofday\n");
		return(-1);
	}
	return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}

void update_image_ghostcells(unsigned char *image, int height, int width);
void update_coeff_ghostcells(float *coeff, int height, int width);

int main(int argc, char *argv[]) {
	int numprocs, rank, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &namelen);

	// Part I: allocate and initialize variables
	double time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8;	// time variables
	time_0 = get_time();
	const char *filename = "input.pgm";
	const char *outputname = "output.png";
	int width, height, pixelWidth;
	long n_pixels,n_pixels_with_ghost;
	int n_iter = 50;
	float lambda = 0.5;
	float mean, variance, std_dev;	//local region statistics
	float *north_deriv, *south_deriv, *west_deriv, *east_deriv;	// directional derivatives
	double tmp, sum, sum2;	// calculation variables
	float gradient_square, laplacian, num, den, std_dev2, divergence;	// calculation variables
	float *diff_coef;	// diffusion coefficient
	float *temp_diff_coef;
	float diff_coef_north, diff_coef_south, diff_coef_west, diff_coef_east;	// directional diffusion coefficients
	long k, k2;	// current pixel index
	int px=0, py=0;
	time_1 = get_time();

	// Part II: parse command line arguments
	if(argc<2) {
	  printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>][-x <processor geometery in x>] [-y <processor geometery in y>]\n",argv[0]);
	  return(-1);
	}
	for(int ac=1;ac<argc;ac++) {
		if(MATCH("-i")) {
			filename = argv[++ac];
		} else if(MATCH("-iter")) {
			n_iter = atoi(argv[++ac]);
		} else if(MATCH("-x")) {
		  px = atoi(argv[++ac]);
		} else if(MATCH("-y")) {
		  py = atoi(argv[++ac]);
		} else if(MATCH("-l")) {
			lambda = atof(argv[++ac]);
		} else if(MATCH("-o")) {
			outputname = argv[++ac];
		} else {
		  printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>] [-x <processor geometery in x>] [-y <processor geometery in y>]\n",argv[0]);
		return(-1);
		}
	}
	time_2 = get_time();

	unsigned char *image;
	unsigned char *loc_image;
	unsigned char *tmp_image;
	int loc_width;
	int loc_height;
	int rem_height;
	int count2 = 0;
	int count1 = 0;


	// Part III: read image
	if(rank == 0){
		printf("Reading image...\n");
		tmp_image = stbi_load(filename, &width, &height, &pixelWidth, 0);
		if (!tmp_image) {
			fprintf(stderr, "Couldn't load image.\n");
			return (-1);
		}
		printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);
		n_pixels = height * width;
		n_pixels_with_ghost = (height+2) * (width+2);

		// Copy the image into extended image array(with ghost cells)
		image = (unsigned char*) malloc(sizeof(unsigned char) * n_pixels_with_ghost);
		for (int i = 1; i <= height ; i++) {
		  for (int j = 1; j <= width ; j++) {
		    k = i * (width+2) + j;	// position of current element
		    k2 = (i-1) * width + (j-1);

		    image[k] = tmp_image[k2];
		  }
		}


	}

	MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int *displs =  (int *) malloc(sizeof(int)*numprocs);
	int *loc_sizes = (int *) malloc(sizeof(int) *  numprocs);

	int disp = 0;
	int loc_size = (width+2)*((height/numprocs) + 2);
	int loc_disp = (width+2) * (height/numprocs);


	for(int i=0; i< numprocs; i++){
		loc_sizes[i] = loc_size;
		if (i != numprocs-1)
			 displs[i+1] = (i+1) * loc_disp + disp * (width+2);

		if (i < height % numprocs) {
			loc_sizes[i] += width+2;
			disp++;
			//if(i == rank) loc_height++;
		}
	}
	printf("local sizes %d Rank: %d\n",loc_sizes[rank], rank);

	loc_image = (unsigned char*) malloc(sizeof(unsigned char) * loc_sizes[rank]);

	MPI_Scatterv(image, loc_sizes, displs, MPI_UNSIGNED_CHAR, loc_image, loc_sizes[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	/*printf("sum %f Rank: %d\n",sum, rank);
	printf("sum2 %f Rank: %d\n",sum2, rank);*/


	time_3 = get_time();


	// Part IV: allocate variables
	loc_height = loc_sizes[rank] / (width+2);
	int num_ghosts = 2 * width + 2 * loc_height;
	north_deriv = (float*) malloc(sizeof(float) * (loc_sizes[rank] - num_ghosts));	// north derivative
	south_deriv = (float*) malloc(sizeof(float) * (loc_sizes[rank] - num_ghosts));	// south derivative
	west_deriv = (float*) malloc(sizeof(float) * (loc_sizes[rank] - num_ghosts));	// west derivative
	east_deriv = (float*) malloc(sizeof(float) * (loc_sizes[rank] - num_ghosts));		// east derivative
	diff_coef  = (float*) malloc(sizeof(float) * loc_sizes[rank]);	// diffusion coefficient
	temp_diff_coef  = (float*) malloc(sizeof(float) * loc_sizes[rank]);	// diffusion coefficient

	time_4 = get_time();

	// Part V: compute --- n_iter * (3 * height * width + 41 * (height-1) * (width-1) + 6) floating point arithmetic operations in totaL
	for (int iter = 0; iter < n_iter; iter++) {


		//update_image_ghostcells(image, height+2, width+2);

		// REDUCTION AND STATISTICS
		// --- 3 floating point arithmetic operations per element -> 3*height*width in total
		/*for (int i = 1; i <= height; i++) {
			for (int j = 1; j <= width; j++) {
			        tmp = image[i * (width+2) + j];	// current pixel value
				sum += tmp; // --- 1 floating point arithmetic operations
				sum2 += tmp * tmp; // --- 2 floating point arithmetic operations
			}
		}*/

		double loc_sum = 0;
		double loc_sum2 = 0;
		for(int i = width+2; i < loc_sizes[rank]-width-2; ++i) {
			loc_sum += loc_image[i];
			loc_sum2 += loc_image[i]*loc_image[i];
		}
		MPI_Allreduce(&loc_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&loc_sum2, &sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


		printf("Actual sum %f\n", sum);
		printf("Actual sum2 %f\n", sum2);
		mean = sum / (width*height); // --- 1 floating point arithmetic operations
		variance = (sum2 / (width*height)) - mean * mean; // --- 3 floating point arithmetic operations
		std_dev = variance / (mean * mean); // --- 2 floating point arithmetic operations
		printf("mean: %f, variance: %f, std_dev: %f\n", mean, variance, std_dev);

		//COMPUTE 1
		// --- 32 floating point arithmetic operations per element -> 32*(height-1)*(width-1) in total
		for (int i = 1; i <= loc_height-2 ; i++) {
			for (int j = 1; j <= width ; j++) {
			  k2 = (i-1) * width + (j-1);	// position of current element
			  k = i * (width+2) + j;

			  north_deriv[k2] = loc_image[(i - 1) * (width+2) + j] - loc_image[k];	// north derivative --- 1 floating point arithmetic operations
			  south_deriv[k2] = loc_image[(i + 1) * (width+2) + j] - loc_image[k];	// south derivative --- 1 floating point arithmetic operations
			  west_deriv[k2] = loc_image[i * (width+2) + (j - 1)] - loc_image[k];	// west derivative --- 1 floating point arithmetic operations
			  east_deriv[k2] = loc_image[i * (width+2) + (j + 1)] - loc_image[k];	// east derivative --- 1 floating point arithmetic operations
			  gradient_square = (north_deriv[k2] * north_deriv[k2] + south_deriv[k2] * south_deriv[k2] + west_deriv[k2] * west_deriv[k2] + east_deriv[k2] * east_deriv[k2]) / (loc_image[k] * loc_image[k]); // 9 floating point arithmetic operations
				laplacian = (north_deriv[k2] + south_deriv[k2] + west_deriv[k2] + east_deriv[k2]) / loc_image[k]; // 4 floating point arithmetic operations
				num = (0.5 * gradient_square) - ((laplacian * laplacian)/16.0); // 4 floating point arithmetic operations
				den = 1 + (.25 * laplacian); // 2 floating point arithmetic operations
				std_dev2 = num / (den * den); // 2 floating point arithmetic operations

				den = (std_dev2 - std_dev) / (std_dev * (1 + std_dev)); // 4 floating point arithmetic operations

				diff_coef[k] = 1.0 / (1.0 + den); // 2 floating point arithmetic operations

				if (diff_coef[k] < 0) {
			    diff_coef[k] = 0;
			  } else if (diff_coef[k] > 1)	{
			    diff_coef[k] = 1;
			  }
				if(diff_coef[k] == 0)
					count1++;

				temp_diff_coef[k] = diff_coef[k];


			}
		}

		for (int h = 1; h < loc_height-1; h++) {
			diff_coef[h*(width+2) + width+1] = diff_coef[h*(width+2) + 1];
		}

		if (rank != 0) {
			MPI_Send(diff_coef+(width+2), width+2, MPI_FLOAT, rank-1, 0,
				MPI_COMM_WORLD);//, NULL);
		}

		if (rank != numprocs-1) {
			MPI_Recv(diff_coef + ((width+2)* (loc_height-1)),
				width+2, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, NULL
			);
		}

		if(rank == 0){
			MPI_Send(diff_coef+(width+2), width+2, MPI_FLOAT,  numprocs-1, 0,
				MPI_COMM_WORLD);
		}

		if (rank == numprocs-1){
			MPI_Recv(diff_coef + ((width+2)* (loc_height-1)),
				width+2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL
			);
		}
		//update_coeff_ghostcells(diff_coef, height+2, width+2);



		// COMPUTE 2
		// divergence and image update --- 10 floating point arithmetic operations per element -> 10*(height-1)*(width-1) in total
		for (int i = 1; i <= loc_height-2; i++) {
			for (int j = 1; j <= width; j++) {
			  k2 = (i-1) * width + (j-1);
			  k = i * (width+2) + j;	// get position of current element
				if(diff_coef[k] != temp_diff_coef[k])
					diff_coef[k] = temp_diff_coef[k];

				if(diff_coef[k] != temp_diff_coef[k])
				count2++;

			  diff_coef_north = diff_coef[k];	// north diffusion coefficient
			  diff_coef_south = diff_coef[(i + 1) * (width+2) + j];	// south diffusion coefficient
			  diff_coef_west = diff_coef[k];	// west diffusion coefficient
			  diff_coef_east = diff_coef[i * (width+2) + (j + 1)];	// east diffusion coefficient

			  divergence = diff_coef_north * north_deriv[k2] + diff_coef_south * south_deriv[k2] + diff_coef_west * west_deriv[k2] + diff_coef_east * east_deriv[k2]; // --- 7 floating point arithmetic operations
				//if(divergence == 0)

				loc_image[k] = loc_image[k] + 0.25 * lambda * divergence; // --- 3 floating point arithmetic operations
			}
		}
	}
	/*for (int i = 0; i < numprocs; i++) {
			displs[i] += width+2;

			loc_sizes[i] -= 2 * (width+2);
	}*/
	sum = 0;
	for(int i = width+2; i < loc_sizes[rank]-width-2; ++i) {
		sum += loc_image[i];
	}
	printf("c1 %f rank: %d\n", sum, rank);
	//printf("c2 %d rank: %d\n", sum, rank);


	MPI_Gatherv(loc_image, loc_sizes[rank],
		MPI_UNSIGNED_CHAR, image,
		loc_sizes, displs, MPI_UNSIGNED_CHAR,
		0, MPI_COMM_WORLD
	);
	/*if (rank == 0) {	//Copy computed images to master
		for (int i = 1; i < numprocs; ++i) {
			MPI_Recv(image)
		}
	} else {
		MPI_Send(loc_image+width+2, (loc_height-2)*(width+2), MPI_UNSIGNED_CHAR,
			0, 0, MPI_COMM_WORLD);
	}*/
	printf("Here\n");
	if (rank != 0) {
		MPI_Finalize();
		return 0;
	}
	time_5 = get_time();

	//Copy back the extendted image array
	for (int i = 1; i <= height ; i++) {
	  for (int j = 1; j <= width ; j++) {
	    k = i * (width+2) + j;	// position of current element
	    k2 = (i-1) * width + (j-1);
	    tmp_image[k2] = (unsigned char)image[k];
	  }
	}

	// Part VI: write image to file
	stbi_write_png(outputname, width, height, pixelWidth, tmp_image, 0);
	time_6 = get_time();

	// Part VII: get average of sum of pixels for testing and calculate GFLOPS
	// FOR VALIDATION - DO NOT PARALLELIZE
	double test = 0;
	for (int i = 1; i <= height; i++) {
	  for (int j = 1; j <= width; j++) {
	    test += image[i * (width+2) + j];
	  }
	}
	test /= n_pixels;


	float gflops = (float) (n_iter * 1E-9 * (3 * height * width + 41 * (height-1) * (width-1) + 6)) / (time_5 - time_4);
	time_7 = get_time();

	// Part VII: deallocate variables
	//stbi_image_free(tmp_image);
	//free(image);
	//free(north_deriv);
	//free(south_deriv);
	//free(west_deriv);
	//free(east_deriv);
	//free(diff_coef);
	time_8 = get_time();

	MPI_Finalize();
	// print
	printf("Time spent in different stages of the application:\n");
	printf("%9.6f s => Part I: allocate and initialize variables\n", (time_1 - time_0));
	printf("%9.6f s => Part II: parse command line arguments\n", (time_2 - time_1));
	printf("%9.6f s => Part III: read image\n", (time_3 - time_2));
	printf("%9.6f s => Part IV: allocate variables\n", (time_4 - time_3));
	printf("%9.6f s => Part V: compute\n", (time_5 - time_4));
	printf("%9.6f s => Part VI: write image to file\n", (time_6 - time_5));
	printf("%9.6f s => Part VII: get average of sum of pixels for testing and calculate GFLOPS\n", (time_7 - time_6));
	printf("%9.6f s => Part VIII: deallocate variables\n", (time_7 - time_6));
	printf("Total time: %9.6f s\n", (time_8 - time_0));
	printf("Average of sum of pixels: %9.6f\n", test);
	printf("GFLOPS: %f\n", gflops);
	return 0;
}

// Update the ghost cells of image at boundary
void update_image_ghostcells(unsigned char *image, int height, int width)
{

  for (int h = 1; h < height-1; h++) {
    image[h*width + 0] = image[h*width + width-2];
    image[h*width + width-1] = image[h*width + 1];
  }

  for (int w = 1; w < width-1; w++) {
    image[0*width + w] = image[(height-2)*width + w];
    image[(height-1)*width + w] = image[1*width + w];
  }
}

// Update the ghost cells of diff_coeff at boundary
void update_coeff_ghostcells(float *diff_coeff, int height, int width)
{

  for (int h = 1; h < height-1; h++) {
    diff_coeff[h*width + width-1] = diff_coeff[h*width + 1];
  }

  for (int w = 1; w < width-1; w++) {
    diff_coeff[width*(height-1) + w] = diff_coeff[1*width + w];
  }
}
