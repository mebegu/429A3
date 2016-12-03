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
#include <omp.h>

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

int run_1D(int argc, char *argv[]);
int run_2D(int argc, char *argv[]);
int run_serial(int argc, char *argv[]);

int main(int argc, char *argv[]) {
   // Part II: parse command line arguments
	if(argc<2) {
	  printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>][-x <processor geometery in x>] [-y <processor geometery in y>]\n",argv[0]);
	  return(-1);
	}
   int px, py;
	for(int ac=1;ac<argc;ac++) {
		if(MATCH("-x")) {
		  px = atoi(argv[++ac]);
		} else if(MATCH("-y")) {
		  py = atoi(argv[++ac]);
		}
	}
   if (px * py == 1) {
      return run_serial(argc, argv);
   } else if (px == 1) {
      return run_1D(argc, argv);
   } else {
      return run_2D(argc, argv);
   }
   printf("You should never see this :(\n");
   return 1;
}
/*One dimentional code...*/
int run_1D(int argc, char *argv[])
{
   int numprocs, rank, namelen, provided;
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
	displs[0] = 0;

	for(int i=0; i< numprocs; i++){
		loc_sizes[i] = loc_size;
		if (i != numprocs-1)
			 displs[i+1] = (i+1) * loc_disp;// + disp * (width+2);
	  	else
	  		loc_sizes[i] += (width+2) * (height%numprocs);

	}


	loc_image = (unsigned char*) malloc(sizeof(unsigned char) * loc_sizes[rank]);
	MPI_Scatterv(image, loc_sizes, displs, MPI_UNSIGNED_CHAR, loc_image, loc_sizes[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);


	time_3 = get_time();


	// Part IV: allocate variables
	loc_height = loc_sizes[rank] / (width+2);
	int num_ghosts = 2 * width + 2 * (loc_height-2);
	north_deriv = (float*) malloc(sizeof(float) * (loc_sizes[rank] - num_ghosts));	// north derivative
	south_deriv = (float*) malloc(sizeof(float) * (loc_sizes[rank] - num_ghosts));	// south derivative
	west_deriv = (float*) malloc(sizeof(float) * (loc_sizes[rank] - num_ghosts));	// west derivative
	east_deriv = (float*) malloc(sizeof(float) * (loc_sizes[rank] - num_ghosts));		// east derivative
	diff_coef  = (float*) malloc(sizeof(float) * loc_sizes[rank]);	// diffusion coefficient

	time_4 = get_time();

	// Part V: compute --- n_iter * (3 * height * width + 41 * (height-1) * (width-1) + 6) floating point arithmetic operations in totaL
	for (int iter = 0; iter < n_iter+1; iter++) {

		if (rank%2 == 1) {
			MPI_Send(loc_image+width+2, width+2, MPI_UNSIGNED_CHAR, (rank-1+numprocs)%numprocs, 1,MPI_COMM_WORLD);
			MPI_Recv(loc_image+((loc_height-1)*(width+2)), width+2, MPI_UNSIGNED_CHAR, (rank+1)%numprocs, 1,MPI_COMM_WORLD, NULL);
		}
		else {
			MPI_Recv(loc_image+((loc_height-1)*(width+2)), width+2, MPI_UNSIGNED_CHAR, (rank+1)%numprocs, 1, MPI_COMM_WORLD, NULL);
			MPI_Send(loc_image+width+2, width+2, MPI_UNSIGNED_CHAR, (rank-1+numprocs)%numprocs, 1, MPI_COMM_WORLD);
		}


		if (rank%2 == 1) {
			MPI_Send(loc_image+((loc_height-2)*(width+2)), width+2, MPI_UNSIGNED_CHAR, (rank+1)%numprocs, 1,MPI_COMM_WORLD);
			MPI_Recv(loc_image, width+2, MPI_UNSIGNED_CHAR, (rank-1+numprocs)%numprocs, 1,MPI_COMM_WORLD, NULL);
		}
		else {
			MPI_Recv(loc_image, width+2, MPI_UNSIGNED_CHAR, (rank-1+numprocs)%numprocs, 1, MPI_COMM_WORLD, NULL);
			MPI_Send(loc_image+((loc_height-2)*(width+2)), width+2, MPI_UNSIGNED_CHAR, (rank+1)%numprocs, 1, MPI_COMM_WORLD);
		}

		#pragma omp parallel for
		for (int h = 1; h < loc_height-1; h++) {
			loc_image[h*(width+2)] = loc_image[h*(width+2) + width];
			loc_image[h*(width+2)+width+1] = loc_image[h*(width+2)+1];
		}

		if (iter == n_iter) break;
		//REDUCTION AND STATISTICS
		double loc_sum = 0;
		double loc_sum2 = 0;

		#pragma omp parallel for reduction(+:loc_sum, loc_sum2)
		for(int i = width+2; i < loc_sizes[rank]-width-2; ++i) {
			if (i%(width+2) != 0 && i%(width+2) != (width+1)) {
				double temp = loc_image[i];
				loc_sum += temp;
				loc_sum2 += temp * temp;
			}
		}

		MPI_Allreduce(&loc_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&loc_sum2, &sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


		mean = sum / (width*height); // --- 1 floating point arithmetic operations
		variance = (sum2 / (width*height)) - mean * mean; // --- 3 floating point arithmetic operations
		std_dev = variance / (mean * mean); // --- 2 floating point arithmetic operations
		//if(rank == 0) printf("iter: %d mean: %f, variance: %f, std_dev: %f\n", iter, mean, variance, std_dev);

		//COMPUTE 1
		// --- 32 floating point arithmetic operations per element -> 32*(height-1)*(width-1) in total
		#pragma omp parallel for private(k2, k, gradient_square, laplacian, num, den, std_dev2) collapse(2)
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
			}
		}

		#pragma omp parallel for
		for (int h = 1; h < loc_height-1; h++) {
			diff_coef[h*(width+2) + width+1] = diff_coef[h*(width+2)+1];
		}

		if (rank%2 == 1) {
			MPI_Send(diff_coef+width+2, width+2, MPI_FLOAT, (rank-1+numprocs)%numprocs, 1,MPI_COMM_WORLD);
			MPI_Recv(diff_coef+((loc_height-1)*(width+2)), width+2, MPI_FLOAT, (rank+1)%numprocs, 2,MPI_COMM_WORLD, NULL);
		}
		else {
			MPI_Recv(diff_coef+((loc_height-1)*(width+2)), width+2, MPI_FLOAT, (rank+1)%numprocs, 1, MPI_COMM_WORLD, NULL);
			MPI_Send(diff_coef+width+2, width+2, MPI_FLOAT, (rank-1+numprocs)%numprocs, 2, MPI_COMM_WORLD);
		}

		// COMPUTE 2
		// divergence and image update --- 10 floating point arithmetic operations per element -> 10*(height-1)*(width-1) in total
		#pragma omp parallel for private(diff_coef_north, diff_coef_south, diff_coef_west, diff_coef_east, divergence, k2,k)  collapse(2)
		for (int i = 1; i <= loc_height-2; i++) {
			for (int j = 1; j <= width; j++) {
			  k2 = (i-1) * width + (j-1);
			  k = i * (width+2) + j;	// get position of current element

			  diff_coef_north = diff_coef[k];	// north diffusion coefficient
			  diff_coef_south = diff_coef[(i + 1) * (width+2) + j];	// south diffusion coefficient
			  diff_coef_west = diff_coef[k];	// west diffusion coefficient
			  diff_coef_east = diff_coef[i * (width+2) + (j + 1)];	// east diffusion coefficient

			  divergence = diff_coef_north * north_deriv[k2] + diff_coef_south * south_deriv[k2] + diff_coef_west * west_deriv[k2] + diff_coef_east * east_deriv[k2]; // --- 7 floating point arithmetic operations

				loc_image[k] = loc_image[k] + 0.25 * lambda * divergence; // --- 3 floating point arithmetic operations
			}
		}

	}
	time_5 = get_time();
	for (size_t i = 0; i < numprocs; i++) {
		displs[i] += (width+2);
		loc_sizes[i] -= 2*(width+2);
	}
	MPI_Gatherv(loc_image, loc_sizes[rank],
		MPI_UNSIGNED_CHAR, image,
		loc_sizes, displs, MPI_UNSIGNED_CHAR,
		0, MPI_COMM_WORLD
	);

	if (rank != 0) {
		free(north_deriv);
		free(south_deriv);
		free(west_deriv);
		free(east_deriv);
		free(diff_coef);
		free(displs);
		free(loc_sizes);
		free(loc_image);
		//MPI_Finalize();
		//return 0;
	}

   else {
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
   	stbi_image_free(tmp_image);
   	free(image);
   	free(north_deriv);
   	free(south_deriv);
   	free(west_deriv);
   	free(east_deriv);
   	free(diff_coef);
   	free(displs);
   	free(loc_sizes);
   	free(loc_image);
   	time_8 = get_time();

   	// print
   	printf("Time spent in different stages of the application:\n");
   	printf("%9.6f s => Part I: allocate and initialize variables\n", (time_1 - time_0));
   	printf("%9.6f s => Part II: parse command line arguments\n", (time_2 - time_1));
   	printf("%9.6f s => Part III: read image\n", (time_3 - time_2));
   	printf("%9.6f s => Part IV: allocate variables\n", (time_4 - time_3));
   	printf("%9.6f s => Part V: compute\n", (time_5 - time_4));
   	printf("%9.6f s => Part VI: write image to file\n", (time_6 - time_5));
   	printf("%9.6f s => Part VII: get average of sum of pixels for testing and calculate GFLOPS\n", (time_7 - time_6));
   	printf("%9.6f s => Part VIII: deallocate variables\n", (time_8 - time_7));
   	printf("Total time: %9.6f s\n", (time_8 - time_0));
   	printf("Average of sum of pixels: %9.6f\n", test);
   	printf("GFLOPS: %f\n", gflops);
   }
   MPI_Finalize();
	return 0;
}

/*1-Dimentional code...*/
int run_2D(int argc, char *argv[])
{
   int numprocs, rank, namelen, provided;
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
   if (px * py != numprocs) {
      if (rank == 0) {
         printf("Assertion failed: x*y != np\n");
         printf("x*y=%d\n", px*py);
         printf("np=%d\n", numprocs);
      }
      return 0;
   }
   time_2 = get_time();

   unsigned char *image;
   unsigned char *loc_image;
   unsigned char *tmp_image;
   int loc_width;
   int loc_height;

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

   int my_y = rank / px;
   int my_x = rank % px;
   loc_width = width/px;
   loc_height = height/py;
   if(my_x == px-1) loc_width += width % px;
   if(my_y == py-1) loc_height += height % py;
   int pwidth = loc_width+2;
   int pheight = loc_height+2;
   //Distribute the image parts
   if (rank == 0) {
      unsigned char *local_buf = (unsigned char*)malloc(sizeof(unsigned char) * (loc_width+2+width%px) * (loc_height+2+height%py));
      for (int i = 0; i < py; ++i) {
         for (int j = 0; j < px; ++j) {
            if (i == 0 && j == 0) continue;
            int temp_h = pheight;
            int temp_w = pwidth;
            if(j == px-1) temp_w += width % px;
            if(i == py-1) temp_h += height % py;
            int current_start_row = i*(height/py);
            int currant_start_col = j*(width/px);
            int index = 0;
            for (int r = current_start_row; r < current_start_row + temp_h; r++) {
               for (int c = currant_start_col; c < currant_start_col + temp_w; c++) {
                  local_buf[index++] = image[r*(width+2) + c];
               }
            }
            MPI_Send(local_buf, index, MPI_UNSIGNED_CHAR, i*px+j, 0, MPI_COMM_WORLD);
         }
      }
      free(local_buf);
   }
   loc_image = (unsigned char *)malloc(sizeof(unsigned char) * (loc_height+2) * (loc_width+2));
   if (rank == 0) {
      #pragma omp parallel for  collapse(2)
      for (int i = 0; i < pheight; ++i) {
         for (int j = 0; j < pwidth; ++j) {
            loc_image[i*pwidth + j] = image[i*(width+2) + j];
         }
      }
   } else {
      MPI_Recv(loc_image, pheight*pwidth, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, NULL);
   }
   //COMPUTATION
   time_3 = get_time();
   // Part IV: allocate variables

   int num_ghosts = 2 * (loc_width+2) + 2 * (loc_height);
   int loc_size = (loc_width+2)*(loc_height+2);
   north_deriv = (float*) malloc(sizeof(float) * (loc_size - num_ghosts));	// north derivative
   south_deriv = (float*) malloc(sizeof(float) * (loc_size - num_ghosts));	// south derivative
   west_deriv = (float*) malloc(sizeof(float) * (loc_size - num_ghosts));	// west derivative
   east_deriv = (float*) malloc(sizeof(float) * (loc_size - num_ghosts));		// east derivative
   diff_coef  = (float*) malloc(sizeof(float) * loc_size);	// diffusion coefficient
   unsigned char *img_send_buffer = (unsigned char*)malloc(sizeof(unsigned char) * (loc_height+2));
   unsigned char *img_recv_buffer = (unsigned char*)malloc(sizeof(unsigned char) * (loc_height+2));
   float *diff_send_buffer = (float*)malloc(sizeof(float) * (loc_height+2));
   float *diff_recv_buffer = (float*)malloc(sizeof(float) * (loc_height+2));

   time_4 = get_time();

   for (int iter = 0; iter < n_iter+1; iter++) {
      //TODO: Send the image ghosts

      //Send to up/recieve from below
      int upper = ((my_y+py-1)%py)*px + my_x;
      int lower = ((my_y+1)%py)*px + my_x;
      if (my_y%2 == 1) {
         MPI_Send(loc_image+loc_width+2, loc_width+2, MPI_UNSIGNED_CHAR, upper, 1, MPI_COMM_WORLD);
         MPI_Recv(loc_image+((loc_height+1)*(loc_width+2)), loc_width+2, MPI_UNSIGNED_CHAR, lower, 2,MPI_COMM_WORLD, NULL);
      }
      else {
         MPI_Recv(loc_image+((loc_height+1)*(loc_width+2)), loc_width+2, MPI_UNSIGNED_CHAR, lower, 1, MPI_COMM_WORLD, NULL);
         MPI_Send(loc_image+loc_width+2, loc_width+2, MPI_UNSIGNED_CHAR, upper, 2, MPI_COMM_WORLD);
      }

      //Send to below/receive from up
      if (my_y%2 == 1) {
         MPI_Send(loc_image+((loc_height)*(loc_width+2)), loc_width+2, MPI_UNSIGNED_CHAR, lower, 3, MPI_COMM_WORLD);
         MPI_Recv(loc_image, loc_width+2, MPI_UNSIGNED_CHAR, upper, 4,MPI_COMM_WORLD, NULL);
      }
      else {
         MPI_Recv(loc_image, loc_width+2, MPI_UNSIGNED_CHAR, upper, 3, MPI_COMM_WORLD, NULL);
         MPI_Send(loc_image+((loc_height)*(loc_width+2)), loc_width+2, MPI_UNSIGNED_CHAR, lower, 4, MPI_COMM_WORLD);
      }

      /*Send to right*/
      int right = (my_x+1)%px + my_y*px;
      int left = (my_x+px-1)%px + my_y*px;
      //Fill the image send buffer
      #pragma omp parallel for
      for (int j = 0; j < (loc_height+2); j++) {
         img_send_buffer[j] = loc_image[j*(loc_width+2) + loc_width];
      }
      //Send buffers
      if (my_x%2 == 1) {
         MPI_Send(img_send_buffer, loc_height+2, MPI_UNSIGNED_CHAR, right, 5, MPI_COMM_WORLD);
         MPI_Recv(img_recv_buffer, loc_height+2, MPI_UNSIGNED_CHAR, left, 6,MPI_COMM_WORLD, NULL);
      }
      else {
         MPI_Recv(img_recv_buffer, loc_height+2, MPI_UNSIGNED_CHAR, left, 5, MPI_COMM_WORLD, NULL);
         MPI_Send(img_send_buffer, loc_height+2, MPI_UNSIGNED_CHAR, right, 6, MPI_COMM_WORLD);
      }
      //Fill the ghost cells
      #pragma omp parallel for
      for (int j = 0; j < (loc_height+2); j++) {
         loc_image[j*(loc_width+2)] = img_recv_buffer[j];
      }

      /*Send to left*/
      #pragma omp parallel for
      for (int j = 0; j < (loc_height+2); j++) {
         img_send_buffer[j] = loc_image[j*(loc_width+2)+1];
      }
      //Send buffers
      if (my_x%2 == 1) {
         MPI_Send(img_send_buffer, loc_height+2, MPI_UNSIGNED_CHAR, left, 7, MPI_COMM_WORLD);
         MPI_Recv(img_recv_buffer, loc_height+2, MPI_UNSIGNED_CHAR, right, 8,MPI_COMM_WORLD, NULL);
      }
      else {
         MPI_Recv(img_recv_buffer, loc_height+2, MPI_UNSIGNED_CHAR, right, 7, MPI_COMM_WORLD, NULL);
         MPI_Send(img_send_buffer, loc_height+2, MPI_UNSIGNED_CHAR, left, 8, MPI_COMM_WORLD);
      }
      //Fill the ghost cells
      #pragma omp parallel for
      for (int j = 0; j < (loc_height+2); j++) {
         loc_image[j*(loc_width+2) + loc_width+1] = img_recv_buffer[j];
      }
      if (iter == n_iter) break;
      //REDUCTION AND STATISTICS
      double loc_sum = 0;
      double loc_sum2 = 0;
      #pragma omp parallel for reduction(+:loc_sum, loc_sum2)
      for(int i = loc_width+2; i < loc_size-(loc_width+2); ++i) {
         if (i%(loc_width+2) != 0 && i%(loc_width+2) != (loc_width+1)) {
            double temp = loc_image[i];
            loc_sum += temp;
            loc_sum2 += temp * temp;
         }
      }
      MPI_Allreduce(&loc_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&loc_sum2, &sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      mean = sum / (width*height); // --- 1 floating point arithmetic operations
      variance = (sum2 / (width*height)) - mean * mean; // --- 3 floating point arithmetic operations
      std_dev = variance / (mean * mean); // --- 2 floating point arithmetic operations
      //if(rank == 0) printf("iter: %d mean: %f, variance: %f, std_dev: %f\n", iter, mean, variance, std_dev);
      //COMPUTE 1
      #pragma omp parallel for private(k2, k, gradient_square, laplacian, num, den, std_dev2) collapse(2)
      for (int i = 1; i <= loc_height ; i++) {
         for (int j = 1; j <= loc_width; j++) {
           k2 = (i-1) * loc_width + (j-1);	// position of current element
           k = i * (loc_width+2) + j;

           north_deriv[k2] = loc_image[(i - 1) * (loc_width+2) + j] - loc_image[k];	// north derivative --- 1 floating point arithmetic operations
           south_deriv[k2] = loc_image[(i + 1) * (loc_width+2) + j] - loc_image[k];	// south derivative --- 1 floating point arithmetic operations
           west_deriv[k2] = loc_image[i * (loc_width+2) + (j - 1)] - loc_image[k];	// west derivative --- 1 floating point arithmetic operations
           east_deriv[k2] = loc_image[i * (loc_width+2) + (j + 1)] - loc_image[k];	// east derivative --- 1 floating point arithmetic operations
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

         }
      }


      if (my_y%2 == 1) {
         MPI_Send(diff_coef+loc_width+2, loc_width+2, MPI_FLOAT, upper, 9,MPI_COMM_WORLD);
         MPI_Recv(diff_coef+((loc_height+1)*(loc_width+2)), loc_width+2, MPI_FLOAT, lower, 10,MPI_COMM_WORLD, NULL);
      }
      else {
         MPI_Recv(diff_coef+((loc_height+1)*(loc_width+2)), loc_width+2, MPI_FLOAT, lower, 9, MPI_COMM_WORLD, NULL);
         MPI_Send(diff_coef+loc_width+2, loc_width+2, MPI_FLOAT, upper, 10, MPI_COMM_WORLD);
      }

      /*Send to left*/
      #pragma omp parallel for
      for (int j = 0; j <= (loc_height+1); j++) {
         diff_send_buffer[j] = diff_coef[j*(loc_width+2)+1];
      }
      //Send buffers

      if (my_x%2 == 1) {
         MPI_Send(diff_send_buffer, loc_height+2, MPI_FLOAT, left, 11, MPI_COMM_WORLD);
         MPI_Recv(diff_recv_buffer, loc_height+2, MPI_FLOAT, right, 12,MPI_COMM_WORLD, NULL);
      }
      else {
         MPI_Recv(diff_recv_buffer, loc_height+2, MPI_FLOAT, right, 11, MPI_COMM_WORLD, NULL);
         MPI_Send(diff_send_buffer, loc_height+2, MPI_FLOAT, left, 12, MPI_COMM_WORLD);
      }
      //Fill the ghost cells
      #pragma omp parallel for
      for (int j = 1; j <= loc_height+1; j++) {
         diff_coef[j*(loc_width+2) + loc_width + 1] = diff_recv_buffer[j];
      }

      // COMPUTE 2
      // divergence and image update --- 10 floating point arithmetic operations per element -> 10*(height-1)*(width-1) in total
      #pragma omp parallel for private(diff_coef_north, diff_coef_south, diff_coef_west, diff_coef_east, divergence, k2,k)  collapse(2)
      for (int i = 1; i <= loc_height; i++) {
         for (int j = 1; j <= loc_width; j++) {
           k2 = (i-1) * loc_width + (j-1);
           k = i * (loc_width+2) + j;	// get position of current element

           diff_coef_north = diff_coef[k];	// north diffusion coefficient
           diff_coef_south = diff_coef[(i + 1) * (loc_width+2) + j];	// south diffusion coefficient
           diff_coef_west = diff_coef[k];	// west diffusion coefficient
           diff_coef_east = diff_coef[i * (loc_width+2) + (j + 1)];	// east diffusion coefficient

           divergence = diff_coef_north * north_deriv[k2] + diff_coef_south * south_deriv[k2] + diff_coef_west * west_deriv[k2] + diff_coef_east * east_deriv[k2]; // --- 7 floating point arithmetic operations

            loc_image[k] = loc_image[k] + 0.25 * lambda * divergence; // --- 3 floating point arithmetic operations
         }
      }
   }
      time_5 = get_time();

   if(rank == 0) {
      unsigned char *buf = (unsigned char*)malloc(sizeof(unsigned char) * (loc_width+2+width%px) * (loc_height+2+height%py));
      for (int i = 0; i < py; ++i) {
         for (int j = 0; j < px; ++j) {
            if (i+j == 0) continue;
            int temp_h = pheight;
            int temp_w = pwidth;
            if(j == px-1) temp_w += width % px;
            if(i == py-1) temp_h += height % py;
            int current_start_row = i*(height/py);
            int currant_start_col = j*(width/px);
            MPI_Recv(buf, temp_h*temp_w, MPI_UNSIGNED_CHAR, i*px+j, 13, MPI_COMM_WORLD, NULL);
            int index = 0;
            for (int r = current_start_row; r < current_start_row + temp_h; r++) {
               for (int c = currant_start_col; c < currant_start_col + temp_w; c++) {
                  image[r*(width+2) + c] = buf[index++];
               }
            }
         }
      }
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < pheight; ++i) {
         for (int j = 0; j < pwidth; ++j) {
             image[i*(width+2) + j] = loc_image[i*pwidth + j];
         }
      }
      free(buf);
   } else {
      MPI_Send(loc_image, loc_size, MPI_UNSIGNED_CHAR, 0, 13, MPI_COMM_WORLD);
   }

   if(rank != 0){
      free(north_deriv);
      free(south_deriv);
      free(west_deriv);
      free(east_deriv);
      free(diff_coef);
      free(diff_recv_buffer);
      free(diff_send_buffer);
      free(img_recv_buffer);
      free(img_send_buffer);
      free(loc_image);
      //MPI_Finalize();
      //return 0;
   }

   if (rank == 0) {
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
      stbi_image_free(tmp_image);
      free(image);
      free(north_deriv);
      free(south_deriv);
      free(west_deriv);
      free(east_deriv);
      free(diff_coef);
      free(diff_recv_buffer);
      free(diff_send_buffer);
      free(img_recv_buffer);
      free(img_send_buffer);
      free(loc_image);
      time_8 = get_time();

      // print
      printf("Time spent in different stages of the application:\n");
      printf("%9.6f s => Part I: allocate and initialize variables\n", (time_1 - time_0));
      printf("%9.6f s => Part II: parse command line arguments\n", (time_2 - time_1));
      printf("%9.6f s => Part III: read image\n", (time_3 - time_2));
      printf("%9.6f s => Part IV: allocate variables\n", (time_4 - time_3));
      printf("%9.6f s => Part V: compute\n", (time_5 - time_4));
      printf("%9.6f s => Part VI: write image to file\n", (time_6 - time_5));
      printf("%9.6f s => Part VII: get average of sum of pixels for testing and calculate GFLOPS\n", (time_7 - time_6));
      printf("%9.6f s => Part VIII: deallocate variables\n", (time_8 - time_7));
      printf("Total time: %9.6f s\n", (time_8 - time_0));
      printf("Average of sum of pixels: %9.6f\n", test);
      printf("GFLOPS: %f\n", gflops);
   }
   MPI_Finalize();
   return 0;
}
/*Serial code...*/
int run_serial(int argc, char *argv[])
{
   void update_image_ghostcells(unsigned char *image, int height, int width);
   void update_coeff_ghostcells(float *coeff, int height, int width);
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

	// Part III: read image
	printf("Reading image...\n");
	unsigned char *tmp_image = stbi_load(filename, &width, &height, &pixelWidth, 0);
	if (!tmp_image) {
		fprintf(stderr, "Couldn't load image.\n");
		return (-1);
	}
	printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);
	n_pixels = height * width;
	n_pixels_with_ghost = (height+2) * (width+2);
	// Copy the image into extended image array(with ghost cells)
	unsigned char *image = (unsigned char*) malloc(sizeof(unsigned char) * n_pixels_with_ghost);
	for (int i = 1; i <= height ; i++) {
	  for (int j = 1; j <= width ; j++) {
	    k = i * (width+2) + j;	// position of current element
	    k2 = (i-1) * width + (j-1);

	    image[k] = tmp_image[k2];
	  }
	}

	time_3 = get_time();


	// Part IV: allocate variables
	north_deriv = (float*) malloc(sizeof(float) * n_pixels);	// north derivative
	south_deriv = (float*) malloc(sizeof(float) * n_pixels);	// south derivative
	west_deriv = (float*) malloc(sizeof(float) * n_pixels);	// west derivative
	east_deriv = (float*) malloc(sizeof(float) * n_pixels);		// east derivative
	diff_coef  = (float*) malloc(sizeof(float) * n_pixels_with_ghost);	// diffusion coefficient

	time_4 = get_time();

	// Part V: compute --- n_iter * (3 * height * width + 41 * (height-1) * (width-1) + 6) floating point arithmetic operations in totaL
	for (int iter = 0; iter < n_iter; iter++) {
		sum = 0;
		sum2 = 0;
		//printf("%d\n", iter);
		update_image_ghostcells(image, height+2, width+2);

		// REDUCTION AND STATISTICS
		// --- 3 floating point arithmetic operations per element -> 3*height*width in total
      #pragma omp parallel for reduction(+:sum, sum2)
		for (int i = 1; i <= height; i++) {
			for (int j = 1; j <= width; j++) {
			   double temp = image[i * (width+2) + j];	// current pixel value
				sum += temp; // --- 1 floating point arithmetic operations
				sum2 += temp * temp; // --- 2 floating point arithmetic operations
			}
		}
		mean = sum / n_pixels; // --- 1 floating point arithmetic operations
		variance = (sum2 / n_pixels) - mean * mean; // --- 3 floating point arithmetic operations
		std_dev = variance / (mean * mean); // --- 2 floating point arithmetic operations
		//printf("iter: %d mean: %f, variance: %f, std_dev: %f\n", iter, mean, variance, std_dev);
		//COMPUTE 1
		// --- 32 floating point arithmetic operations per element -> 32*(height-1)*(width-1) in total
      #pragma omp parallel for private(k2, k, gradient_square, laplacian, num, den, std_dev2) collapse(2)
		for (int i = 1; i <= height ; i++) {
			for (int j = 1; j <= width ; j++) {
			  k2 = (i-1) * width + (j-1);	// position of current element
			  k = i * (width+2) + j;

			  north_deriv[k2] = image[(i - 1) * (width+2) + j] - image[k];	// north derivative --- 1 floating point arithmetic operations
			  south_deriv[k2] = image[(i + 1) * (width+2) + j] - image[k];	// south derivative --- 1 floating point arithmetic operations
			  west_deriv[k2] = image[i * (width+2) + (j - 1)] - image[k];	// west derivative --- 1 floating point arithmetic operations
			  east_deriv[k2] = image[i * (width+2) + (j + 1)] - image[k];	// east derivative --- 1 floating point arithmetic operations

			  gradient_square = (north_deriv[k2] * north_deriv[k2] + south_deriv[k2] * south_deriv[k2] + west_deriv[k2] * west_deriv[k2] + east_deriv[k2] * east_deriv[k2]) / (image[k] * image[k]); // 9 floating point arithmetic operations
			  laplacian = (north_deriv[k2] + south_deriv[k2] + west_deriv[k2] + east_deriv[k2]) / image[k]; // 4 floating point arithmetic operations
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
			}
		}

		update_coeff_ghostcells(diff_coef, height+2, width+2);

		// COMPUTE 2
		// divergence and image update --- 10 floating point arithmetic operations per element -> 10*(height-1)*(width-1) in total
      #pragma omp parallel for private(diff_coef_north, diff_coef_south, diff_coef_west, diff_coef_east, divergence, k2,k)  collapse(2)
		for (int i = 1; i <= height; i++) {
			for (int j = 1; j <= width; j++) {
			  k2 = (i-1) * width + (j-1);
			  k = i * (width+2) + j;	// get position of current element

			  diff_coef_north = diff_coef[k];	// north diffusion coefficient
			  diff_coef_south = diff_coef[(i + 1) * (width+2) + j];	// south diffusion coefficient
			  diff_coef_west = diff_coef[k];	// west diffusion coefficient
			  diff_coef_east = diff_coef[i * (width+2) + (j + 1)];	// east diffusion coefficient

			  divergence = diff_coef_north * north_deriv[k2] + diff_coef_south * south_deriv[k2] + diff_coef_west * west_deriv[k2] + diff_coef_east * east_deriv[k2]; // --- 7 floating point arithmetic operations
			  image[k] = image[k] + 0.25 * lambda * divergence; // --- 3 floating point arithmetic operations
			}
		}
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
	stbi_image_free(tmp_image);
	free(image);
	free(north_deriv);
	free(south_deriv);
	free(west_deriv);
	free(east_deriv);
	free(diff_coef);
	time_8 = get_time();

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
   #pragma omp parallel for
  for (int h = 1; h < height-1; h++) {
    image[h*width + 0] = image[h*width + width-2];
    image[h*width + width-1] = image[h*width + 1];
  }
  #pragma omp parallel for
  for (int w = 1; w < width-1; w++) {
    image[0*width + w] = image[(height-2)*width + w];
    image[(height-1)*width + w] = image[1*width + w];
  }
}

// Update the ghost cells of diff_coeff at boundary
void update_coeff_ghostcells(float *diff_coeff, int height, int width)
{
   #pragma omp parallel for
  for (int h = 1; h < height-1; h++) {
    diff_coeff[h*width + width-1] = diff_coeff[h*width + 1];
  }
  #pragma omp parallel for
  for (int w = 1; w < width-1; w++) {
    diff_coeff[width*(height-1) + w] = diff_coeff[1*width + w];
  }
}
