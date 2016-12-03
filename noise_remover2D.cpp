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

void update_image_ghostcells(unsigned char *image, int height, int width);
void update_coeff_ghostcells(float *coeff, int height, int width);

int main(int argc, char *argv[]) {
   int numprocs, rank, namelen, provided;
   char processor_name[MPI_MAX_PROCESSOR_NAME];

   MPI_Init(&argc, &argv);

   //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
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
      printf("Assertion failed: x*y != np\n");
      printf("x*y=%d\n", px*py);
      printf("np=%d\n", numprocs);
      return 0;
   }
   time_2 = get_time();

   unsigned char *image;
   unsigned char *loc_image;
   unsigned char *tmp_image;
   int loc_width;
   int loc_height;
   /*int rem_height;
   int count2 = 0;
   int count1 = 0;*/


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
   //printf("x: %d", px);
   //printf("y: %d", py);

   loc_width = width/px;
   loc_height = height/py;
   int pwidth = loc_width+2;
   int pheight = loc_height+2;
   //Distribute the image parts
   if (rank == 0) {
      unsigned char *local_buf = (unsigned char*)malloc(sizeof(unsigned char) * (loc_width+2) * (loc_height+2));
      for (int i = 0; i < py; ++i) {
         for (int j = 0; j < px; ++j) {
            if (i == 0 && j == 0) continue;
            int current_start_row = i*(height/py);
            int currant_start_col = j*(width/px);
            int index = 0;
            for (int r = current_start_row; r < current_start_row + pheight; r++) {
               //printf("r:%d\n", r);
               for (int c = currant_start_col; c < currant_start_col + pwidth; c++) {
                  local_buf[index++] = image[r*(width+2) + c];
                  //local_buf[(r-currant_start_row)*pwidth+(c-currant_start_col)] = image[r*(width+2) + c];
               }
            }
            //MPI_Finalize();
            //return 0;
            //printf("Sending to %d...\n", i*px+j);
            MPI_Send(local_buf, index, MPI_UNSIGNED_CHAR, i*px+j, 0, MPI_COMM_WORLD);
         }
      }
      free(local_buf);
      //printf("1...\n");
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
      //printf("%d receiving...\n", rank);
      MPI_Recv(loc_image, pheight*pwidth, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, NULL);
      //printf("%d received...\n", rank);
   }
   //COMPUTATION
   time_3 = get_time();
   // Part IV: allocate variables
	//loc_height = loc_sizes[rank] / (width+2);
   int my_y = rank / px;
   int my_x = rank % px;
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
      //printf("rank %d,x %d, y %d,  right %d, left %d\n", rank, my_x, my_y, right, left);
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
				tmp = loc_image[i];
				loc_sum += tmp;
				loc_sum2 += tmp * tmp;
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
				//if(divergence == 0) printf("sıfır\n");

				loc_image[k] = loc_image[k] + 0.25 * lambda * divergence; // --- 3 floating point arithmetic operations
			}
		}
   }
      time_5 = get_time();

   /*for (size_t i = 0; i < n_pixels_with_ghost; i++) {
      image[i] = 0;
   }*/
   if(rank == 0) {
      unsigned char *buf = (unsigned char*)malloc(sizeof(unsigned char) * (loc_height+2) * (loc_width+2));
      for (int i = 0; i < py; ++i) {
         for (int j = 0; j < px; ++j) {
            if (i+j == 0) continue;
            int current_start_row = i*(height/py);
            int currant_start_col = j*(width/px);
            MPI_Recv(buf, (loc_height+2)*(loc_width+2), MPI_UNSIGNED_CHAR, i*px+j, 13, MPI_COMM_WORLD, NULL);
            int index = 0;
            for (int r = current_start_row; r < current_start_row + pheight; r++) {
               for (int c = currant_start_col; c < currant_start_col + pwidth; c++) {
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

   /*printf("sum %f Rank: %d\n",sum, rank);*/
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
      MPI_Finalize();
      return 0;
   }

   MPI_Finalize();


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
   return 0;
}
