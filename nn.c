//gcc test.c handle_file.c -lm -Wall -Wextra -O2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<errno.h>
#include<string.h>
#include "handle_file.h"
#define N0 784
#define N1 100
#define N2 10
#define alpha_ratio 0.9
#define epoch 3
double alpha = 0.6;

void show_ch( const char * const ch ){
  int rows = 28, columns = 28;
  for( int row = 0; row < rows; row ++ ){
    for( int column = 0; column < columns; column ++ ){
      int pixel = ch[ row * columns + column ] & 0xff;
      if( pixel < 120 ) printf(" ");
      else printf("*");
    }
    printf("\n");
  }
  printf("\n");
}

double sig( double x ){
  //double th = 34.538776394910684;
  double th = 10;
  if( th < x ) x = th;
  if( x < - th ) x = - th;
  return 1.0 / ( 1.0 + exp( - x ) );
}

double dsig( const double x ){
  return x * ( 1.0 - x );
}

double uniform( void ){
  return (( double ) rand() + 1.0 ) / (( double ) RAND_MAX + 2.0 );
}

double norm( void ){
  double z = sqrt( -2.0 * log( uniform() ) ) * sin( 2.0 * M_PI * uniform() );
  return z;
}
void feed_forward( double w0_1[ N0 ][ N1 ], double w1_2[ N1 ][ N2 ],
		   char input[ N0 ], double y0[ N0 ],
		   double x1[ N1 ], double y1[ N1 ],
		   double x2[ N2 ], double y2[ N2 ] ){
  for( int i = 0; i < N0; i ++ ){
    double in = ( int )( input[ i ] & 0xff ) * 0.99 / 255.0 + 0.01;
    y0[ i ] = in;
    for( int j = 0; j < N1; j ++ ){
      x1[ j ] += w0_1[ i ][ j ] * in;
    }
  }
  for( int i = 0; i < N1; i ++ ){
    y1[ i ] = sig( x1[ i ] );
  }
  for( int i = 0; i < N1; i ++ ){
    double y1_i = y1[ i ];
    for( int j = 0; j < N2; j ++ ){
      x2[ j ] += w1_2[ i ][ j ] * y1_i;
    }
  }
  for( int i = 0; i < N2; i ++ ){
    y2[ i ] = sig( x2[ i ] );
  }
}

int argmax( double array[], int len ){
  double max = array[ 0 ];
  int max_i = 0;
  for( int i = 1; i < len; i ++ ){
    if( max < array[ i ] ){
      max = array[ i ];
      max_i = i;
    }
  }
  return max_i;
}

void back_propagation( double w0_1[ N0 ][ N1 ], double w1_2[ N1 ][ N2 ],
		       double y0[ N0 ],
		       double y1[ N1 ],
		       double y2[ N2 ],
		       int label ){
  double d2[ N2 ];
  for( int i = 0; i < N2; i ++ ){
    double t = 0.01;
    if( i == label ) t = 0.99;
    d2[ i ] = ( t - y2[ i ] ) * dsig( y2[ i ] );
  }
   
  for( int i = 0; i < N1; i ++ ){
    double y1_i = y1[ i ];
    for( int j = 0; j < N2; j ++ ){
      w1_2[ i ][ j ] += alpha * d2[ j ] * y1_i;
    }
  }
  double d1[ N1 ] = { 0 };
  for( int i = 0; i < N1; i ++ ){
    for( int j = 0; j < N2; j ++ ){
      d1[ i ] += d2[ j ] * w1_2[ i ][ j ] * dsig( y1[ i ] );
    }
  }
    
  for( int i = 0; i < N0; i ++ ){
    double y0_i = y0[ i ];
    for( int j = 0; j < N1; j ++ ){
      w0_1[ i ][ j ] += alpha * d1[ j ] * y0_i;
    }
  }
}

int test(){
  srand( 100 );
  FILE *fp_label = file_open( "train-labels-idx1-ubyte" );
  FILE *fp_image = file_open( "train-images-idx3-ubyte" );
  unsigned int n = read_int( fp_label );
  if( n != read_int( fp_image ) ||
      read_int( fp_image ) != 28 ||
      read_int( fp_image ) != 28
      )
    die_with_error();

  //n = 300;
  
  //initialize
  double w0_1[ N0 ][ N1 ];
  double w1_2[ N1 ][ N2 ];
  for( int i = 0; i < N0; i ++ ){
    for( int j = 0; j < N1; j ++ ){
      w0_1[ i ][ j ] = norm();
    }
  }
  for( int i = 0; i < N1; i ++ ){
    for( int j = 0; j < N2; j ++ ){
      w1_2[ i ][ j ] = norm();
    }
  }

  //learning
  int count = 0;
  for( int epoch_i = 0; epoch_i < epoch; epoch_i ++ ){
    printf("#Epoch %d / %d\t%lf\n", epoch_i, epoch, alpha );
    for( unsigned int iter = 0; iter < n; iter ++ ){
      int label = read_byte( fp_label );

      if( iter % 5000 == 0 ){
	printf("  %5d / %d\n", iter, n );
	if( iter == 0 ){
	  printf("\tPerformance: %.2lf\n", count * 100.0 / n  );
	  count = 0;
	} else
	  printf("\tPerformance: %.2lf\n", count * 100.0 / ( iter + 1 )  );	  
      }
      
      char input[ N0 ];
      if( !fread( input, sizeof( char ), N0, fp_image ) )
	die_with_error();

      //show_ch( input );

      //feeding forward
      double y0[ N0 ];
      double x1[ N1 ] = { 0 };
      double y1[ N1 ];

      double x2[ N2 ] = { 0 };
      double y2[ N2 ];

      feed_forward( w0_1, w1_2,
		    input, y0,
		    x1,  y1,
		    x2,  y2
		    );

      int arg = argmax( y2, N2 );
      if( arg == label )
	count ++;    

      back_propagation( w0_1, w1_2,
			y0,
			y1,
			y2,
			label
			);
          
    }
    alpha *= alpha_ratio;
    
    fseek( fp_label, 8, SEEK_SET );
    fseek( fp_image, 16, SEEK_SET );
  }
  printf("\n\tPerformance: %.2lf\n", count * 100.0 / n );
  printf("Ended learing.\nTesting ...\n");

  //test
  FILE *test_fp_label = file_open( "t10k-labels-idx1-ubyte" );
  FILE *test_fp_image = file_open( "t10k-images-idx3-ubyte" );
  //FILE *test_fp_label = file_open( "train-labels-idx1-ubyte" );
  //FILE *test_fp_image = file_open( "train-images-idx3-ubyte" );
  unsigned int test_n = read_int( test_fp_label );
  if( test_n != read_int( test_fp_image ) ||
      read_int( test_fp_image ) != 28 ||
      read_int( test_fp_image ) != 28
      )
    die_with_error();

  count = 0;
  //  test_n = 10;
  
  for( unsigned int iter = 0; iter < test_n; iter ++ ){
    int label = read_byte( test_fp_label );
    //printf("%d: label is [ %d ]\n", iter, label );

    char input[ N0 ];
    if( !fread( input, sizeof( char ), N0, test_fp_image ) )
      die_with_error();

    //feeding forward
    double y0[ N0 ];
    double x1[ N1 ] = { 0 };
    double y1[ N1 ];

    double x2[ N2 ] = { 0 };
    double y2[ N2 ];

    feed_forward( w0_1, w1_2,
		  input, y0,
		  x1,  y1,
		  x2,  y2
		  );
    int arg = argmax( y2, N2 );

    if( arg == label )
      count ++;    
  }

  printf("Performance: %.3lf\n", ( ( double ) count ) / test_n * 100 );
  
  fclose( fp_label );
  fclose( fp_image );
  fclose( test_fp_label );
  fclose( test_fp_image );

  
  FILE *fp_result;
  char *filename_result = "nn.js";
  if( !( fp_result = fopen( filename_result, "w") ) ){
    fprintf( stderr, "Could not open %s: %s\n", filename_result, strerror( errno ) );
    exit( 1 );
  }
  fprintf( fp_result, "var w0_1 = [\n" );
  for( int i = 0; i < N0; i ++ ){
    fprintf( fp_result, " [" );
    for( int j = 0; j < N1 - 1; j ++ ){
      fprintf( fp_result, "%lf,", w0_1[ i ][ j ] );
    } 
    fprintf( fp_result, "%lf", w0_1[ i ][ N1 - 1 ] );
    if( i < N0 - 1 ) fprintf( fp_result, "],\n" );
    else fprintf( fp_result, "]\n" );
  }
  fprintf( fp_result, "];\nvar w1_2 = [\n" );
  for( int i = 0; i < N1; i ++ ){
    fprintf( fp_result, " [" );
    for( int j = 0; j < N2 - 1; j ++ ){
      fprintf( fp_result, "%lf,", w1_2[ i ][ j ] );
    }
    fprintf( fp_result, "%lf", w1_2[ i ][ N2 - 1 ] );
    if( i < N1 - 1 ) fprintf( fp_result, "],\n" );
    else fprintf( fp_result, "]\n" );
  }
  fprintf( fp_result, "];\n" );
  fclose( fp_result );
  
  
  FILE *fp_result_c;
  char *filename_result_c = "nn_weight.h";
  if( !( fp_result_c = fopen( filename_result_c, "w") ) ){
    fprintf( stderr, "Could not open %s: %s\n", filename_result_c, strerror( errno ) );
    exit( 1 );
  }
  fprintf( fp_result_c, "double w0_1[ %d ][ %d ] = {\n", N0, N1 );
  for( int i = 0; i < N0; i ++ ){
    fprintf( fp_result_c, " { " );
    for( int j = 0; j < N1 - 1; j ++ ){
      fprintf( fp_result_c, "%lf, ", w0_1[ i ][ j ] );
    } 
    fprintf( fp_result_c, "%lf ", w0_1[ i ][ N1 - 1 ] );
    if( i < N0 - 1 ) fprintf( fp_result_c, "},\n" );
    else fprintf( fp_result_c, "}\n" );
  }
  fprintf( fp_result_c, "};\ndouble w1_2[ %d ][ %d ] = {\n", N1, N2 );
  for( int i = 0; i < N1; i ++ ){
    fprintf( fp_result_c, " { " );
    for( int j = 0; j < N2 - 1; j ++ ){
      fprintf( fp_result_c, "%lf, ", w1_2[ i ][ j ] );
    }
    fprintf( fp_result_c, "%lf ", w1_2[ i ][ N2 - 1 ] );
    if( i < N1 - 1 ) fprintf( fp_result_c, "},\n" );
    else fprintf( fp_result_c, "}\n" );
  }
  fprintf( fp_result_c, "};\n" );
  fclose( fp_result_c );
  
  return 0;
}


int main(){
  test();

  return 0;
}

  /*
  for( int i = 0; i < N1; i ++ ){
    for( int j = 0; j < N2; j ++ ){
      printf("%.4lf ", w1_2[ i ][ j ] );
    }
    printf("\n");
  }
  */
