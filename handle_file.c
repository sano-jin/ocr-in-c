#include<stdio.h>
#include<errno.h>
#include<string.h>
#include<stdlib.h>
#include "handle_file.h"
#define N 4

void die_with_error(){
  fprintf( stderr, "error\n");
  exit( 1 );
}

unsigned int read_int( FILE *fp ){
  unsigned int n = 0;
  char buf[ N ];
  if( !fread( buf, sizeof( char ), 4, fp ) )
    die_with_error();
  
  for( int i = 0; i < 4 ; i ++ ){
    //printf("%02x ", buf[ i ] & 0xff );
    n += ( buf[ i ] & 0xff ) << ( 8 * ( 3 - i ) );
  }
  //printf("\n n = [ %d ]\n\n", n );
  
  return n;
}

int read_byte( FILE *fp ){
  char buf[ N ];
  if( !fread( buf, sizeof( char ), 1, fp ) )
    die_with_error();
    
  return buf[ 0 ] & 0xff;
}

FILE *file_open( const char * const filename ){
  FILE *fp;
  int len = strlen( filename );
  char *filepath = ( char * )malloc( ( len + 9 ) * sizeof( char ) );
  if( filepath == NULL ){
    printf("malloc\n");
    die_with_error();
  }
  strcpy( filepath, "dataset/" );
  strcpy( filepath + 8, filename );
  
  if( !( fp = fopen( filepath, "rb") ) ){
    fprintf( stderr, "Could not open %s: %s\n", filepath, strerror( errno ) );
    exit( 1 );
  }
  int magic_n = read_int( fp );
  printf("Opening [ %s ]\tmagic number is [ %08x ]\n", filepath, magic_n );
  free( filepath );
  return fp;
}
