#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {
  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("%s\n", processor_name);
  printf("%d\n", world_size);
  printf("%d\n", world_rank);
  printf("\n");


  MPI_Finalize();
  return 0;
}
