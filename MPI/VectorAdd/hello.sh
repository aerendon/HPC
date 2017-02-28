#!/bin/bash

#SBATCH --job-name=vec_add.c
#SBATCH --output=vec_add.sout
#
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun vec_add.out
