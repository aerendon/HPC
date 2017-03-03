#!/bin/bash

#SBATCH --job-name=mat_mult.c
#SBATCH --output=mat_mult.sout
#
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun mat_mult.out
