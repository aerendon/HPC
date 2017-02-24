#!/bin/bash

#SBATCH --job-name=hello.c
#SBATCH --output=hello.sout
#
#SBATCH --ntasks=3
#SBATCH --nodes=3
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun hello.out
