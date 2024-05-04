# Parallel_Computing
using MPI or CUDA to implement or speed up specific program
to configure the environment of mpi we should execlusive the command:
sudo apt-get install mpich
sudo apt-get install libopenmpi-dev
sudo apt-get install zlib1g-dev
using command :
sudo find / -name mpi.h 
to find the location of mpi.h
using vim open the file "~/.bashrc"
add "export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/lib/x86_64-linux-gnu/openmpi/include/" at the end of the document.
using command:
source ~/.bashrc