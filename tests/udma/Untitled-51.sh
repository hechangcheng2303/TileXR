#!/usr/bin/env bash
set -x

mpirun -f hostfile -n 64 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 64 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 64 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 64 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 64 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 64 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 64 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 64 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 64 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 64 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 64 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 64 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms


mpirun -f hostfile -n 128 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 128 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 128 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 128 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 128 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 128 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 128 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 128 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 128 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 128 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 128 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 128 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms

mpirun -f hostfile -n 256 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 256 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 256 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 256 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 256 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 256 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 256 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 256 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 256 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 256 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 256 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 256 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms

mpirun -f hostfile -n 512 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 512 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 512 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 512 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 512 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 512 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 512 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 512 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
mpirun -f hostfile -n 512 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 512 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 512 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 512 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms


mpirun -f hostfile -n 1024 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 1024 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 1024 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 1024 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aicpu_ts
mpirun -f hostfile -n 1024 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 1024 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 1024 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
mpirun -f hostfile -n 1024 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a ccu_ms
#mpirun -f hostfile -n 1024 ./bin/all_gather_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
#mpirun -f hostfile -n 1024 ./bin/alltoall_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
#mpirun -f hostfile -n 1024 ./bin/all_reduce_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv
#mpirun -f hostfile -n 1024 ./bin/reduce_scatter_test -p 8 -b 8 -e 2g -f 2 -d int8 -a aiv





