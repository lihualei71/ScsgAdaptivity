#!/bin/bash

#SBATCH --job-name=expr
#SBATCH --output=../log/SCSG_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --array=1-126

ml python/3.6.1

LINE=$(sed -n ${SLURM_ARRAY_TASK_ID}p "SCSG_expr_params.txt")
data=$(echo $LINE | cut -d ' ' -f 1)
etaL=$(echo $LINE | cut -d ' ' -f 2)
fname=$(echo "../results/SCSG-data-${data}-etaL-${etaL}.p")
python3 ../python/expr.py -f "${fname}" -d "${data}" -c "${etaL}" --seed "20191120"
