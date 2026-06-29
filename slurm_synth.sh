#!/bin/bash
#!
#! SLURM job script for POC_MLP synthetic (FACED EEG) training on Wilkes3 (A100)
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J poc_synth
#! Which project should be charged:
#SBATCH -A MLMI-omo26-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! How many CPUs per task? On Wilkes3 a single GPU's host share is 32 CPUs /
#! ~256 GB RAM — claim it so the ~5 GB data load never starves.
#SBATCH --cpus-per-task=32
#! Specify the number of GPUs per node (between 1 and 4):
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=02:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! ######################################################################################
#! Redirect data + run outputs to personal scratch (override the macOS defaults in
#! config.py / paths.py without editing them).

export SYNTH_DATA_PATH="/rds/user/omo26/hpc-work/eeg_data/faced_data.npy"
export RUNS_BASE="/home/omo26/rds/hpc-work/EEG_NonRev/hpc_runs"

#! ######################################################################################

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! Optionally modify the environment seen by the application:
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

PYTHON_EXEC="$HOME/poc_venv/bin/python"
application="$PYTHON_EXEC -u main_synth.py"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"

#! Match CPU threads to the allocation. Training is GPU-bound, but the post-run
#! diagnostics (visualize_synth) do heavy CPU linear algebra — single-threaded
#! at T=2000 that stalls for many minutes.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"
mkdir -p logs "$RUNS_BASE/synth_runs"
JOBID=$SLURM_JOB_ID
CMD="$application > logs/out.$JOBID"

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "SYNTH_DATA_PATH: $SYNTH_DATA_PATH"
echo "RUNS_BASE:       $RUNS_BASE"

if [ "$SLURM_JOB_NODELIST" ]; then
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
