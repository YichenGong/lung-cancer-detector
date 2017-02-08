#!/bin/bash -e

#PBS -V
#PBS -N jupyterNote_DSB
#PBS -l nodes=1:ppn=1:gpus=1,mem=16GB,walltime=12:00:00
#PBS -j oe

cd $SCRATCH/lung-cancer-detector

## do not need to modify after this line
module purge
module purge
module load opencv/intel/2.4.10
module load tensorflow/python2.7/20161207
module load keras/1.1.1
module load matplotlib/intel/1.5.3
module load pandas/intel/0.17.1
module load jupyter/1.0.0
module load cython/intel/0.25.2

module list

pip install --user pydicom
pip install --user scikit-image

jupyter_port=$(shuf -i 6000-7000 -n 1)
ssh_port=$(shuf -i 7000-8000 -n 1)

cat<<EOF 

The instructions to setup ssh tunneling for jupyter note book from your Mac computer to the compute node that you have jupyter notebook running on

Step 1 :

If yuou are working in NYU campus, please open an iTerm window, run command

ssh -L ${ssh_port}:$(hostname):22 $USER@mercer.es.its.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to mercer with command

ssh $USER@mercer

Please open an iTerm window, run command

ssh -L ${ssh_port}:$(hostname):22 $USER@mercer

Step 2:

Keep the window in step 1 open, open a new iTerm terminal, run

ssh -L ${jupyter_port}:localhost:${jupyter_port} $USER@localhost -p ${ssh_port}

Step 3:

Keep the iTerm windows in the previouse 2 steps open. Now open browser, input

http://localhost:${jupyter_port}

you should be able to connect to jupyter notebook running remotly on mercer compute node

EOF

jupyter notebook --no-browser --port ${jupyter_port}

