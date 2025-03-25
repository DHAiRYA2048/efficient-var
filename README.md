# How to run
## Setting up
1. create a folder in HPC:
```
mkdir /scratch/<net_id>/hart_inference>
```
2. create venv:
```
conda env create -f environment.yml -n <venv_name>
```
>note: i got many issues while installing libs, let me know if you run into any errors.
3. activate the env:
```
conda activate <venv_name>
```

## Requesting resource
4. enter an interactive session, the following command will request 1 V100 node for 2 hours:
```
srun --mem=8G --account=pr_169_general --gres=gpu:v100:1 --cpus-per-task=8 --time=02:00:00 --pty bash -i
```
5. run inference:
```
python hart/sample.py --prompt <your_prompt>
```
6. result will be saved in a folder named `samples/`


7. to view the image, go to your pc's terminal and run
```
scp <net_id>@greene.hpc.nyu.edu:<path_to_image> <destination>
```
for instance, i would run the following to store the image in my cwd: `scp dps9998@greene.hpc.nyu.edu:/scratch/dps9998/hart_inference/samples/0.png .`

if you want to run a batch job instead, you need to create a file first. check 5k_batchjob.sh for an example of how to create the file. submit the batch job using `sbatch <file_name>`.
