# How to run
1. create a folder in HPC: `mkdir /scratch/<net_id>/hart_inference>`
2. create venv: `conda env create -f environment.yml -n <venv_name>`
note: i got many issues while installing libs, let me know if you run into any errors.
3. activate the env: `conda activate <venv_name>`
4. run inference: `python hart/sample.py --prompt <your_prompt>`
5. result will be saved in a folder named 'samples/'
6. to view the image, go to your pc's terminal and run `scp <net_id>@greene.hpc.nyu.edu:<path_to_image> <destination>`
for instance, i would run the following to store the image in my cwd: `scp dps9998@greene.hpc.nyu.edu:/scratch/dps9998/hart_inference/samples/0.png .`
