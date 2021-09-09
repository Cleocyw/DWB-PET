## This repository contains the coding files for MSc project: DWB-PET image denoising with CNN. The files are in the branch `master`.

# Structure
The structure of the repository is as following. Firstly, there are two experiment corresponding to `XCAT` and `Patient`. In each folder, there is a `unet.py` file for a complete denoising process. Folder `Plot` contains the code to plot different figures displayed in my dissertation.
* XCAT
  * Plot
  * unet_xcat.py
* Patient
  * Plot
  * unet.py
 
# Requirement
Python 3.6 or later (earlier versions may work, but are untested)
PyTorch 1.8 or later (earlier versions may work, but are untested)
GPU tmem = 8GB (can be achieved from UCL cluster node: `beaker`)

# Instruction
An example of `.sh` script is shown as follows:
```
#$ -l tmem=8G
#$ -l h_rt=999:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -cwd
#$ -j y

source /share/apps/source_files/python/python-3.6.4.source
source /share/apps/source_files/cuda/cuda-10.0.source
python3 abc.py argument_vector
```
Next, the format of argument_vector and the use of different files will be explained.
* XCAT
  * `unet_xcat.py 'file_name' 'input_type'`, where `file_name` should be `xcat.mat` and `input_type` should be one of `mean`, `noise`, `blurry`.
  * `computational_time.py 'file_name' 'input_type'`, same as above. This will return the computational time of running the network for 800 iterations in unit      of seconds.
  * Plot
    *  `metric_plot100.py folder_name`, where `folder_name` should the path of the saved npy, e.g. `'/cat_mean_frame/'`. This gives plots of the metrics values         against iterations and time.
    * `TAC_plot.py file_name folder_name location`, where `file_name` should be `xcat.mat`, `folder_name` should the path of the saved npy and `location` can          be `tumor` or `liver`. 
    * `plot_metrics_compare.ipynb` shows comparison of metrics among different inputs.
* Patient
  * `u-net_patient.py file_name input_type`, where `file_name` can be, e.g., `p10.mat` and `input_type` should be one of `mean`, `noise`, `blurry`.
  * Plot
    * `transfer_npy_to_img.ipynb` shows how to load npy file and save its images.
    * `TAC_plot_patient.py file_name folder_name`, argv same as before.
    * `difference_denoise.ipynb` shows the noise removed.
