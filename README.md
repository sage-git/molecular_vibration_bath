# molecular_vibration_bath

Source code for [Modeling Intermolecular and Intramolecular Modes of Liquid Water Using Multiple Heat Baths: Machine Learning Approach](https://pubs.acs.org/doi/10.1021/acs.jctc.9b01288)

# External resource

MD source code for POLY2VS water potential is not included here.
Please contact to Taisuke Hasegawa.

# Typical Execution

1. Run water MD simulation with GROMACS, using input files in `gromacs2mylog/gromacs_input`
2. Convert the trajectory.
```
gmx trjconv -pbc mol -o traj_nopbc.trr
gmx traj -f traj_nopbc.trr -ox -ov -fp -nojump
```
3. Convert the output file to the input file of the machine learning.
```
python spc_log_converter.py
# traj.pkl will be created
```
4. Execute training script.
```
python train.py
```

5. Plot spectral density functions of baths.
```
# after performing ML training for SPC/E data
python copy_data.py (directory which contains last_b_amp_*.log)
mv Jomega.log Jomega_spce.log
# after performing ML training for TIP4P data
python copy_data.py (directory which contains last_b_amp_*.log)
mv Jomega.log Jomega_tip4p.log
python plot_two_models.py
```

# Source code description

Now writing...