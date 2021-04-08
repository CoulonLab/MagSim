The force maps in these folders were made as follow:
- `MagSim_pillar_Keizer-et-al.ipynb` was used with appropriate parameters (see info therein and in Keizer et al.) to produce file `forceMap.tif`
- File `forceMap.tif` was open with Fiji and channels 4,5,6 were removed (note: `forceMap.tif` has 6 channels, which are the X, Y and Z components of the force, simply repeated twice so that Fiji opens the tif properly)
- The values inside the pillar volume were set to 0
- The image was rescaled to make pixel size identical to the experimental data
- Two versions of the file were made, one as such, and one flipped vertically where the Fy channel (c=2) was multiplied by -1.

