# Help for simple phantom

![alt text](https://gitlab.gbar.dtu.dk/MREIT/IP2021/raw/master/Figures/simple_phantom.png "Simple_phantom")

In this directory, there are three folders in which codes related to the simple
phantom were placed. 

To run codes, you need to install FEniCS/2019.1.0 (For more information, see: 
https://fenicsproject.org/).

For each folder, there are four subfolders. Each subfolder is related to a  
subcase forthe simple phantom (for example Case_A1 contains codes for  
reconstruction conductivity from one measurement/one bonudary condition (A) and
three components of interior current density (1)).

Codes were writtern to run on DTU HPC system. First file "submint1.sh" should be
submit to HPC to generated the simulated data. After generating data, file 
"submit2.sh" can be submit to HPC in order to recostruct condcutivty 
distribution. If you want to run codes on a different HPC system, "submit1.sh" 
and "submit2.sh" should be modified accordingly. Codes can be run on PC, 
as well. However, please note the minimum computational requirment for each 
case, as indicated in the manuscript. 