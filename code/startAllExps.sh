#/bin/bash
# you will need screen for this.

echo Running all experiments...

# neural
screen -dmS h3partnet1  bash -c "./runExperiment.py household 3participant neural 1 30"
screen -dmS d3partnet1  bash -c "./runExperiment.py driving 3participant neural 1 30"

screen -dmS htasknet1  bash -c "./runExperiment.py household LOOtask neural 1 30"
screen -dmS dtasknet1  bash -c "./runExperiment.py driving LOOtask neural 1 30"

# Constant-mean GP
screen -dmS h3partgp0  bash -c "./runExperiment.py household 3participant gp 0 3"
screen -dmS d3partgp0  bash -c "./runExperiment.py driving 3participant gp 0 3"

screen -dmS htaskgp0  bash -c "./runExperiment.py household LOOtask gp 0 3"
screen -dmS dtaskgp0  bash -c "./runExperiment.py driving LOOtask gp 0 3"

# PMGP 
screen -dmS h3partgp1  bash -c "./runExperiment.py household 3participant gp 1 3"
screen -dmS d3partgp1  bash -c "./runExperiment.py driving 3participant gp 1 3"

screen -dmS htaskgp1  bash -c "./runExperiment.py household LOOtask gp 1 3"
screen -dmS dtaskgp1  bash -c "./runExperiment.py driving LOOtask gp 1 3"

# POGP 
screen -dmS h3partgp2  bash -c "./runExperiment.py household 3participant gp 2 3"
screen -dmS d3partgp2  bash -c "./runExperiment.py driving 3participant gp 2 3"

screen -dmS htaskgp2  bash -c "./runExperiment.py household LOOtask gp 2 3"
screen -dmS dtaskgp2  bash -c "./runExperiment.py driving LOOtask gp 2 3"

# LG
screen -dmS h3partlg0  bash -c "./runExperiment.py household 3participant lineargaussian 0 2"
screen -dmS d3partlg0  bash -c "./runExperiment.py driving 3participant lineargaussian 0 2"

screen -dmS htasklg0  bash -c "./runExperiment.py household LOOtask lineargaussian 0 2"
screen -dmS htasklg0  bash -c "./runExperiment.py driving LOOtask lineargaussian 0 2"


# constant
screen -dmS h3partct0  bash -c "./runExperiment.py household 3participant constant 0 2"
screen -dmS d3partct0  bash -c "./runExperiment.py driving 3participant constant 0 2"

screen -dmS htaskct0  bash -c "./runExperiment.py household LOOtask constant 0 2"
screen -dmS htaskct0  bash -c "./runExperiment.py driving LOOtask constant 0 2"

