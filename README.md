# The Transfer of Human Trust in Robot Capabilities across Tasks #
This is the supplementary material for the paper *The Transfer of Human Trust in Robot Capabilities across Tasks*, Robotics Science and Systems (RSS) 2018. 

## Requirements ##
To run the code, you will need:
- python3 
- pytorch v0.3.0.post4 (Note: v0.4.0 appears to introduce some breaking changes)
- spacy (for the word vectors)
- numpy
- scipy
- sklearn

The testing machine used the Anaconda distribution. To run the experiments automatically using the `startAllExps.sh` script, you will need unix `screen`.

## Folders and Files ##
- `code` contains all the code used to run the experiments and analyze the results.   
- `data` contains the data collected during our experiments and the word embeddings. 
- `docs` contains supplementary information (e.g., experimental setup details) that didn't make it to the paper due to space constraints. 

## Quickstart Q&A ##

### How can I replicate the plots in the paper? ###
The easiest way is to run though the `ModeComparison.ipynb` jupyter notebook. This will make use of the our saved results (in the `code/results` directory). 

### How do I re-run Experiments E1 and E2 in the paper? ###
If you want to re-run everything, you can run the `startAllExps.sh` script file. This uses `screen` to start a whole batch of parallel runs. This will take a while, so go have some coffee/sleep. Results will be saved to the `code/results` directory. Saved models will be in `code/savedmodels`. If you want to re-run specific experiments (with certain algorithm/dataset combinations), take a look at the individual commands in the `startAllExps.sh` script and the `runExperiment.py` pythonfile. The `runExperiment.py` file is relatively easy to use. 
```
./runExperiment.py [household|driving] [experiment type] [alg] [alg options] [latent task dimensionality]
```
As an example, to run the constant-mean GP with a latent task space of 3 dimensions (test on held out participants), run 
```
./runExperiment.py household 3participant gp 0 3
```

### Where can I find the models described in the paper? ###
The models are in `trustmodels.py` in separate classes. The neural model is called `NeuralTrustNet` and the GP is called `GPTrustTransfer`. The different GP variants are controlled via the switches. To use the prior mean function, set the `usepriormean` parameter to `True`. To use pseudo-observations, set `usepriorpoints` to `True`. You can actually use both, but this didn't seem to give strictly better results in our initial trials. 



