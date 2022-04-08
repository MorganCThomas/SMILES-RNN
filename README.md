# SMILES-RNN

This repo contains code for a SMILES-based recurrent neural network used for *de novo* molecule generation with several  reinforcement learning algorithms available for molecule optimization. This is a stripped-back, heavily modified version of [REINVENT 2.0](https://github.com/MolecularAI/Reinvent/tree/reinvent.v.2.0) with some added functionality. This was written to be used in conjunction with [MolScore](https://github.com/MorganCThomas/MolScore) - although any other scoring function can also be used.

## Installation
First setup a conda environment with the correct requirements.

```
conda env create -f environment.yml
```

Or to update a prexisting environment.

```
conda env update --name myenv --file environment.yml
```

The package can also be installed into the conda evironment.

```
python setup.py install  # if you plan to make changes use 'develop' instead of 'install'
```

## Usage
Arguments to any of the scripts can be printed by running 

```
python <script> --help
```

## Training a prior

To train a prior run the *train_prior.py* script. You may note below that several other grammars are also implemented including [DeepSMILES](https://chemrxiv.org/engage/chemrxiv/article-details/60c73ed6567dfe7e5fec388d) and [SELFIES](https://iopscience.iop.org/article/10.1088/2632-2153/aba947) which are generated by conversion from SMILES. When using randomization (which can be done at train time) the SMILES are first randomized and then each random SMILES is converted to the alternative grammar. You can optionally pass in validation of test SMILES where the log likelihood will be compared during training which can be monitored via tensorboard. *Currently choosing a specific GPU device does not work, it will run on the default GPU device (i.e., index 0).

```
Train an initial prior model using SMILES. 

optional arguments:
  -h, --help            show this help message and exit

Required arguments:
  -i TRAIN_SMILES, --train_smiles TRAIN_SMILES
                        Path to smiles file (default: None)
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Output directory to save model (default: None)
  -s SUFFIX, --suffix SUFFIX
                        Suffix to name files (default: None)

Optional arguments:
  --grammar {SMILES,deepSMILES,deepSMILES_r,deepSMILES_cr,deepSMILES_c,deepSMILES_cb,deepSMILES_b,SELFIES}
                        Choice of grammar to use, SMILES will be encoded and decoded via grammar (default: SMILES)
  --randomize           Training smiles will be randomized using default arguments (10 restricted) (default: False)
  --valid_smiles VALID_SMILES
                        Validation smiles (default: None)
  --test_smiles TEST_SMILES
                        Test smiles (default: None)
  --validate_frequency VALIDATE_FREQUENCY
                        (default: 500)
  --n_epochs N_EPOCHS   (default: 5)
  --batch_size BATCH_SIZE
                        (default: 128)
  -d DEVICE, --device DEVICE
                        cpu/gpu or device number (default: gpu)

Network parameters:
  --layer_size LAYER_SIZE
                        (default: 512)
  --num_layers NUM_LAYERS
                        (default: 3)
  --cell_type {lstm,gru}
                        (default: gru)
  --embedding_layer_size EMBEDDING_LAYER_SIZE
                        (default: 256)
  --dropout DROPOUT     (default: 0.0)
  --learning_rate LEARNING_RATE
                        (default: 0.001)
  --layer_normalization

```

## Sampling from a trained prior

You can sample a trained model by running the *sample_model.py* script.

```
Sample smiles from model

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to checkpoint (.ckpt) (default: None)
  -o OUTPUT, --output OUTPUT
                        Path to save file (e.g. Data/Prior_10k.smi) (default: None)
  -d DEVICE, --device DEVICE
                        (default: gpu)
  -n NUMBER, --number NUMBER
                        (default: 10000)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature to sample (1: multinomial, <1: Less random, >1: More random) (default: 1.0)
  --unique              Keep sampling until n unique canonical molecules have been sampled (default: False)

```

## Fine-tuning

You can also fine-tune a trained model with a smaller dataset of SMILES by running the *fine_tune.py* script. If the pre-trained model was trained with an alternative grammar, these SMILES will also be converted at train time i.e., you always input molecules as SMILES.

```
Fine-tune a pre-trained prior model based on a smaller dataset

optional arguments:
  -h, --help            show this help message and exit

Required arguments:
  -p PRIOR, --prior PRIOR
                        Path to prior file (default: None)
  -i TUNE_SMILES, --tune_smiles TUNE_SMILES
                        Path to fine-tuning smiles file (default: None)
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Output directory to save model (default: None)
  -s SUFFIX, --suffix SUFFIX
                        Suffix to name files (default: None)

Optional arguments:
  --randomize           Training smiles will be randomized using default arguments (10 restricted) (default: False)
  --valid_smiles VALID_SMILES
                        Validation smiles (default: None)
  --test_smiles TEST_SMILES
                        Test smiles (default: None)
  --n_epochs N_EPOCHS   (default: 10)
  --batch_size BATCH_SIZE
                        (default: 128)
  -d DEVICE, --device DEVICE
                        cpu/gpu or device number (default: gpu)
  -f FREEZE, --freeze FREEZE
                        Number of RNN layers to freeze (default: None)
```

## Reinforcement learning

Finally, reinforcement learning can be run with the *reinforcement_learning.py* script. Note that this is written to work with [MolScore]() to handle the objective task i.e., molecule scoring. However, one can also use the underlying *ReinforcementLearning* class found in the *model/RL.py* module where another scoring function can be provided. This class has several methods for different reinforcement learning algorithms including:
- Reinforce
- REINVENT
- BAR
- Hill-Climb
- Augmented Hill-Climb

I also experimented with PPO and A2C algorithms but failed to reproduce reported [results](https://openreview.net/forum?id=Bk0xiI1Dz) - which has also been noted [elsewhere](https://github.com/BenevolentAI/guacamol_baselines/issues/6). However, I have implemented an *RNNCritic*  should anybody wish to have a go.

There are generic arguments that can be viewed by running `python reinforcement_learning.py --help`

```
Optimize an RNN towards a reward via reinforment learning

optional arguments:
  -h, --help            show this help message and exit

Required arguments:
  -p PRIOR, --prior PRIOR
                        Path to prior checkpoint (.ckpt) (default: None)
  -m MOLSCORE_CONFIG, --molscore_config MOLSCORE_CONFIG
                        Path to molscore config (.json) (default: None)

Optional arguments:
  -a AGENT, --agent AGENT
                        Path to agent checkpoint (.ckpt) (default: None)
  -d DEVICE, --device DEVICE
                        (default: gpu)
  -f FREEZE, --freeze FREEZE
                        Number of RNN layers to freeze (default: None)
  --save_freq SAVE_FREQ
                        How often to save models (default: 100)
  --verbose             Whether to print loss (default: False)
  --smiles_prefix SMILES_PREFIX
                        Smiles prefix added after generation (i.e. for scoring
                        (default: None)

RL strategy:
  {RV,RV2,BAR,AHC,HC,HC-reg,RF,RF-reg}
                        Which reinforcement learning algorithm to use
```

And RL algorithm specific arguments that can be viewed by running e.g., `python reinforcement_learning.py AHC --help`

```
Augmented Hill-Climb

optional arguments:
  -h, --help            show this help message and exit
  --n_steps N_STEPS     (default: 500)
  --batch_size BATCH_SIZE
                        (default: 64)
  -s SIGMA, --sigma SIGMA
                        Scaling coefficient of score (default: 60)
  -k [0-1], --topk [0-1]
                        Fraction of top molecules to keep (default: 0.5)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Adam learning rate (default: 0.0005)
```