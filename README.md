# Improving LiDAR Fidelity Using Pattern Transplanting

Official tensorflow implementation of "Improving LiDAR Fidelity Using Pattern Transplanting. Bojrab et al. T-ITS 2022."

This paper reviews the use of ground and aerial LiDAR collection for the purpose of HD Map creation for Autonomous Driving. We propose a novel data preparation to convert 4D LiDAR to two-channel, 2.5D imagery. This preparation provides the means to transfer degradation patterns between collections without temporal or spatial overlap. We call this Pattern Transplanting. The dimensionality reducing LiDAR preparation allows our modified UNet to achieve highly accurate results in both Depth and Reflectance channels.

## Source Code
main.py         -- The top-level python script to run for training models.\
command_line.py -- The command line arguments for the program. This allows different permutations to be changed at runtime.\
pipeline.py     -- The data pipeline ingest for the training/validation data
train.py        -- The training routine with learning plateaus.
unet.py         -- Unbalanced, multi-stream UNet architecture with scriptable options to change its size.

### Example Command Lines:

**Example 1:** Training the main full-convolution model from testing with Pattern Transplanting
```buildoutcfg
>>>python main.py --in-channels 0 1 --out-channels 0 1 --kernels 128 256 256 256 --kernel-size 4 4 4 5 --strides 2 2 2 1 --l2-regularization 1e-5 --batch 5 --patience 50 --max-epochs 500 --learning-rate 5e-5 --momentum .0 --model-dir ./tests/example1/ ./data/
```
*--in-channels and --out-channels ensures we are using and generating both channels. \
--kernels, --kernel-size and --strides defines the model topology used in the experiment.\
--patience and --max-epochs defines the plateau-reductions schedule and total number of epochs.\
--learning-rate and --momentum define the base learning rate without momentum enabled.\
--model-dir is where the save models and tensorboard results will be saved.\
./data is a positional argument where the training/validation data exists.*

**Example 2:** Training the main 2x separable convolution model with Random Masking
```buildoutcfg
>>>python main.py --in-channels 0 1 --out-channels 0 1 --depth-multiplier 2 --kernels 128 256 256 256 --kernel-size 4 4 4 5 --strides 2 2 2 1 --l2-regularization 1e-5 --batch 5 --patience 50 --max-epochs 500 --learning-rate 5e-5 --momentum .0 --model-dir --random-mask ./tests/example1/ ./data/
```
*--depth-multiplier turns on separable convolutions with the specified scale factor.\
--random-mask turns on the randomization of masking with the same number of pixels dropped as the scanner pattern.*

## The Data
The dataset with random selection achieves ~1.2M unique training example combinations. We provide the tfrecord files for our data, and pipeline.py provides code to read and prepare the training data. The records are too large for github, so they must be downloaded separately:

[Download Dataset Now - 1.4 GB](https://improving-lidar-fidelity.s3.us-east-2.amazonaws.com/using/pattern-transplanting.zip)

Unzip into ./ImprovingLiDARPatternTransplanting/data/*

A few image examples are provided in the directory for visualization purposes only.
