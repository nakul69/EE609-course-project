# Imitation Learning

This is the repository for the Imitation learning project and contains the implementation of Behavioral Cloning, DAGGER and SMILe algorithms.

## Files

PPOCartPole.py : Uses the official keras implementation to build the expert for the CartPole Environment.(Reference: (https://keras.io/examples/rl/ppo_cartpole))
PPOMountainCar.py : Build and saves the expert for the MountainCar Environment. Hyperparameters are slightly changed
DaggerCartPole.py : Dagger for CartPole and saves the result using pickle library
DaggerMountainCar.py : Dagger for MountainCar
smile.py : Smile for Cartpole environment
smile_mountaincar.py : Smile for Mountaincar.py
behavioral_cloningCartPole.py : behavior cloning implementation for CartPole
behavioral_cloningMountainCar.py : behavior cloning implementation for MountainCar
plot.py : used to plot the data by changing the correpong filename
test_PPOexpert.py : tests the trained PPO expert for different environments

##Execution

First run the PPO files to train the expert for different environments and save the trained models in corresponding files

Then execute the files correspinding to different algorithms for different environments to save the correponding results

Execute plot.py with the corresponding filenames to get the required plots
