# PAI-Utils

Programming Artificial Intelligence Utilities is a package that aims to make
artificial intelligence and machine learning programming easier through
abstractions of extensive APIs, research paper implementations, and data
manipulation.

Modules in this package are used extensively in [this playlist](https://www.youtube.com/watch?v=4_oJMTfTBYY&list=PLZkmLNZl0Wkw9KSJMBhbalbsxmrjdlgv3).
The reinforcement modules are used [here](https://github.com/Tiger767/OpenAIGymResults).

Package Features
- Analytics
  - Plotting of data through embedding algorithms, such as Isomap and TSNE
- Audio
  - Recording and playing
  - Volume, speed, and pitch manipulation
  - Trimming and Splitting
  - Spectrogram, Fbanks, and MFCC creation
  - Audio file conversions
- Image
  - Simplified OpenCV Interface
- Autoencoder
  - Trainer and Predictor
  - Trainer with extra decoder
  - Basic network architecture creation
- Evolution Algorithm
  - One dimensional evolution algorithm
  - Hyperparameter tuner
- VAE
  - Trainer
- GAN
  - Trainer and Predictor
  - GANI Trainer (GAN which takes provided Inputs)
- Neural Network
  - Trainer and Predictor
  - Dense layers that combine batch norm
  - Convolution layers that combine batch norm, max pooling, upsampling, and transposing
- Reinforcement
  - OpenAI Gym wrapper
  - Multi-agent adverserial environment
  - Greedy, ascetic, and stochastic policies
  - Noise Policies
  - Exponential, linear, and constant decay
  - Ring and normal memory
  - Agents
    - QAgent: Q-learning with a table
    - DQNAgent Q-learning with a neural network model
    - PGAgent: State to action neural network model (Actor) trained with
               policy gradients
    - DDPGAgent: State to continous action space neural network model trained
                 with deterministic policy gradients (Not working yet)
- Reinforcement Agents
  - DQNPGAgent: A combination of a DQN and PG agent into one agent
  - A2CAgent: Advantage Actor Critic agent
  - PPOAgent: Proximal Policy Optimization agent
  - TD3Agent: Twin Delayed DDPG Agent (Not working yet)
