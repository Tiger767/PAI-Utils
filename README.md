# PAI-Utils

Programming Artificial Intelligence Utilities is a package that aims to make
artificial intelligence and machine learning programming easier through
abstractions of extensive APIs, research paper implementations, and data
manipulation.

Package Features
- Analytics
  - Plotting of data through embedding algorithms, such as Isomap and TSNE
- Audio
  - Recording and playing
  - Volume, speed, and pitch manipulation
  - Trimming and Splitting
  - Nonoverlapping Spectrogram creation
  - Audio file conversions
- Autoencoder
  - Trainner and Predictor
  - Basic network architecture creation
- Evolution Algorithm
  - One dimensional evolution algorithm
  - Hyperparameter tuner
- GAN Network
  - Trainner and Predictor
- Neural Network
  - Trainner and Predictor
  - Dense layers that combine batch norm
  - Convolution layers that combine batch norm, max pooling, upsampling,
    and transposing
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
                 with deterministic policy gradients
- Reinforcement Agents
  - DQNPNAgent: A combination of a DQN and PG agent
  - A2CAgent: Advantage Actor Critic agent
  - PPOAgent: Proximal Policy Optimization agent
  - TD3Agent: Twin Delayed DDPG Agent
