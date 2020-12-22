![](./logo.png)

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
  - Spectrogram, Fbanks, and MFCC creation
  - Audio file conversions
- Image
  - Simplified OpenCV Interface
- Autoencoder
  - Trainer and Predictor
  - Trainer with extra decoder
  - VAE Trainer
- Evolution Algorithm
  - One dimensional evolution algorithm
  - Hyperparameter tuner
- GAN
  - GAN Trainer
  - GANI Trainer (GAN which takes provided Inputs)
  - Cycle GAN Trainer
  - Predictors
- Neural Network
  - Trainer and Predictor
  - Dense layers that combine batch norm
  - Convolution layers that combine batch norm, max pooling, upsampling, and transposing
- Reinforcement
  - OpenAI Gym wrapper
  - Multi-agent adverserial environment
  - Greedy, ascetic, and stochastic policies
  - Noise policies
  - Exponential, linear, and constant decay
  - Normal memory and efficient time distributed memory (for stacked states)
  - Agents
    - QAgent: Q-learning with a table
    - DQNAgent Q-learning with a neural network model
    - PGAgent: State to action neural network model (Actor) trained with
               policy gradients
    - DDPGAgent: State to continous action space neural network model trained
                 with deterministic policy gradients
- Reinforcement Agents
  - DQNPGAgent: Combination of a DQN and PG agent into one agent
  - A2CAgent: Advantage Actor Critic agent
  - PPOAgent: Proximal Policy Optimization agent
  - TD3Agent: Twin Delayed DDPG Agent
  - PGCAgent: Continuous variant of PGAgent
  - A2CCAgent: Continuous variant of A2CAgent

    
# Module `paiutils`




    
## Sub-modules

* [paiutils.analytics](#paiutils.analytics)
* [paiutils.audio](#paiutils.audio)
* [paiutils.autoencoder](#paiutils.autoencoder)
* [paiutils.evolution_algorithm](#paiutils.evolution_algorithm)
* [paiutils.gan](#paiutils.gan)
* [paiutils.image](#paiutils.image)
* [paiutils.neural_network](#paiutils.neural_network)
* [paiutils.reinforcement](#paiutils.reinforcement)
* [paiutils.reinforcement_agents](#paiutils.reinforcement_agents)
* [paiutils.util_funcs](#paiutils.util_funcs)






    
# Module `paiutils.analytics`

Author: Travis Hammond
Version: 12_21_2020





    
## Classes


    
### Class `Analyzer` 




>     class Analyzer(
>         x_data,
>         y_data,
>         labels,
>         label_colors=None
>     )


Analyzer is a class used for manipulating and viewing
classification datasets for analytical purposes. It can
also be used for unclassified data by passing in the same
value for y_data and the same label for all x_data.

Initializes the Analyzer with the dataset.


##### Args
**```x_data```**
:   A numpy ndarray


**```y_data```**
:   A numpy ndarray, which is a onehot encoding or ndx
        that corresponds to the label in labels


**```labels```**
:   A list of strings, which are labels for the y_data


**```label_colors```**
:   A list of list that contain 3 integers, which
              represent a color of a label for plotting









    
#### Methods


    
##### Method `boxplot` 




>     def boxplot(
>         self,
>         x,
>         figsize=(8, 8),
>         ndx_groups=None
>     )


Creates a boxplot for each group of x.


##### Args
**```x```**
:   A numpy ndarray of  1D positonal points for x_data


**```figsize```**
:   A tuple of 2 integers/floats, which are
         width and height, respectively


**```ndx_groups```**
:   A dictionary returned from create_label_ndx_groups



##### Returns
unmodified x

    
##### Method `calculate_distribution_of_labels` 




>     def calculate_distribution_of_labels(
>         self
>     )


Calculates the number of samples in each label.


##### Returns
A dictionary with strings (labels) as
keys and integers (number of samples) as values

    
##### Method `create_label_ndx_groups` 




>     def create_label_ndx_groups(
>         self
>     )


Creates a dictionary with ndx of each group in each label.


##### Returns
A dictionary with strings (labels) as keys
and list of integers (indexes of x_data) as values

    
##### Method `expand_data` 




>     def expand_data(
>         self,
>         size_per_label,
>         ndx_groups=None
>     )


Creates an Analyzer with a dataset that has been expanded
   by randomly choosing data from each group to get to the
   desired size of each group.


##### Args
**```size_per_label```**
:   A dictionary with labels as keys and sizes
                as values, or an integer, which is the size
                for all labels


**```ndx_groups```**
:   A dictionary returned from create_label_ndx_groups



##### Returns
An Analyzer

    
##### Method `isomap` 




>     def isomap(
>         self,
>         n_neighbors=5,
>         n_components=3,
>         eigen_solver='auto',
>         tol=0,
>         max_iter=None,
>         path_method='auto',
>         neighbors_algorithm='auto',
>         n_jobs=None
>     )


Creates an Isomap and fits x_data.


##### Args
**```n_neighbors```**
:   An integer, which is the number of neighbors
             considered for each point


**```n_components```**
:   An integer, which is the number of coordinates
              for the manifold


**```eigen_solver```**
:   A string ('auto', 'arpack', 'dense'),
              which is solver for the problem


**```tol```**
:   A float, which is the convergence tolerance for
     eigen solvers (arpack, lobpcg)


**```max_iter```**
:   An integer, which is the max number of iteration
          for the arpack solver


**```path_method```**
:   A string ('auto', 'FW', 'D'), which is the
             algorthim used to find the shortest path


**```neighbors_algorithm```**
:   A string ('auto', 'brute',
                     'kd_tree', 'ball_tree'), which is the
                     algorithm for nearest neighbors search


**```n_jobs```**
:   An integer (-1 all), which is the number of parallel
        jobs to run



##### Returns
A numpy ndarray, which has a shape like
(length of x_data, n_components)

    
##### Method `locally_linear_embedding` 




>     def locally_linear_embedding(
>         self,
>         n_neighbors=5,
>         n_components=3,
>         reg=0.001,
>         eigen_solver='auto',
>         tol=1e-06,
>         max_iter=100,
>         method='standard',
>         hessian_tol=0.0001,
>         modified_tol=1e-12,
>         neighbors_algorithm='auto',
>         random_state=None,
>         n_jobs=None
>     )


Computes the locally linear embedding of x_data.


##### Args
**```n_neighbors```**
:   An integer, which is the number of neighbors
             considered for each point


**```n_components```**
:   An integer, which is the number of coordinates
              for the manifold


**```reg```**
:   A float, which is the regularization constant


**```eigen_solver```**
:   A string ('auto', 'arpack', 'dense'),
              which is solver for the problem


**```tol```**
:   A float, which is the convergence tolerance for
     eigen solvers (arpack)


**```max_iter```**
:   An integer, which is the max number of iteration
          for the arpack solver


**```method```**
:   A string ('standard', 'hessian', 'modified', 'ltsa'),
        which is the embedding algorithm


**```hessian_tol```**
:   A float, which is the tolerance for Hessian method


**```modified_tol```**
:   A float, which is the tolerance for LLE method


**```neighbors_algorithm```**
:   A string ('auto', 'brute',
                     'kd_tree', 'ball_tree'), which is the
                     algorithm for nearest neighbors search


**```random_state```**
:   An integer, which is a seed for random number
              generator


**```n_jobs```**
:   An integer (-1 all), which is the number of parallel
        jobs to run



##### Returns
A numpy ndarray, which has a shape like
(length of x_data, n_components)

    
##### Method `mds` 




>     def mds(
>         self,
>         n_components=3,
>         metric=True,
>         n_init=4,
>         max_iter=300,
>         verbose=0,
>         eps=0.001,
>         random_state=None,
>         dissimilarity='euclidean',
>         n_jobs=None
>     )


Creates a Multidimensional scaling and fits x_data.


##### Args
**```n_components```**
:   An integer, which is the number of dimensions
              in which to immerse the dissimilarities


**```metric```**
:   A boolean, which determines if metric MDS is performed


**```n_init```**
:   An integer, which is the number of times the SMACOF
        algortithm will be run with different initializations


**```max_iter```**
:   An integer, which is the max number of iterations
          of the SMACOF algorithm for a single run


**```verbose```**
:   An integer, which determines the level of verbositity


**```eps```**
:   A float, which is the relative tolerance with regard to stress
     (determines when convergence is reached)


**```random_state```**
:   An integer, which is a seed for random number
              generator


**```dissimilarity```**
:   A string ('euclidean', 'precomputed'), which
               determines the measure to use


**```n_jobs```**
:   An integer (-1 all), which is the number of parallel
        jobs to run



##### Returns
A numpy ndarray, which has a shape like
(length of x_data, n_components)

    
##### Method `plot` 




>     def plot(
>         self,
>         x,
>         figsize=(8, 8)
>     )


Plots x on a graph.


##### Args
**```x```**
:   A numpy ndarray of positonal points for x_data


**```figsize```**
:   A tuple of 2 integers/floats, which are
         width and height, respectively



##### Returns
unmodified x

    
##### Method `shrink_data` 




>     def shrink_data(
>         self,
>         size_per_label,
>         ndx_groups=None
>     )


Creates an Analyzer with a dataset that has been shrunk
   by randomly choosing data from each group to get to the
   desired size of each group.


##### Args
**```size_per_label```**
:   A dictionary with labels as keys and sizes
                as values, or an integer, which is the size
                for all labels


**```ndx_groups```**
:   A dictionary returned from create_label_ndx_groups



##### Returns
An Analyzer

    
##### Method `tsne` 




>     def tsne(
>         self,
>         n_components=3,
>         perplexity=30.0,
>         early_exaggeration=12.0,
>         learning_rate=200.0,
>         n_iter=1000,
>         n_iter_without_progress=300,
>         min_grad_norm=1e-07,
>         metric='euclidean',
>         init='random',
>         verbose=0,
>         random_state=None,
>         method='barnes_hut',
>         angle=0.5,
>         n_jobs=None
>     )


Creates a t-distributed Stochastic Neighbor Embedding and fits x_data.


##### Args
**```n_components```**
:   An integer, which is the dimension of
              embedded space


**```perplexity```**
:   A float, which is related to the number of
            nearest neigbors that are used in manifold learning


**```early_exaggeration```**
:   A float, which controls how tight natural
                    clusters are embedded


**```learning_rate```**
:   A float within 10.0-1000.0 (inclusive), which
               higher makes data more like a 'ball', and lower
               makes data cloudy with fewer outliers


**```n_iter```**
:   An integer, which is the max number of iterations
        for optimization


**```n_iter_without_progress```**
:   An integer, which is the max number
                         of iterations to continue without
                         progress


**```min_grad_norm```**
:   A float, which is the threshold for optimization
               to continue


**```metric```**
:   A string ('euclidean'), which determines the metric to use
        for calculating distance between features arrays


**```init```**
:   A string ('random', 'pca'), which determines how to
      initalize the embedding


**```verbose```**
:   An integer, which determines the level of verbositity


**```random_state```**
:   An integer, which is a seed for random number
              generator


**```method```**
:   A string ('barnes_hut', 'exact'), which is the gradient
        calculation algorithm


**```angle```**
:   A float for barnes_hut, which is the determines the trade
       off between speed and accuracy


**```n_jobs```**
:   An integer (-1 all), which is the number of parallel
        jobs to run



##### Returns
A numpy ndarray, which has a shape like
(length of x_data, n_components)



    
# Module `paiutils.audio` 

Author: Travis Hammond
Version: 12_21_2020




    
## Functions


    
### Function `adjust_speed` 




>     def adjust_speed(
>         audio,
>         rate,
>         multiplier=1
>     )


Adjusts the speed of the audio and keeps the RMS power the same.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```multiplier```**
:   A float, which is the amount to adjust the relative speed



##### Returns
A numpy ndarray, which has 1 dimension

    
### Function `adjust_volume` 




>     def adjust_volume(
>         audio,
>         multiplier=1
>     )


Adjusts the volume of the audio.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```multiplier```**
:   A float, which is the amount to adjust the relative volume



##### Returns
A numpy ndarray, which has 1 dimension

    
### Function `blend` 




>     def blend(
>         audio1,
>         audio2,
>         audio1_weight=0.5,
>         audio2_weight=None
>     )


Blends two audios together.


##### Args
**```audio1```**
:   A numpy ndarray, which has 1 dimension and values within
        -1.0 to 1.0 (inclusive)


**```audio2```**
:   A numpy ndarray, which has 1 dimension and values within
        -1.0 to 1.0 (inclusive)


**```audio1_weight```**
:   A float, which is the weight of audio 1
               and should be within 0.0 and 1.0 (exclusive)


**```audio2_weight```**
:   A float, which is the weight of audio 2
               and should be within 0.0 and 1.0 (exclusive)



##### Returns
A numpy ndarray, which has 1 dimension

    
### Function `calc_duration` 




>     def calc_duration(
>         audio,
>         rate
>     )


Calculates the length of the audio in seconds.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken



##### Returns
A float

    
### Function `calc_rms` 




>     def calc_rms(
>         audio
>     )


Calculates the Root Mean Square of the audio.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)



##### Returns
A float, which is the rms of the audio

    
### Function `change_rate` 




>     def change_rate(
>         audio,
>         rate,
>         new_rate,
>         atype=None
>     )


Changes the audio's sample rate.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```new_rate```**
:   An integer, which is the rate to change the audio to



##### Returns
A tuple of the loaded audio, rate, and atype

    
### Function `compute_fbank` 




>     def compute_fbank(
>         signal,
>         samplerate=16000,
>         winlen=0.025,
>         winstep=0.01,
>         nfilt=26,
>         nfft=512,
>         lowfreq=0,
>         highfreq=None,
>         preemph=0.97,
>         winfunc=<function <lambda>>
>     )


Compute Mel-filterbank energy features from an audio signal.
Code adapted from python_speech_features, written orginally by James Lyons.


##### Args
**```signal```**
:   the audio signal from which to compute features.
        Should be an N*1 array


**```samplerate```**
:   the sample rate of the signal we are working with, in Hz.


**```winlen```**
:   the length of the analysis window in seconds. Default is
        0.025s (25 milliseconds)


**```winstep```**
:   the step between successive windows in seconds. Default
         is 0.01s (10 milliseconds)


**```nfilt```**
:   the number of filters in the filterbank, default 26.


**```nfft```**
:   the FFT size. Default is None, which uses the calculate_nfft
      function to choose the smallest size that does not drop
      sample data.


**```lowfreq```**
:   lowest band edge of mel filters. In Hz, default is 0.


**```highfreq```**
:   highest band edge of mel filters. In Hz, default
          is samplerate/2


**```preemph```**
:   apply preemphasis filter with preemph as coefficient.
         0 is no filter. Default is 0.97.


**```winfunc```**
:   the analysis window to apply to each frame. By default
         no window is applied. You can use numpy window functions
         here e.g. winfunc=numpy.hamming



##### Returns
2 values. The first is a numpy array of size (NUMFRAMES by nfilt)
    containing features. Each row holds 1 feature vector. The
    second return value is the energy in each frame
    (total energy, unwindowed)

    
### Function `compute_mfcc` 




>     def compute_mfcc(
>         signal,
>         samplerate=16000,
>         winlen=0.025,
>         winstep=0.01,
>         numcep=13,
>         nfilt=26,
>         nfft=None,
>         lowfreq=0,
>         highfreq=None,
>         preemph=0.97,
>         ceplifter=22,
>         append_energy=True,
>         winfunc=<function <lambda>>
>     )


Computes MFCC features from an audio signal.
Code adapted from python_speech_features, written orginally by James Lyons.


##### Args
**```signal```**
:   the audio signal from which to compute features.
        Should be an N*1 array


**```samplerate```**
:   the sample rate of the signal we are working with, in Hz.


**```winlen```**
:   the length of the analysis window in seconds. Default is
        0.025s (25 milliseconds)


**```winstep```**
:   the step between successive windows in seconds. Default
         is 0.01s (10 milliseconds)


**```numcep```**
:   the number of cepstrum to return, default 13


**```nfilt```**
:   the number of filters in the filterbank, default 26.


**```nfft```**
:   the FFT size. Default is None, which uses the calculate_nfft
      function to choose the smallest size that does not drop
      sample data.


**```lowfreq```**
:   lowest band edge of mel filters. In Hz, default is 0.


**```highfreq```**
:   highest band edge of mel filters. In Hz, default is
          samplerate/2


**```preemph```**
:   apply preemphasis filter with preemph as coefficient.
         0 is no filter. Default is 0.97.


**```ceplifter```**
:   apply a lifter to final cepstral coefficients.
           0 is no lifter. Default is 22.


**```append_energy```**
:   if this is true, the zeroth cepstral coefficient is
               replaced with the log of the total frame energy.


**```winfunc```**
:   the analysis window to apply to each frame. By default
         no window is applied. You can use numpy window functions
         here e.g. winfunc=numpy.hamming



##### Returns
A numpy array of size (NUMFRAMES by numcep) containing features.
    Each row holds 1 feature vector.

    
### Function `compute_spectrogram` 




>     def compute_spectrogram(
>         audio,
>         rate,
>         frame_duration,
>         real=True
>     )


Computes a nonoverlapping spectrogram.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```frame_duration```**
:   A float, which is the duration of each frame


**```real```**
:   A boolean, which determines if one side hermitian ffts
      should be used or real ffts



##### Returns
A tuple of a numpy ndarray, which has 2 dimensions
    (frame, frequency powers), and an integer (new rate)

    
### Function `convert_amplitude_to_db` 




>     def convert_amplitude_to_db(
>         amplitude,
>         ref_func=<function <lambda>>,
>         min_threshold=1e-10,
>         db_threshold=80.0
>     )


Converts amplitude to decibels.


##### Args
**```amplitude```**
:   A numpy ndarray, which has 1 or 2 dimensions


**```min_threshold```**
:   A float, which is the minimum magnitude


**```db_threshold```**
:   A float, which is the threshold for the audio
        in decibels



##### Returns
A numpy ndarray, which has 1 or 2 dimensions

    
### Function `convert_atype_to_width` 




>     def convert_atype_to_width(
>         atype
>     )


Converts an audio type to the number of bytes each value takes.


##### Args
**```atype```**
:   A string, which is an audio type



##### Returns
An integer, which is the number of bytes wide

    
### Function `convert_audio_to_db` 




>     def convert_audio_to_db(
>         audio,
>         rate,
>         frame_duration,
>         ref_func=<function <lambda>>,
>         min_threshold=1e-10,
>         db_threshold=80.0
>     )


Converts the audio to decibels.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```frame_duration```**
:   A float, which is the duration of each frame


**```ref_func```**
:   A function, which takes a magnitude and returns a value


**```min_threshold```**
:   A float, which is the minimum magnitude


**```db_threshold```**
:   A float, which is the threshold for the audio
              in decibels



##### Returns
A tuple of a numpy ndarray, which has 1 dimension,
    and an integer (new rate)

    
### Function `convert_power_to_db` 




>     def convert_power_to_db(
>         power,
>         ref_func=<function <lambda>>,
>         min_threshold=1e-10,
>         db_threshold=80.0
>     )


Converts power to decibels.


##### Args
**```power```**
:   A numpy ndarray, which has 1 or 2 dimensions


**```min_threshold```**
:   A float, which is the minimum magnitude


**```db_threshold```**
:   A float, which is the threshold for the audio
        in decibels



##### Returns
A numpy ndarray, which has 1 or 2 dimensions

    
### Function `convert_spectrogram_to_audio` 




>     def convert_spectrogram_to_audio(
>         spectrogram,
>         rate,
>         real=True
>     )


Converts a nonoverlapping spectrogram back to audio.


##### Args
**```spectrogram```**
:   A numpy ndarray, which has 2 dimensions


**```rate```**
:   An integer, which is the rate at which each frame is taken


**```real```**
:   A boolean, which determines if one side hermitian ffts
      should be used or real ffts



##### Returns
A tuple of a numpy ndarray, which has 1 dimension,
    and an integer (new rate)

    
### Function `convert_width_to_atype` 




>     def convert_width_to_atype(
>         width
>     )


Converts a number of bytes to an audio type.


##### Args
**```width```**
:   An integer, which is the number of bytes wide



##### Returns
A string, which is the audio type

    
### Function `file_play` 




>     def file_play(
>         filename
>     )


Plays the audio file.


##### Args
**```filename```**
:   A string, which is the directory or filename of the
          file to load



    
### Function `file_record` 




>     def file_record(
>         filename,
>         seconds,
>         rate,
>         atype=None,
>         recording_device_name='Microphone'
>     )


Records audio from the recording device to a file.


##### Args
**```filename```**
:   A string, which is the directory or filename of the
          file to load


**```seconds```**
:   A float, which is the length of the recording


**```rate```**
:   An integer, which is the rate at which samples are taken


**```atype```**
:   A string, which is the audio type (default: int16)


**```recording_device_name```**
:   A string, which is the name of the
                       recording device



    
### Function `find_gaps` 




>     def find_gaps(
>         audio,
>         rate,
>         frame_duration,
>         ambient_power=0.0001
>     )


Finds the length of gaps in the audio.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```frame_duration```**
:   A float, which is the duration of each frame
                to check


**```ambient_power```**
:   A float, which is the Root Mean Square of ambient noise



##### Returns
A list of tuples with the first value in the tuple being the start
    of a gap and the second value the end

    
### Function `for_each_frame` 




>     def for_each_frame(
>         audio,
>         rate,
>         frame_duration,
>         func
>     )


Calls a function on each frame.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```frame_duration```**
:   A float, which is the duration of each frame


**```func```**
:   A function, which takes a frame and returns a value



##### Returns
A tuple of a numpy ndarray of results from func and integer
    (new rate)

    
### Function `load` 




>     def load(
>         filename,
>         rate=None,
>         assert_mono=True
>     )


Changes the audio's sample rate.


##### Args
**```filename```**
:   A string, which is the directory or filename of the
          file to load


**```rate```**
:   An integer, which is the rate at which samples are taken


**```assert_mono```**
:   A boolean, which determines if an assertion error
             should be raise if there are more than one channel
             in the audio or if it should be converted to one
             channel



##### Returns
A tuple of the loaded audio, rate, and atype

    
### Function `play` 




>     def play(
>         audio,
>         rate,
>         atype=None
>     )


Plays the audio.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```atype```**
:   A string, which is the audio type (default: int16)



    
### Function `plot` 




>     def plot(
>         audio,
>         seconds=0
>     )


Plots the audio on a graph.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```seconds```**
:   A float, which is the number of seconds to show the plot



    
### Function `record` 




>     def record(
>         seconds,
>         rate,
>         atype=None,
>         recording_device_name='Microphone'
>     )


Records audio from the recording device.


##### Args
**```seconds```**
:   A float, which is the length of the recording


**```rate```**
:   An integer, which is the rate at which samples are taken


**```atype```**
:   A string, which is the audio type (default: int16)


**```recording_device_name```**
:   A string, which is the name of the
                       recording device



##### Returns
A tuple of the loaded audio, rate, and atype

    
### Function `save` 




>     def save(
>         filename,
>         audio,
>         rate,
>         atype=None
>     )


Saves the audio to a file.


##### Args
**```filename```**
:   A string, which is the directory or filename of the
          file to load


**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```atype```**
:   A string, which is the audio type (default: int16)



    
### Function `set_duration` 




>     def set_duration(
>         audio,
>         rate,
>         seconds,
>         mode='R',
>         pad_value=0
>     )


Sets the duration of audio in seconds.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```seconds```**
:   A float, which is the duration to set the audio to


**```mode```**
:   A string ('L','R','B'), which determines where to pad or remove


**```pad_values```**
:   A float within -1.0 to 1.0 (inclusive), which will be
            the value if the audio is padded



##### Returns
A numpy ndarray, which has 1 dimension and values within
    -1.0 to 1.0 (inclusive)

    
### Function `set_length` 




>     def set_length(
>         audio,
>         length,
>         mode='R',
>         pad_value=0
>     )


Sets the length of audio.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```length```**
:   An integer, which is the length to set the audio to


**```mode```**
:   A string ('L','R','B'), which determines where to pad or remove


**```pad_values```**
:   A float within -1.0 to 1.0 (inclusive), which will be
            the if the audio is padded



##### Returns
A numpy ndarray, which has 1 dimension and values within
    -1.0 to 1.0 (inclusive)

    
### Function `set_power` 




>     def set_power(
>         audio,
>         power
>     )


Sets the power of the audio.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```power```**
:   A float, which is the Root Mean Square to set the audio to



##### Returns
A numpy ndarray, which has 1 dimension

    
### Function `set_speed` 




>     def set_speed(
>         audio,
>         rate,
>         seconds
>     )


Sets the speed of the audio and keeps the RMS power the same.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```seconds```**
:   A float, which is the number of seconds the audio
         should be set to



##### Returns
A numpy ndarray, which has 1 dimension

    
### Function `shift_pitch` 




>     def shift_pitch(
>         audio,
>         rate,
>         steps
>     )


Shifts the pitch of the audio.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken



##### Returns
A numpy ndarray, which has 1 dimension

    
### Function `split` 




>     def split(
>         audio,
>         rate,
>         frame_duration,
>         ambient_power=0.0001,
>         min_gap=None
>     )


Splits the audio into audio segments on ambient frames.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```frame_duration```**
:   A float, which is the duration of each frame
                to check


**```ambient_power```**
:   A float, which is the Root Mean Square of ambient noise


**```min_gap```**
:   An integer, which is the number of frames to consider until
         ambient frames are removed



##### Returns
A list of numpy ndarray, which are 1 dimension each and have
    values within -1.0 to 1.0 (inclusive)

    
### Function `trim_all` 




>     def trim_all(
>         audio,
>         rate,
>         frame_duration,
>         ambient_power=0.0001
>     )


Trims ambient silence in the audio anywhere.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```frame_duration```**
:   A float, which is the duration of each frame
                to check


**```ambient_power```**
:   A float, which is the Root Mean Square of ambient noise



##### Returns
A numpy ndarray, which has 1 dimension and values within
    -1.0 to 1.0 (inclusive)

    
### Function `trim_sides` 




>     def trim_sides(
>         audio,
>         rate,
>         frame_duration,
>         ambient_power=0.0001
>     )


Trims ambient silence in the audio only on the sides.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```frame_duration```**
:   A float, which is the duration of each frame
                to check


**```ambient_power```**
:   A float, which is the Root Mean Square of ambient noise



##### Returns
A numpy ndarray, which has 1 dimension and values within
    -1.0 to 1.0 (inclusive)

    
### Function `vad_split` 




>     def vad_split(
>         audio,
>         rate,
>         frame_duration,
>         aggressiveness=1
>     )


Splits the audio into audio segments on non-speech frames.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer, which is the rate at which samples are taken


**```frame_duration```**
:   A float, which is the duration of each frame
                to check



##### Returns
A list of numpy ndarray, which are 1 dimension each and
    have values within -1.0 to 1.0 (inclusive)

    
### Function `vad_trim_all` 




>     def vad_trim_all(
>         audio,
>         rate,
>         frame_duration,
>         aggressiveness=1
>     )


Trims anywhere in the audio that does not contain speech.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer (8000, 16000, 32000, 48000), which is the rate
      at which samples are taken


**```frame_duration```**
:   A float (.01, .02, .03), which is the duration
                of each frame to check


**```aggressiveness```**
:   A integer (0, 1, 2, 3), which is the level of
                aggressiveness to trim non-speech



##### Returns
A numpy ndarray, which has 1 dimension and values within
    -1.0 to 1.0 (inclusive)

    
### Function `vad_trim_sides` 




>     def vad_trim_sides(
>         audio,
>         rate,
>         frame_duration,
>         aggressiveness=1
>     )


Trims the sides in the audio that do not contain speech.


##### Args
**```audio```**
:   A numpy ndarray, which has 1 dimension and values within
       -1.0 to 1.0 (inclusive)


**```rate```**
:   An integer (8000, 16000, 32000, 48000), which is the rate
      at which samples are taken


**```frame_duration```**
:   A float (.01, .02, .03), which is the duration
                of each frame to check


**```aggressiveness```**
:   A integer (0, 1, 2, 3), which is the level of
                aggressiveness to trim non-speech



##### Returns
A numpy ndarray, which has 1 dimension and values within
    -1.0 to 1.0 (inclusive)




    
# Module `paiutils.autoencoder` 

Author: Travis Hammond
Version: 12_21_2020





    
## Classes


    
### Class `AutoencoderExtraDecoderTrainer` 




>     class AutoencoderExtraDecoderTrainer(
>         encoder_model,
>         decoder_model,
>         decoder_model2,
>         data,
>         include_y_data=True
>     )


Autoencoder with an Extra Decoder Trainer is an Autoencoder Trainer
with a extra decoder that can be trained to y-data.

Initializes train, validation, and test data.


##### Args
**```encoder_model```**
:   A compiled keras model


**```decoder_model```**
:   A compiled keras model (full model shares
               optimizer and other attributes with this model)


**```decoder_model2```**
:   The second decoder is trained
                to map the encoder to a
                different output
                (not part of the full model)


**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      Ex. {'train_x': [...], 'train_y: [...]}


**```include_y_data```**
:   A boolean, which determines if y-data should
                be appened with the x-data for training the
                autoencoder




    
#### Ancestors (in MRO)

* [paiutils.autoencoder.AutoencoderTrainer](#paiutils.autoencoder.AutoencoderTrainer)
* [paiutils.neural_network.Trainer](#paiutils.neural_network.Trainer)






    
#### Methods


    
##### Method `eval_extra_decoder` 




>     def eval_extra_decoder(
>         self,
>         train_data=True,
>         validation_data=True,
>         test_data=True,
>         batch_size=None,
>         verbose=True,
>         **kwargs
>     )


Evaluates the second decoder model with the
   train/validation/test data.


##### Args
**```train_data```**
:   A boolean, which determines if
            train_data should be evaluated


**```validation_data```**
:   A boolean, which determines if
                 validation_data should be evaluated


**```test_data```**
:   A boolean, which determines if
           test_data should be evaluated


**```batch_size```**
:   An integer, which is the number of samples
            per graident update


**```verbose```**
:   A boolean, which determines the verbositiy level



##### Returns
A dictionary of the results

    
##### Method `set_data` 




>     def set_data(
>         self,
>         data,
>         include_y_data=True
>     )


Sets train, validation, and test data from data.


##### Args
**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      Ex. {'train_x': [...], 'train_y: [...]}


**```include_y_data```**
:   A boolean, which determines if y-data should
                be appened with the x-data for training the
                autoencoder



    
##### Method `train_extra_decoder` 




>     def train_extra_decoder(
>         self,
>         epochs,
>         batch_size=None,
>         verbose=True,
>         **kwargs
>     )


Trains the second decoder keras model on the outputs
   of the assumingly trained encoder.


##### Args
**```epochs```**
:   An integer, which is the number of complete
        iterations to train


**```batch_size```**
:   An integer, which is the number of samples
            per graident update


**```callbacks```**
:   A list of keras Callback instances,
           which are called during training and validation


**```verbose```**
:   A boolean, which determines the verbositiy level



    
### Class `AutoencoderPredictor` 




>     class AutoencoderPredictor(
>         path,
>         uses_encoder_model=False,
>         uses_decoder_model=False,
>         custom_objects=None
>     )


AutoenocderPredictor is used for loading and predicting keras models.

Initializes the model and weights.


##### Args
**```path```**
:   A string, which is the path to a folder containing
      model.json, weights.h5, note.txt, and maybe encoder/decoder
      parts


**```uses_encoder_model```**
:   A boolean, which determines if encoder model
                    should be used for predictions
                    (cannot also enable uses_decoder_model)


**```uses_decoder_model```**
:   A boolean, which determines if decoder model
                    should be used for predictions
                    (cannot also enable uses_encoder_model)


**```custom_objects```**
:   A dictionary mapping to custom classes
                or functions for loading the model




    
#### Ancestors (in MRO)

* [paiutils.neural_network.Predictor](#paiutils.neural_network.Predictor)






    
### Class `AutoencoderTrainer` 




>     class AutoencoderTrainer(
>         encoder_model,
>         decoder_model,
>         data
>     )


AutoencoderTrainer is used for loading, saving,
and training keras autoencoder models.

Initializes train, validation, and test data.


##### Args
**```encoder_model```**
:   A compiled keras model


**```decoder_model```**
:   A compiled keras model (full model shares
               optimizer and other attributes with this model)


**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      If the train/validation/test key is present without
      the _x the value will be used as a
      generator/Keras-Sequence/TF-Dataset and
      keys with _x will be ignored.
      Ex. {'train_x': [...]}
      Ex. {'train': generator()}
      Ex. {'train': tf.data.Dataset(), 'test': generator()}




    
#### Ancestors (in MRO)

* [paiutils.neural_network.Trainer](#paiutils.neural_network.Trainer)


    
#### Descendants

* [paiutils.autoencoder.AutoencoderExtraDecoderTrainer](#paiutils.autoencoder.AutoencoderExtraDecoderTrainer)
* [paiutils.autoencoder.VAETrainer](#paiutils.autoencoder.VAETrainer)





    
#### Methods


    
##### Method `set_data` 




>     def set_data(
>         self,
>         data
>     )


Sets train, validation, and test data from data.


##### Args
**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      If the train/validation/test key is present without
      the _x the value will be used as a
      generator/Keras-Sequence/TF-Dataset and
      keys with _x will be ignored.
      Ex. {'train_x': [...]}
      Ex. {'train': generator()}
      Ex. {'train': tf.data.Dataset(), 'test': generator()}



    
### Class `VAETrainer` 




>     class VAETrainer(
>         encoder_model,
>         decoder_model,
>         data,
>         rloss_coef=1000,
>         use_logits=True
>     )


VAETrainer is used for loading, saving,
and training keras variational autoencoder models.

Initializes train, validation, and test data.


##### Args
**```encoder_model```**
:   A compiled keras model


**```decoder_model```**
:   A compiled keras model (full model shares
               optimizer and other attributes with this model)


**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      If the train/validation/test key is present without
      the _x the value will be used as a
      generator/Keras-Sequence/TF-Dataset and
      keys with _x will be ignored.
      Ex. {'train_x': [...], 'train_y: [...]}
      Ex. {'train': generator()}
      Ex. {'train': tf.data.Dataset(), 'test': generator()}


**```rloss_coef```**
:   A scalar value, which scales the reconstruction
            loss


**```use_logits```**
:   A boolean that determines if binary crossentropy
            should be used with logit inputs (decoder_model
            loss will be ignored for training)




    
#### Ancestors (in MRO)

* [paiutils.autoencoder.AutoencoderTrainer](#paiutils.autoencoder.AutoencoderTrainer)
* [paiutils.neural_network.Trainer](#paiutils.neural_network.Trainer)



    
#### Class variables


    
##### Variable `VAEModel` 




<code>Model</code> groups layers into an object with training and inference features.


Arguments
-----=
inputs: The input(s) of the model: a <code>keras.Input</code> object or list of
    <code>keras.Input</code> objects.
outputs: The output(s) of the model. See Functional API example below.
name: String, the name of the model.

There are two ways to instantiate a <code>Model</code>:

1 - With the "Functional API", where you start from <code>Input</code>,
you chain layer calls to specify the model's forward pass,
and finally you create your model from inputs and outputs:

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

2 - By subclassing the <code>Model</code> class: in that case, you should define your
layers in <code>\_\_init\_\_</code> and you should implement the model's forward pass
in <code>call</code>.

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

model = MyModel()
```

If you subclass <code>Model</code>, you can optionally have
a <code>training</code> argument (boolean) in <code>call</code>, which you can use to specify
a different behavior in training and inference:

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()
```

Once the model is created, you can config the model with losses and metrics
with <code>model.compile()</code>, train the model with <code>model.fit()</code>, or use the model
to do prediction with <code>model.predict()</code>.






    
# Module `paiutils.evolution_algorithm` 

Author: Travis Hammond
Version: 12_21_2020





    
## Classes


    
### Class `Crossover` 




>     class Crossover


Crossover contains functions that return functions
that mix parent genes and return offspring.






    
#### Static methods


    
##### `Method dual` 




>     def dual(
>         formula=<function Crossover.<lambda>>
>     )


Creates a crossover function that mixes two parents
   to create one offspring.


##### Args
**```formula```**
:   A function that takes Parent A and Parent B
         and returns one Child



##### Returns
A crossover function

    
##### `Method population_avg` 




>     def population_avg()


Creates a crossover function that averages all the parents to
   create all the exact same offspring.


##### Returns
A crossover function

    
##### `Method population_shuffle` 




>     def population_shuffle()


Creates a crossover function that randomly
   shuffles all the genes between all the parents.


##### Returns
A crossover function

    
##### `Method single` 




>     def single(
>         variable_size=False
>     )


Creates a crossover function that does not perform
   any crossover, but instead creates a child from a
   single parent. (Parents may produce more than one child)


##### Args
**```variable_size```**
:   A boolean, which determines if the
               genome size can change



##### Returns
A crossover function

    
##### `Method triple` 




>     def triple(
>         formula=<function Crossover.<lambda>>
>     )


Creates a crossover function that mixes three parents
   to create one offspring.


##### Args
**```formula```**
:   A function that takes Parent A, Parent B, and Parent C
         and returns one Child



##### Returns
A crossover function


    
### Class `EvolutionAlgorithm` 




>     class EvolutionAlgorithm(
>         fitness_func,
>         selection_func,
>         mutation_func,
>         crossover_func,
>         size_mutation_func=None
>     )


EvolutionAlgorithm is a class that is able to simulate 'natural'
selection of genes and genomes.

Creates an evolution algorithm by the provided functions.


##### Args
**```fitness_func```**
:   A function that takes a list or numpy ndarray of
              genomes (offspring), and returns list of fitness
              values


**```selection_func```**
:   A function that takes a list or numpy ndarray of
                genomes (offspring) and fitness values, and
                returns the selected genomes (offspring)


**```mutation_func```**
:   A function that takes a list or numpy ndarray of
               genomes (offspring), and returns the offspring
               mutated


**```crossover_func```**
:   A function that takes a list or numpy array of
                genomes (parents), and returns offspring


**```size_mutation_func```**
:   A function that takes a list or numpy
                    ndarray of genomes, and returns the
                    genomes with mutated sizes









    
#### Methods


    
##### Method `simulate` 




>     def simulate(
>         self,
>         base_genome,
>         generations,
>         population,
>         selection_size,
>         return_all_genomes=False,
>         verbose=True
>     )


Simulates natural selection of genomes.


##### Args
**```base_genome```**
:   A list of floats or integers (genes)


**```generations```**
:   An integer, which is the number of complete cycles of
             performing crossovers, mutations, and selections on
             the entire population


**```population```**
:   An integer, which is the number of genomes in a
            generation


**```selection_size```**
:   An integer, which is the number of
                offspring to select from the population
                each generation/cycle


**```return_all_genomes```**
:   A boolean, which determiens if all
                    the genomes with their corresponding
                    fitness values should be returned


**```verbose```**
:   A boolean, which determines if information
         will be printed to the console



##### Returns
A list of tuples each containing a fitness value and a genome

    
##### Method `simulate_islands` 




>     def simulate_islands(
>         self,
>         base_genome,
>         generations,
>         population,
>         selection_size,
>         islands,
>         island_migrations,
>         threaded=False,
>         verbose=True
>     )


Simulates natural selection of genomes with isolating islands.


##### Args
**```base_genome```**
:   A list of floats or integers (genes)


**```generations```**
:   An integer, which is the number of complete cycles of
             performing crossovers, mutations, and selections on
             the entire population


**```population```**
:   An integer, which is the number of genomes in a
            generation


**```selection_size```**
:   An integer, which is the number of
                offspring to select from the population
                each generation/cycle


**```islands```**
:   An integer, which is the number of isolated islands in
         the simulation


**```island_migrations```**
:   An integer, which is the number of migrations
                   of the offspring between the isolated islands


**```threaded```**
:   A boolean, which determines if the islands should be run
          on in parallel


**```verbose```**
:   A boolean, which determines if information
         will be printed to the console



##### Returns
A list of tuples each containing a fitness value and a genome

    
### Class `Fitness` 




>     class Fitness


Fitness contains functions that return functions that
calculate fitness.






    
#### Static methods


    
##### `Method match_mse` 




>     def match_mse(
>         target_genome,
>         variable_size=False
>     )


Creates a fitness_func that computes
   the the mean squared error of the target
   genome and the offsprings' genome.


##### Args
**```target_genome```**
:   A list or numpy array that is the
             target genome


**```variable_size```**
:   A boolean, which determines if the
               genome size can change



##### Returns
A fitness function


    
### Class `HyperparameterTuner` 




>     class HyperparameterTuner


This class is used for tuning hyper parameters.

Initalizes lists to keep track of parameters.







    
#### Methods


    
##### Method `boolean` 




>     def boolean(
>         self,
>         volatility=0.1,
>         inital_value=True
>     )


Returns a function that when called returns the
   value of that parameter.


##### Args
**```volatility```**
:   A float, which is the rate that this parameter
            is mutated


**```inital_value```**
:   A boolean, which is the starting value
              of the parameter



##### Returns
A parameter function, which returns a boolean

    
##### Method `list` 




>     def list(
>         self,
>         alist,
>         volatility=0.1,
>         inital_ndx=None
>     )


Returns a function that when called returns a element
   from the list.


##### Args
**```alist```**
:   A list of values, which can be mutated to


**```volatility```**
:   A float, which is the rate that this parameter
            is mutated


**```inital_ndx```**
:   A integer, which is the starting index
            of the parameter



##### Returns
A parameter function, which returns a number in the
    uniform range

    
##### Method `tune` 




>     def tune(
>         self,
>         generations,
>         population,
>         selection_size,
>         eval_func,
>         lowest_best=True,
>         crossover_func=None,
>         verbose=False
>     )


Tunes the parameters to get the best parameters with
   an evolution algorithim.


##### Args
**```generations```**
:   An integer, which is the number of complete cycles of
             performing crossovers, mutations, and selections on
             the entire population


**```population```**
:   An integer, which is the number of genomes in a
            generation


**```selection_size```**
:   An integer, which is the number of
                parameter combinations to select from the
                population each generation/cycle


**```eval_func```**
:   A function, which returns a single value that
           represents the parameters fitness


**```lowest_best```**
:   A boolean, which determines if lower fitness values
             are better or worse


**```crossover_func```**
:   A function that takes a list or numpy array of
                genomes (parents), and returns offspring
                (defaults to no crossover)


**```verbose```**
:   A boolean, which determines if the evolution
         algorithm should print information to the screen



    
##### Method `uniform` 




>     def uniform(
>         self,
>         lower_bound,
>         upper_bound,
>         volatility=0.1,
>         inital_value=None,
>         integer=False
>     )


Returns a function that when called returns the
   value of that parameter.


##### Args
**```lower_bound```**
:   A float or integer, which is the lowest
             value that the parameter can be mutated to


**```upper_bound```**
:   A float or integer, which is the highest
             value that the parameter can be mutated to


**```volatility```**
:   A float, which is the rate that this parameter
            is mutated


**```inital_value```**
:   A float or integer, which is the starting value
              of the parameter


**```integer```**
:   A boolean, which determiens if the parameter should
          be rounded and cast to an integer



##### Returns
A parameter function, which returns a number in the
    uniform range

    
### Class `Mutation` 




>     class Mutation


Mutation contains functions that return functions
that mutate genes.






    
#### Static methods


    
##### `Method additive` 




>     def additive(
>         mutation_rates,
>         distributions,
>         normal=True,
>         round_values=False,
>         variable_size=False
>     )


Creates a mutation function that can add to the current value
   of a gene.


##### Args
**```mutation_rates```**
:   A list of floats within 0-1 (exclusive),
                or a single float if variable size is True


**```distributions```**
:   A list of either lower and upper bounds
               or means and standard deviations
               (depends on param normal), or a single
               distribution if variable size is True


**```normal```**
:   A boolean, which determines if the random distribution
        is normal or uniform


**```round_values```**
:   A boolean, which determines if mutations should be
              rounded to the nearest whole integer


**```variable_size```**
:   A boolean, which determines if the number of genes
               in the genome can change



##### Returns
A mutation function

    
##### `Method variable` 




>     def variable(
>         mutation_rates,
>         distributions,
>         normal=True,
>         round_values=False,
>         variable_size=False
>     )


Creates a mutation function that can sets the value
   of a gene.


##### Args
**```mutation_rates```**
:   A list of floats within 0-1 (exclusive),
                or a single float if variable size is True


**```distributions```**
:   A list of either lower and upper bounds
               or means and standard deviations
               (depends on param normal), or a single
               distribution if variable size is True


**```normal```**
:   A boolean, which determines if the random distribution
        is normal or uniform


**```round_values```**
:   A boolean, which determines if mutations should be
              rounded to the nearest whole integer


**```variable_size```**
:   A boolean, which determines if the number of genes
               in the genome can change



##### Returns
A mutation function


    
### Class `Selection` 




>     class Selection


Selection contains functions that return functions
that select offspring based on fitness values.






    
#### Static methods


    
##### `Method select_highest` 




>     def select_highest(
>         variable_size=False
>     )


Creates a selection function that selects offspring
   with the highest fitness value.


##### Args
**```variable_size```**
:   A boolean, which determines if the
               genome size can change



##### Returns
A selection function

    
##### `Method select_lowest` 




>     def select_lowest(
>         variable_size=False
>     )


Creates a selection function that selects offspring
   with the lowest fitness value.


##### Args
**```variable_size```**
:   A boolean, which determines if the
               genome size can change



##### Returns
A selection function


    
### Class `SizeMutation` 




>     class SizeMutation


SizeMutation contains functions that return functions
that mutate the genome size.






    
#### Static methods


    
##### `Method complete_mutations` 




>     def complete_mutations(
>         size_mutation_rate,
>         probabilities,
>         funcs
>     )


Creates a complete size mutation function from incomplete
   size mutation functions.


##### Args
**```size_mutation_rate```**
:   A float within 0-1 (exclusive), which is the
                    rate of a genome size mutating


**```probabilities```**
:   A list of floats within 0-1 (exclusive), which
               contains the chance of each size mutation function
               being used


**```funcs```**
:   A list of incomplete size mutation functions



##### Returns
A complete size mutation function

    
##### `Method first_gene_addition` 




>     def first_gene_addition(
>         value=None
>     )


Creates a size mutation function that inserts
   a gene in the begining of the genome.


##### Args
**```value```**
:   A value to set the new gene to
       (Default copies current genome value)



##### Returns
An incomplete size mutation function

    
##### `Method first_gene_deletion` 




>     def first_gene_deletion()


Creates a size mutation function that deletes
   a gene at the begining of the genome.


##### Returns
An incomplete size mutation function

    
##### `Method genome_double` 




>     def genome_double(
>         value=None
>     )


Creates a size mutation function that doubles the
   size of the current genome.


##### Args
**```value```**
:   A value to set the new genes to
    (Default copies current genome values)



##### Returns
An incomplete size mutation function

    
##### `Method genome_half` 




>     def genome_half(
>         keep_left=True
>     )


Creates a size mutation function that halfs the
   size of the current genome.


##### Args
**```keep_left```**
:   A boolean, which determines if the
           left or right size should be kept



##### Returns
An incomplete size mutation function

    
##### `Method last_gene_addition` 




>     def last_gene_addition(
>         value=None
>     )


Creates a size mutation function that inserts
   a gene at the end of the genome.


##### Args
**```value```**
:   A value to set the new gene to
       (Default copies current genome value)



##### Returns
An incomplete size mutation function

    
##### `Method last_gene_deletion` 




>     def last_gene_deletion()


Creates a size mutation function that deletes
   a gene at the end of the genome.


##### Returns
An incomplete size mutation function

    
##### `Method random_gene_addition` 




>     def random_gene_addition(
>         value=None
>     )


Creates a size mutation function that randomly
   inserts a gene in the genome.


##### Args
**```value```**
:   A value to set the new gene to
       (Default copies current genome value)



##### Returns
An incomplete size mutation function

    
##### `Method random_gene_deletion` 




>     def random_gene_deletion()


Creates a size mutation function that randomly
   deletes a gene in the genome.


##### Returns
An incomplete size mutation function




    
# Module `paiutils.gan` 

Author: Travis Hammond
Version: 12_21_2020





    
## Classes


    
### Class `CycleGANPredictor` 




>     class CycleGANPredictor(
>         path,
>         uses_x_model=True,
>         uses_generator=True,
>         custom_objects=None
>     )


CycleGANPredictor is used for loading and predicting Cylce GAN keras models.

Initializes the model and weights.


##### Args
**```path```**
:   A string, which is the path to a folder containing
      model.json, weights.h5, note.txt, and maybe encoder/decoder
      parts


**```uses_x_model```**
:   A boolean, which determines if the x or y
              model should be loaded


**```uses_generator```**
:   A boolean, which determines if the generator
                or discriminator should be loaded


**```custom_objects```**
:   A dictionary mapping to custom classes
                or functions for loading the model




    
#### Ancestors (in MRO)

* [paiutils.neural_network.Predictor](#paiutils.neural_network.Predictor)






    
### Class `CycleGANTrainer` 




>     class CycleGANTrainer(
>         gen_model,
>         dis_model,
>         data,
>         idt_loss_coef=0,
>         cycle_loss_coef=10
>     )


Cycle Generative Adversarial Network Trainer is used for
loading, saving, and training keras GAN models.

Initializes data and GANModel.


##### Args
**```gen_model```**
:   A compiled keras model, which is the generator


**```dis_model```**
:   A compiled keras model, which is the discriminator


**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      If the train/validation/test key is present without
      the _x/_y the value will be used as a
      generator/Keras-Sequence/TF-Dataset and
      keys with _x/_y will be ignored.
      Ex. {'train_x': [...], 'train_y: [...]}
      Ex. {'train': generator()}
      Ex. {'train': tf.data.Dataset(), 'test': generator()}


**```idt_loss_coef```**
:   A float, which is the amount of the identity
               loss (generator model's loss function)
               to be added to the generator loss


**```cycle_loss_coef```**
:   A float, which is the amount of the cycle
                 loss to be added to the gen model loss




    
#### Ancestors (in MRO)

* [paiutils.neural_network.Trainer](#paiutils.neural_network.Trainer)



    
#### Class variables


    
##### Variable `GANModel` 




<code>Model</code> groups layers into an object with training and inference features.


Arguments
-----=
inputs: The input(s) of the model: a <code>keras.Input</code> object or list of
    <code>keras.Input</code> objects.
outputs: The output(s) of the model. See Functional API example below.
name: String, the name of the model.

There are two ways to instantiate a <code>Model</code>:

1 - With the "Functional API", where you start from <code>Input</code>,
you chain layer calls to specify the model's forward pass,
and finally you create your model from inputs and outputs:

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

2 - By subclassing the <code>Model</code> class: in that case, you should define your
layers in <code>\_\_init\_\_</code> and you should implement the model's forward pass
in <code>call</code>.

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

model = MyModel()
```

If you subclass <code>Model</code>, you can optionally have
a <code>training</code> argument (boolean) in <code>call</code>, which you can use to specify
a different behavior in training and inference:

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()
```

Once the model is created, you can config the model with losses and metrics
with <code>model.compile()</code>, train the model with <code>model.fit()</code>, or use the model
to do prediction with <code>model.predict()</code>.




    
### Class `GANIPredictor` 




>     class GANIPredictor(
>         path,
>         uses_generator=True,
>         custom_objects=None
>     )


GANIPredictor is used for loading and predicting GANI keras models.

Initializes the model and weights.


##### Args
**```path```**
:   A string, which is the path to a folder containing
      model.json, weights.h5, note.txt, and maybe encoder/decoder
      parts


**```uses_generator```**
:   A boolean, which determines if the generator
                or discriminator should be loaded


**```custom_objects```**
:   A dictionary mapping to custom classes
                or functions for loading the model




    
#### Ancestors (in MRO)

* [paiutils.neural_network.Predictor](#paiutils.neural_network.Predictor)






    
### Class `GANITrainer` 




>     class GANITrainer(
>         gen_model,
>         dis_model,
>         data,
>         idt_loss_coef=0
>     )


Generative Adversarial Network with provided Inputs
Trainer is used for loading, saving, and training
keras GAN models that do not have random inputs.

Initializes data and GANModel.


##### Args
**```gen_model```**
:   A compiled keras model, which is the generator


**```dis_model```**
:   A compiled keras model, which is the discriminator


**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      If the train/validation/test key is present without
      the _x/_y the value will be used as a
      generator/Keras-Sequence/TF-Dataset and
      keys with _x/_y will be ignored.
      Ex. {'train_x': [...], 'train_y: [...]}
      Ex. {'train': generator()}
      Ex. {'train': tf.data.Dataset(), 'test': generator()}


**```idt_loss_coef```**
:   A float, which is the amount of the identity
               loss (generator model's loss function)
               to be added to the generator loss




    
#### Ancestors (in MRO)

* [paiutils.neural_network.Trainer](#paiutils.neural_network.Trainer)



    
#### Class variables


    
##### Variable `GANModel` 




<code>Model</code> groups layers into an object with training and inference features.


Arguments
-----=
inputs: The input(s) of the model: a <code>keras.Input</code> object or list of
    <code>keras.Input</code> objects.
outputs: The output(s) of the model. See Functional API example below.
name: String, the name of the model.

There are two ways to instantiate a <code>Model</code>:

1 - With the "Functional API", where you start from <code>Input</code>,
you chain layer calls to specify the model's forward pass,
and finally you create your model from inputs and outputs:

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

2 - By subclassing the <code>Model</code> class: in that case, you should define your
layers in <code>\_\_init\_\_</code> and you should implement the model's forward pass
in <code>call</code>.

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

model = MyModel()
```

If you subclass <code>Model</code>, you can optionally have
a <code>training</code> argument (boolean) in <code>call</code>, which you can use to specify
a different behavior in training and inference:

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()
```

Once the model is created, you can config the model with losses and metrics
with <code>model.compile()</code>, train the model with <code>model.fit()</code>, or use the model
to do prediction with <code>model.predict()</code>.




    
### Class `GANPredictor` 




>     class GANPredictor(
>         path,
>         noise_fn=None,
>         weights_name='gen_model_weights.h5',
>         model_name='gen_model.json',
>         custom_objects=None
>     )


Generative Adversarial Network Predictor is used for
loading and predicting keras GAN models.

Loads the model and weights.


##### Args
**```path```**
:   A string, which is the path to a folder containing
      model.json, weights.h5, and maybe note.txt


**```noise_fn```**
:   A function for generating input for the GAN
          (Default: tf.random.uniform)


**```weights_name```**
:   A string, which is the name of the weights to load


**```model_name```**
:   A string, which is the name of the model to load


**```custom_objects```**
:   A dictionary mapping to custom classes
                or functions for loading the model




    
#### Ancestors (in MRO)

* [paiutils.neural_network.Predictor](#paiutils.neural_network.Predictor)






    
#### Methods


    
##### Method `generate` 




>     def generate(
>         self,
>         n=1,
>         conditions=None
>     )


Generates n samples.


##### Args
**```n```**
:   A integer, which is the number of
     samples to produce


**```conditions```**
:   A ndarray of model conditional input



##### Returns
A result from the model output

    
##### Method `predict` 




>     def predict(
>         self,
>         noise,
>         condition=None
>     )


Predicts on a single sample.


##### Args
**```noise```**
:   A 1D ndarray for the input of the model


**```condition```**
:   A ndarray of model conditional input



##### Returns
A result from the model output

    
##### Method `predict_all` 




>     def predict_all(
>         self,
>         noise,
>         conditions=None,
>         batch_size=None
>     )


Predicts on many samples.


##### Args
**```noise```**
:   A 1D ndarray for the input of the model


**```conditions```**
:   A ndarray of model conditional input



##### Returns
A result from the model output

    
### Class `GANTrainer` 




>     class GANTrainer(
>         gen_model,
>         dis_model,
>         data,
>         conditional=False,
>         noise_fn=None,
>         idt_loss_coef=0
>     )


Generative Adversarial Network Trainer is used for loading, saving,
and training keras GAN models.

Initializes data and GANModel.


##### Args
**```gen_model```**
:   A compiled keras model, which is the generator


**```dis_model```**
:   A compiled keras model, which is the discriminator


**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      If the train/validation/test key is present without
      the _x/_y the value will be used as a
      generator/Keras-Sequence/TF-Dataset and
      keys with _x/_y will be ignored.
      Ex. {'train_x': [...], 'train_y: [...]}
      Ex. {'train': generator()}
      Ex. {'train': tf.data.Dataset(), 'test': generator()}


**```conditional```**
:   A boolean, which determines if the GAN is a
             conditional GAN


**```noise_fn```**
:   A TF function that takes a shape and returns
          noise (Default: uniform noise)


**```idt_loss_coef```**
:   A float, which is the amount of the identity
               loss (generator model's loss function)
               to be added to the generator loss




    
#### Ancestors (in MRO)

* [paiutils.neural_network.Trainer](#paiutils.neural_network.Trainer)



    
#### Class variables


    
##### Variable `GANModel` 




<code>Model</code> groups layers into an object with training and inference features.


Arguments
-----=
inputs: The input(s) of the model: a <code>keras.Input</code> object or list of
    <code>keras.Input</code> objects.
outputs: The output(s) of the model. See Functional API example below.
name: String, the name of the model.

There are two ways to instantiate a <code>Model</code>:

1 - With the "Functional API", where you start from <code>Input</code>,
you chain layer calls to specify the model's forward pass,
and finally you create your model from inputs and outputs:

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

2 - By subclassing the <code>Model</code> class: in that case, you should define your
layers in <code>\_\_init\_\_</code> and you should implement the model's forward pass
in <code>call</code>.

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

model = MyModel()
```

If you subclass <code>Model</code>, you can optionally have
a <code>training</code> argument (boolean) in <code>call</code>, which you can use to specify
a different behavior in training and inference:

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()
```

Once the model is created, you can config the model with losses and metrics
with <code>model.compile()</code>, train the model with <code>model.fit()</code>, or use the model
to do prediction with <code>model.predict()</code>.






    
# Module `paiutils.image` 

Author: Travis Hammond
Version: 12_21_2020




    
## Functions


    
### Function `apply_clahe` 




>     def apply_clahe(
>         image,
>         clip_limit=40.0,
>         tile_grid_size=(8, 8)
>     )


Applys CLAHE (Contrast Limited Adaptive Histogram Equalization).


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions (preferably 2)


**```clip_limit```**
:   A float, which is the threshold for contrasting


**```tile_grid_size```**
:   A tuple of 2 natural numbers, which is the number
                of rows and columns, respectively



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `bgr2hls` 




>     def bgr2hls(
>         image
>     )


Converts a BGR image to a HLS image.


##### Args
**```image```**
:   A numpy ndarray, which has 3 dimensions



##### Returns
A numpy ndarray, which has 3 dimensions

    
### Function `bgr2hsv` 




>     def bgr2hsv(
>         image
>     )


Converts a BGR image to a HSV image.


##### Args
**```image```**
:   A numpy ndarray, which has 3 dimensions



##### Returns
A numpy ndarray, which has 3 dimensions

    
### Function `bgr2rgb` 




>     def bgr2rgb(
>         image
>     )


Converts a BGR image to a RGB image.


##### Args
**```image```**
:   A numpy ndarray, which has 3 dimensions



##### Returns
A numpy ndarray, which has 3 dimensions

    
### Function `blend` 




>     def blend(
>         image1,
>         image2,
>         image1_weight=0.5,
>         image2_weight=None
>     )


Blends two images together.


##### Args
**```image1```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```image2```**
:   A numpy ndarray, which has same dimensions as image1


**```image1_weight```**
:   A float, which is the intensity of image1
               to preserve


**```image2_weight```**
:   A float, which is the intensity of image2
               to preserve



##### Returns
A numpy ndarray, which has the same number of dimensions as image1

    
### Function `compute_color_ranges` 




>     def compute_color_ranges(
>         images,
>         percentage_captured=50,
>         num_bounds=1,
>         use_evolution_algo=False
>     )


Computes the color ranges that captures a percentage of the image.
This algorithm is not well designed and is mainly for testing purposes.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions (BGR)


**```percentage_captured```**
:   An integer within 0-100 (inclusive), which acts
                     as the threshold for the bounds


**```num_bounds```**
:   An integer, which does nothing


**```use_evolution_algo```**
:   A boolean, which does nothing



##### Returns
A tuple of 2 list with the former containing lower
    bounds and the latter upper bounds

    
### Function `create_histograms` 




>     def create_histograms(
>         images,
>         hsv_images=False,
>         channels=None,
>         vrange=None,
>         num_bins=None
>     )


Create histograms.


##### Args
**```image```**
:   A list of numpy ndarray, which each ndarray is 2 or
       3 dimensions (must all have same dimensions)


**```hsv_images```**
:   A boolean, which determines if the image is HSV


**```channels```**
:   A list of integers within 0-2 (inclusive), which are the
          channels to get the histograms of


**```vrange```**
:   A list the same length as channels with list containing 2
        integers containing the lower and upper+1 value of a channel


**```num_bins```**
:   An integer, which is the number of bins to have for
          the histograms



##### Returns
A numpy ndarray, which is a list of the histograms

    
### Function `create_magnitude_spectrum` 




>     def create_magnitude_spectrum(
>         image
>     )


Creates a magnitude spectrum from image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions (BGR)



##### Returns
A numpy ndarray, which has 2 dimensions

    
### Function `create_mask_of_colors_in_range` 




>     def create_mask_of_colors_in_range(
>         image,
>         lower_bounds,
>         upper_bounds
>     )


Creates a mask of the colors in within the lower and upper bounds.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions (BGR)


**```lower_bounds```**
:   A tuple of 3 integers (HSV), which is the lower bound


**```upper_bounds```**
:   A tuple of 3 integers (HSV), which is the upper bound



##### Returns
A numpy ndarray, which has 2 dimensions

    
### Function `crop` 




>     def crop(
>         image,
>         shape,
>         horizontal_center=0,
>         vertical_center=0
>     )


Crops the image with a given center coord and the shape of a rectangle.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```shape```**
:   A tuple of 2 integers, which is the shape of the rectangle


**```horizontal_center```**
:   An integer, which is the offset from the
                   image's horizontal center


**```vertical_center```**
:   An integer, which is the  offset from the
                 image's vertical center



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `crop_rect` 




>     def crop_rect(
>         image,
>         vertical,
>         horizontal,
>         width,
>         height
>     )


Crops a rectangle out of the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```vertical```**
:   An integer, which is the vertical coord for the top left
          of the rectangle


**```horizontal```**
:   An integer, which is the horizontal coord for the top left
          of the rectangle


**```width```**
:   An integer, which is the width of the rectangle


**```height```**
:   An integer, which is the height of the rectangle



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `crop_rect_coords` 




>     def crop_rect_coords(
>         image,
>         vertical1,
>         horizontal1,
>         vertical2,
>         horizontal2
>     )


Crops a rectangle out of the image through two coords.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```vertical1```**
:   An integer, which is the vertical coord for the top left
           of the rectangle


**```horizontal1```**
:   An integer, which is the horizontal coord for the top left
             of the rectangle


**```vertical2```**
:   An integer, which is the vertical coord for the bottom right
           of the rectangle


**```horizontal2```**
:   An integer, which is the horizontal coord for the bottom
             right of the rectangle



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `denormalize` 




>     def denormalize(
>         image
>     )


Denormalizes an image that is between -1 and 1 to 0 and 255.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions and is normalized



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `equalize` 




>     def equalize(
>         image
>     )


Equalizes the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `freq_filter_image` 




>     def freq_filter_image(
>         image,
>         high=True
>     )


Filters frequencies in the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions (BGR)



##### Returns
A numpy ndarray, which has 2 dimensions

    
### Function `gray` 




>     def gray(
>         image
>     )


Converts a BGR image to a grayscale image.


##### Args
**```image```**
:   A numpy ndarray, which has 3 dimensions



##### Returns
A numpy ndarray, which has 2 dimensions

    
### Function `hflip` 




>     def hflip(
>         image
>     )


Horizontally flips the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `hls2bgr` 




>     def hls2bgr(
>         image
>     )


Converts a HLS image to a BGR image.


##### Args
**```image```**
:   A numpy ndarray, which has 3 dimensions



##### Returns
A numpy ndarray, which has 3 dimensions

    
### Function `hsv2bgr` 




>     def hsv2bgr(
>         image
>     )


Converts a HSV image to a BGR image.


##### Args
**```image```**
:   A numpy ndarray, which has 3 dimensions



##### Returns
A numpy ndarray, which has 3 dimensions

    
### Function `increase_brightness` 




>     def increase_brightness(
>         image,
>         percentage,
>         relative=False
>     )


Increases the brightness of image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```percentage```**
:   An integer, which is how much to increase


**```relative```**
:   A boolean, which determines if the percentage is
          is in terms of max brightness or current brightness



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `load` 




>     def load(
>         filename,
>         target_shape=None,
>         color=True
>     )


Loads an image from a file.


##### Args
**```filename```**
:   A string, which is the directory or filename of the
          file to load


**```target_shape```**
:   A tuple with the vertical size then horizontal size


**```color```**
:   A boolean, which determines if the image should be
       converted to gray scale



##### Returns
A numpy ndarray, which has 2 or 3 dimensions

    
### Function `normalize` 




>     def normalize(
>         image
>     )


Normalizes an image between -1 and 1.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `pad` 




>     def pad(
>         image,
>         ts=0,
>         bs=0,
>         ls=0,
>         rs=0,
>         color=(0, 0, 0)
>     )


Pads the image through adding pixels to each side of the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```ts```**
:   An integer, which is the amount to pad the top side
    of the image


**```bs```**
:   An integer, which is the amount to pad the bottom side
    of the image


**```ls```**
:   An integer, which is the amount to pad the left side
    of the image


**```rs```**
:   An integer, which is the amount to pad the right side
    of the image


**```color```**
:   A tuple of 3 integers or an integer with a range of
       0-255 (inclusive), which is the color of the padding



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `pyr` 




>     def pyr(
>         image,
>         level
>     )


Resize image using pyramids.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```level```**
:   An integer, which if positive enlarges and if negative reduces


returns: A numpy ndarray, which has the same number of dimensions
         of the image

    
### Function `resize` 




>     def resize(
>         image,
>         target_shape,
>         interpolation=None
>     )


Resizes an image to a targeted shape.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```target_shape```**
:   A tuple with the vertical size then horizontal size


**```interpolation```**
:   A cv2 interpolation



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `rgb2bgr` 




>     def rgb2bgr(
>         image
>     )


Converts a RGB image to a BGR image.


##### Args
**```image```**
:   A numpy ndarray, which has 3 dimensions



##### Returns
A numpy ndarray, which has 3 dimensions

    
### Function `rotate` 




>     def rotate(
>         image,
>         angle
>     )


Rotates the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```angle```**
:   A float, which is in terms of degress



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `save` 




>     def save(
>         filename,
>         image,
>         target_shape=None,
>         color=True
>     )


Saves an image to a file.


##### Args
**```filename```**
:   A string, which is the directory or filename to save image to


**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```target_shape```**
:   A tuple with the vertical size then horizontal size


**```color```**
:   A boolean, which determines if the image should be
       converted to gray scale



    
### Function `set_brightness` 




>     def set_brightness(
>         image,
>         percentage,
>         relative=False
>     )


Sets the brightness of image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```percentage```**
:   An integer, which is how much to increase


**```relative```**
:   A boolean, which determines if the percentage is
          is in terms of max brightness or current brightness



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `set_gamma` 




>     def set_gamma(
>         image,
>         gamma=1.0
>     )


Set gamma levels of the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```gamma```**
:   A float, which is the amount to change the images gamma



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `shrink_sides` 




>     def shrink_sides(
>         image,
>         ts=0,
>         bs=0,
>         ls=0,
>         rs=0
>     )


Shrinks/crops the image through shrinking each side of the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```ts```**
:   An integer, which is the amount to shrink the top side
    of the image


**```bs```**
:   An integer, which is the amount to shrink the bottom side
    of the image


**```ls```**
:   An integer, which is the amount to shrink the left side
    of the image


**```rs```**
:   An integer, which is the amount to shrink the right side
    of the image



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `transform_perspective` 




>     def transform_perspective(
>         image,
>         pts,
>         shape
>     )


Transforms the perspective of an image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```pts```**
:   A list of list with 2 integers (possibly floats)


**```shape```**
:   A tuple of 2 integers



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `translate` 




>     def translate(
>         image,
>         vertical=0,
>         horizontal=0
>     )


Translates the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```vertical```**
:   An integer (possibly a float), which is the amount to
          shift the image vertically


**```horizontal```**
:   An integer (possibly a float), which is the amount to
          shift the image horizontally



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `unsharp_mask` 




>     def unsharp_mask(
>         image,
>         kernel_shape=(5, 5),
>         sigma=1.0,
>         amount=1.0,
>         threshold=0
>     )


Sharpens the image through the unsharp masking technique.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```kernel_shape```**
:   A tuple of 2 integers, which is the shape of the
              blurring kernel


**```sigma```**
:   A float, which is the standard deviation of the Gaussian blur


**```amount```**
:   A float, which is the amount to subtracted the blurred
        image from the image


**```threshold```**
:   An integer within 0-255 (inclusive), which is the low
           contrast threshold to copy the image to the sharpened image



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `vflip` 




>     def vflip(
>         image
>     )


Vertically flips the image.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions



##### Returns
A numpy ndarray, which has the same number of dimensions as image

    
### Function `zoom` 




>     def zoom(
>         image,
>         shape,
>         horizontal_center=0,
>         vertical_center=0
>     )


Zooms the image to shape on a given center coord.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```shape```**
:   A tuple of 2 integers, which is the shape of the zoomed image


**```horizontal_center```**
:   An integer, which is the horizontal offset from
                   the image's center


**```vertical_center```**
:   An integer, which is the vertical offset from
                 the image's center



##### Returns
A numpy ndarray, which has the same number of dimensions as image


    
## Classes


    
### Class `Camera` 




>     class Camera(
>         fps=30,
>         camera_device=0
>     )


This class is used for capturing pictures with the
computer's camera.

Initializes the camera and checks if it worked.


##### Args
**```fps```**
:   An integer, which is the number of frames per second


**```camera_device```**
:   An integer, which determines the device to use









    
#### Methods


    
##### Method `capture` 




>     def capture(
>         self,
>         filename=None,
>         target_shape=None,
>         color=True
>     )


Uses the camera object to capture an iamge.


##### Args
**```filename```**
:   A string, which is the directory or filename to
          save image to


**```target_shape```**
:   A tuple with the vertical size then horizontal
              size


**```color```**
:   A boolean, which determines if the image should be
       converted to gray scale



##### Returns
None or a numpy ndarray, which has 2 or 3 dimensions

    
##### Method `close` 




>     def close(
>         self
>     )




    
##### Method `open` 




>     def open(
>         self
>     )




    
##### Method `record` 




>     def record(
>         self,
>         num_frames=None,
>         filename=None,
>         target_shape=None,
>         color=True
>     )


Uses the camera object to capture many iamges in a row.


##### Args
**```num_frmaes```**
:   An integer, which is the number of frames to capture


**```filename```**
:   A string, which is the directory or filename to
          save image to


**```target_shape```**
:   A tuple with the vertical size then horizontal
              size


**```color```**
:   A boolean, which determines if the image should be
       converted to gray scale



##### Returns
None or a list of numpy ndarrays, which have 2 or 3 dimensions

    
### Class `HistogramBackProjector` 




>     class HistogramBackProjector(
>         object_image
>     )


This Class is used to find objects of interest in an image.

Initializes the HBP by computing the object image's histogram.


##### Args
**```object_image```**
:   A numpy ndarray, which has 3 dimensions (BGR)









    
#### Methods


    
##### Method `backproject` 




>     def backproject(
>         self,
>         image,
>         raw=False,
>         threshold=50,
>         disc_kernel=(5, 5)
>     )


Back projects the image to the object image.


##### Args
**```image```**
:   A numpy ndarray, which has 3 dimensions


**```raw```**
:   A boolean, which determines if the output image
     is thresholded


**```threshold```**
:   An integer, which is the threshold of the back
           projected image


**```disc_kernel```**
:   A tuple of 2 integers, which is the size of the
             kernel for filtering



##### Returns
A numpy ndarray, which has 3 dimensions

    
### Class `LockDict` 




>     class LockDict(
>         dict_=None
>     )


This class is used by camera and is a thread safe dict.

Initializes the LockDict.


##### Args
**```dict_```**
:   A dictionary









    
#### Methods


    
##### Method `items` 




>     def items(
>         self
>     )


Returns a list of all the keys and values.


##### Returns
A list of tuples with key then value

    
##### Method `keys` 




>     def keys(
>         self
>     )


Returns a set of all the keys.


##### Returns
A set

    
##### Method `values` 




>     def values(
>         self
>     )


Returns a list of all the values.


##### Returns
A list

    
### Class `TemplateMatcher` 




>     class TemplateMatcher(
>         template,
>         mask=None
>     )


This class is used to find parts of an image that match a template.

Initializes the TemplateMatcher by converting and setting the template.


##### Args
**```template```**
:   A numpy ndarray, which has 2 or 3 dimensions (BGR)


**```mask```**
:   A numpy ndarray, which acts as a binary mask or weights






    
#### Class variables


    
##### Variable `methods` 









    
#### Methods


    
##### Method `match_coords` 




>     def match_coords(
>         self,
>         image,
>         method=5
>     )


Finds the top left point and dimensions (width, height)
   of a subimage that most matches the template.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions (BGR)


**```method```**
:   A cv2 constant or integer, which determines the
        method of finding a match


returns: A tuple of 2 tuples with 2 integers in each
         ((left, top), (width, height)) and a float
         of the confidence

    
##### Method `match_draw_all_rects` 




>     def match_draw_all_rects(
>         self,
>         image,
>         threshold=0.8,
>         color=(0, 255, 0),
>         thickness=2,
>         method=5
>     )


Finds the top left point and dimensions (width, height)
   of all subimages that match the template and then
   draws a rectange with those coords.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions (BGR)


**```threshold```**
:   A float, which is the threshold for being a match
           (higher more of a match)


**```color```**
:   A tuple of 1 or 3 integers, which represents the
       color of the drawn rectangle


**```thickness```**
:   An integer, which is the thickness of the
           rectangle line


**```method```**
:   A cv2 constant or integer, which determines the
        method of finding a match


returns: A tuple of 2 tuples with 2 integers in each
         ((left, top), (width, height))

    
##### Method `match_draw_rect` 




>     def match_draw_rect(
>         self,
>         image,
>         color=(0, 255, 0),
>         thickness=2,
>         method=5
>     )


Finds the top left point and dimensions (width, height)
   of a subimage that most matches the template and then
   draws a rectange with those coords.


##### Args
**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions (BGR)


**```color```**
:   A tuple of 1 or 3 integers, which represents the
       color of the drawn rectangle


**```thickness```**
:   An integer, which is the thickness of the
           rectangle line


**```method```**
:   A cv2 constant or integer, which determines the
        method of finding a match


returns: A float of the confidence

    
### Class `Windows` 




>     class Windows(
>         update_delay=1
>     )


This class is used to displays images.

Initializes the Dictionaries for holding the windows.
   (Can only have one instance per process)


##### Args
**```update_delay```**
:   An integer, which is the number of ms
              to delay each update (must be > 0)






    
#### Class variables


    
##### Variable `CREATED` 








    
#### Static methods


    
##### `Method mouse_callback_logger` 




>     def mouse_callback_logger(
>         event,
>         x,
>         y,
>         flags,
>         param
>     )


Logs all the events of a window.


##### Args
**```event```**
:   A cv2 constant or an integer


**```x```**
:   An integer, which is the horizontal position of the event


**```y```**
:   An integer, which is the vertical position of the event


**```flags```**
:   A cv2 constant or an integet


**```param```**
:   A list of additional variables




    
#### Methods


    
##### Method `add` 




>     def add(
>         self,
>         name='Image',
>         image=None,
>         mouse_callback=None
>     )


Adds an image to the update dictionary.


##### Args
**```name```**
:   A string, which is the unguaranteed name of the window.


**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions


**```mouse_callback```**
:   A function, which can be called on window events



##### Returns
A string, which is the name for the window

    
##### Method `remove` 




>     def remove(
>         self,
>         name
>     )


Removes a window from the update dictionary.


##### Args
**```name```**
:   A string, which is the unguaranteed name of the window.



    
##### Method `set` 




>     def set(
>         self,
>         name,
>         image
>     )


Sets the window to image.


##### Args
**```name```**
:   A string, which is the unguaranteed name of the window.


**```image```**
:   A numpy ndarray, which has 2 or 3 dimensions



    
##### Method `start` 




>     def start(
>         self
>     )


Starts the thread for updating.

    
##### Method `stop` 




>     def stop(
>         self
>     )


Stops the thread from updating the windows and removes.
all windows.



    
# Module `paiutils.neural_network` 

Author: Travis Hammond
Version: 12_21_2020




    
## Functions


    
### Function `conv1d` 




>     def conv1d(
>         filters,
>         kernel_size,
>         strides=1,
>         activation='relu',
>         padding='same',
>         max_pool_size=None,
>         max_pool_strides=None,
>         upsampling_size=None,
>         l1=0,
>         l2=0,
>         batch_norm=True,
>         momentum=0.99,
>         epsilon=1e-05,
>         transpose=False,
>         name=None
>     )


Creates a 1D convolution layer function.


##### Args
**```filters```**
:   An integer, which is the dimensionality of the output space


**```kernel_size```**
:   An integer or tuple of 1 integers, which is the size of
             the convoluition kernel


**```strides```**
:   An integer or tuple of 1 integers, which is stride length
         of the windows


**```activation```**
:   A string or keras/TF activation function


**```padding```**
:   A string ('same', 'valid')


**```max_pool_size```**
:   An integer, which is the size of the pooling windows


**```max_pool_strides```**
:   An integer, which is the factor to downscale by


**```upsampling_size```**
:   An integer, which is the factor to upsample by


**```l1```**
:   A float, which is the amount of L1 regularization


**```l2```**
:   A float, which is the amount of L2 regularization


**```batch_norm```**
:   A boolean, which determines if batch
            normalization is enabled


**```momentum```**
:   A float, which is the momentum for the moving
          mean and variance


**```epsilon```**
:   A float, which adds variance to avoid dividing by zero


**```transpose```**
:   A boolean, which determines if the convolution layer
           should be a deconvolution layer


**```name```**
:   A string, which is the name of the dense layer



##### Returns
A function, which takes a layer as input and returns
    a conv1d(layer)

    
### Function `conv2d` 




>     def conv2d(
>         filters,
>         kernel_size=3,
>         strides=1,
>         activation='relu',
>         padding='same',
>         max_pool_size=None,
>         max_pool_strides=None,
>         l1=0,
>         l2=0,
>         batch_norm=True,
>         momentum=0.99,
>         epsilon=1e-05,
>         upsampling_size=None,
>         transpose=False,
>         name=None
>     )


Creates a 2D convolution layer function.


##### Args
**```filters```**
:   An integer, which is the dimensionality of the output space


**```kernel_size```**
:   An integer or tuple of 2 integers, which is the size of
             the convoluition kernel


**```strides```**
:   An integer or tuple of 2 integers, which is stride length
         of the windows


**```activation```**
:   A string or keras/TF activation function


**```padding```**
:   A string ('same', 'valid')


**```max_pool_size```**
:   An integer or tuple of 2 integers, which is the size
               of the pooling windows


**```max_pool_strides```**
:   An integer or tuple of 2 integers, which is the
                  factor to downscale by


**```l1```**
:   A float, which is the amount of L1 regularization


**```l2```**
:   A float, which is the amount of L2 regularization


**```batch_norm```**
:   A boolean, which determines if batch
            normalization is enabled


**```momentum```**
:   A float, which is the momentum for the moving
          mean and variance


**```epsilon```**
:   A float, which adds variance to avoid dividing by zero


**```upsampling_size```**
:   An integer, which is the factor to upsample by


**```transpose```**
:   A boolean, which determines if the convolution layer
           should be a deconvolution layer


**```name```**
:   A string, which is the name of the dense layer



##### Returns
A function, which takes a layer as input and returns
    a conv2d(layer)

    
### Function `dense` 




>     def dense(
>         units,
>         activation='relu',
>         l1=0,
>         l2=0,
>         batch_norm=True,
>         momentum=0.99,
>         epsilon=1e-05,
>         name=None
>     )


Creates a dense layer function.


##### Args
**```units```**
:   An integer, which is the dimensionality of the output space


**```activation```**
:   A string or keras/TF activation function


**```l1```**
:   A float, which is the amount of L1 regularization


**```l2```**
:   A float, which is the amount of L2 regularization


**```batch_norm```**
:   A boolean, which determines if batch
            normalization is enabled


**```momentum```**
:   A float, which is the momentum for the moving
          mean and variance


**```epsilon```**
:   A float, which adds variance to avoid dividing by zero


**```name```**
:   A string, which is the name of the dense layer



##### Returns
A function, which takes a layer as input and returns a dense(layer)

    
### Function `inception` 




>     def inception(
>         inceptions
>     )


Creates an inception network.


##### Args
**```inceptions```**
:   A list of functions that apply layers or Tensors



##### Returns
A function, which takes a layer and returns inception(layer)


    
## Classes


    
### Class `Predictor` 




>     class Predictor(
>         path,
>         weights_name='model_weights.h5',
>         model_name='model.json',
>         custom_objects=None
>     )


Predictor is used for loading and predicting keras models.

Initializes the model and weights.


##### Args
**```path```**
:   A string, which is the path to a folder containing
      model.json, weights.h5, and maybe note.txt


**```weights_name```**
:   A string, which is the name of the weights to load


**```model_name```**
:   A string, which is the name of the model to load


**```custom_objects```**
:   A dictionary mapping to custom classes
                or functions for loading the model





    
#### Descendants

* [paiutils.autoencoder.AutoencoderPredictor](#paiutils.autoencoder.AutoencoderPredictor)
* [paiutils.gan.CycleGANPredictor](#paiutils.gan.CycleGANPredictor)
* [paiutils.gan.GANIPredictor](#paiutils.gan.GANIPredictor)
* [paiutils.gan.GANPredictor](#paiutils.gan.GANPredictor)





    
#### Methods


    
##### Method `predict` 




>     def predict(
>         self,
>         x
>     )


Predicts on a single sample.


##### Args
**```x```**
:   A ndarray or list/tuple/dict of ndarrays



##### Returns
A result from the model output

    
##### Method `predict_all` 




>     def predict_all(
>         self,
>         x,
>         batch_size=None
>     )


Predicts on many samples.


##### Args
**```x```**
:   A ndarray of model inputs



##### Returns
A result from the model output

    
### Class `Trainer` 




>     class Trainer(
>         model,
>         data
>     )


Trainer is used for loading, saving, and training keras models.

Initializes the model and the train/validation/test data.


##### Args
**```model```**
:   A compiled keras model


**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      If the train/validation/test key is present without
      the _x/_y the value will be used as a
      generator/Keras-Sequence/TF-Dataset and
      keys with _x/_y will be ignored.
      Ex. {'train_x': [...], 'train_y: [...]}
      Ex. {'train': generator(), 'test': [...]}
      Ex. {'train': tf.data.Dataset(), 'test': generator()}





    
#### Descendants

* [paiutils.autoencoder.AutoencoderTrainer](#paiutils.autoencoder.AutoencoderTrainer)
* [paiutils.gan.CycleGANTrainer](#paiutils.gan.CycleGANTrainer)
* [paiutils.gan.GANITrainer](#paiutils.gan.GANITrainer)
* [paiutils.gan.GANTrainer](#paiutils.gan.GANTrainer)


    
#### Class variables


    
##### Variable `GEN_DATA_TYPES` 









    
#### Methods


    
##### Method `eval` 




>     def eval(
>         self,
>         train_data=True,
>         validation_data=True,
>         test_data=True,
>         batch_size=None,
>         verbose=True,
>         **kwargs
>     )


Evaluates the model with the train/validation/test data.


##### Args
**```train_data```**
:   A boolean, which determines if
            train_data should be evaluated


**```validation_data```**
:   A boolean, which determines if
                 validation_data should be evaluated


**```test_data```**
:   A boolean, which determines if
           test_data should be evaluated


**```batch_size```**
:   An integer, which is the number of samples
            per graident update


**```verbose```**
:   A boolean, which determines the verbositiy level



##### Returns
A dictionary of the results

    
##### Method `load` 




>     def load(
>         self,
>         path,
>         custom_objects=None
>     )


Loads models and weights from a folder.
   (overrides the inital provided model)


##### Args
**```path```**
:   A string, which is the path to a folder
      containing model.json, model_weights.h5, note.txt, etc.


**```custom_objects```**
:   A dictionary mapping to custom classes
                or functions for loading the model



##### Returns
A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         note=None
>     )


Saves the models and weights to a new folder.


##### Args
**```path```**
:   A string, which is the path to create a folder in
      containing model.json, model_weights.h5, note.txt, etc.


**```note```**
:   A string, which is a note to save in the folder



##### Returns
A string, which is the given path + created folder

    
##### Method `set_data` 




>     def set_data(
>         self,
>         data
>     )


Sets train, validation, and test data from data.


##### Args
**```data```**
:   A dictionary containg train data
      and optionally validation and test data.
      If the train/validation/test key is present without
      the _x/_y the value will be used as a
      generator/Keras-Sequence/TF-Dataset and
      keys with _x/_y will be ignored.
      Ex. {'train_x': [...], 'train_y: [...]}
      Ex. {'train': generator(), 'test': [...]}
      Ex. {'train': tf.data.Dataset(), 'test': generator()}



    
##### Method `train` 




>     def train(
>         self,
>         epochs,
>         batch_size=None,
>         verbose=True,
>         **kwargs
>     )


Trains the keras model.


##### Args
**```epochs```**
:   An integer, which is the number of complete
        iterations to train


**```batch_size```**
:   An integer, which is the number of samples
            per graident update


**```verbose```**
:   A boolean, which determines the verbositiy level





    
# Module `paiutils.reinforcement` 

Author: Travis Hammond
Version: 12_21_2020





    
## Classes


    
### Class `Agent` 




>     class Agent(
>         action_size,
>         policy
>     )


This class is the base class for all agent classes,
and essentially is a random agent.

Initalizes the agent.


##### Args
**```action_size```**
:   An integer which is the discrete size
             of the action space


**```policy```**
:   A policy instance





    
#### Descendants

* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.QAgent](#paiutils.reinforcement.QAgent)





    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.
   (For this agent all memory is discarded)


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value, which is the action the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action


**```reward```**
:   A float, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode



    
##### Method `end_episode` 




>     def end_episode(
>         self
>     )


Ends the episode for the agent.

    
##### Method `forget` 




>     def forget(
>         self
>     )


Forgets or clears all memory.

    
##### Method `learn` 




>     def learn(
>         self,
>         verbose=True
>     )


Trains the agent on a batch of its experiences.
   (For this agent no learning is needed)


##### Args
**```verbose```**
:   A boolean, which determines if information
         should be printed to the screen



    
##### Method `load` 




>     def load(
>         self,
>         path
>     )


Loads a save from a folder.


##### Args
**```path```**
:   A string, which is the path to a folder to load



##### Returns
A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         note
>     )


Saves a note to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```note```**
:   A string, which is the note to save in the folder



##### Returns
A string, which is the complete path of the save

    
##### Method `select_action` 




>     def select_action(
>         self,
>         state,
>         training=False
>     )


Returns the action the Agent "believes" to be
   suited for the given state.


##### Args
**```state```**
:   A value or list of values, which is the
       state to get the action for


**```training```**
:   A boolean, which determines if the
          agent is training



##### Returns
A value, which is the selected action

    
##### Method `set_playing_data` 




>     def set_playing_data(
>         self
>     )


Sets the episode data.

    
### Class `AsceticPolicy` 




>     class AsceticPolicy


This class is used for calling an Agent's action function and
selecting the most ascetic action.

Initalizes the Policy.


    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Policy](#paiutils.reinforcement.Policy)






    
#### Methods


    
##### Method `select_action` 




>     def select_action(
>         self,
>         action_func,
>         training
>     )


Returns the action the Agent should take.


##### Args
**```action_func```**
:   A function that returns a list of values


**```training```**
:   A boolean, which determines if the
          Agent is in a training states



    
### Class `DDPGAgent` 




>     class DDPGAgent(
>         policy,
>         amodel,
>         cmodel,
>         discounted_rate,
>         create_memory=<function DDPGAgent.<lambda>>,
>         enable_target=False
>     )


This class (Deep Deterministic Policy Gradient Agent) is an Agent
that uses two Neural Networks. An Actor network, which is like
a PGAgent Network and a Critic Network like the DQNAgent
Network. The critic rates the actions of the actor.

Initalizes the DDPG Agent.


##### Args
**```policy```**
:   A NoisePolicy instance


**```amodel```**
:   A keras model, which takes the state as input and outputs
        actions (regularization losses are not applied,
        and compiled loss are not used)


**```cmodel```**
:   A keras model, which takes the state and a action as input
         and outputs Q Values (a judgement)


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```create_memory```**
:   A function, which returns a Memory instance


**```enable_target```**
:   A boolean, which determines if a target model
               should be used for the critic




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)


    
#### Descendants

* [paiutils.reinforcement_agents.TD3Agent](#paiutils.reinforcement_agents.TD3Agent)





    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value or list of values, which is the
        action the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action


**```reward```**
:   A float, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode



    
##### Method `learn` 




>     def learn(
>         self,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         target_update_interval=1,
>         tau=1.0,
>         verbose=True
>     )


Trains the agent on a sample of its experiences.


##### Args
**```batch_size```**
:   An integer, which is the size of each batch
            within the mini_batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled and predictions are
        repredicted)


**```target_update_interval```**
:   An integer, which is the number of
                        complete training instances
                        (repeats do not count) until the
                        target critic model weights are updated


**```tau```**
:   A float, which is the strength of the copy from the
     Actor or Critic model to the target models
     (1.0 is a hard copy and less is softer)


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
##### Method `load` 




>     def load(
>         self,
>         path,
>         load_model=True,
>         load_data=True
>     )


Loads a save from a folder.


##### Args
**```path```**
:   A string, which is the path to a folder to load


**```load_model```**
:   A boolean, which determines if the model
            architectures and weights
            should be loaded


**```load_data```**
:   A boolean, which determines if the memory
           from a folder should be loaded



##### Returns
A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         save_model=True,
>         save_data=True,
>         note='DDPGAgent Save'
>     )


Saves a note, weights of the models, and memory to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```save_model```**
:   A boolean, which determines if the model
            architectures and weights
            should be saved


**```save_data```**
:   A boolean, which determines if the memory
           should be saved


**```note```**
:   A string, which is a note to save in the folder



##### Returns
A string, which is the complete path of the save

    
##### Method `select_action` 




>     def select_action(
>         self,
>         state,
>         training=False
>     )


Returns the action the Agent "believes" to be
   suited for the given state.


##### Args
**```state```**
:   A value or list of values, which is the
       state to predict the actions for


**```training```**
:   A boolean, which determines if the
          agent is training



##### Returns
A value or list of values, which is the
    selected action

    
##### Method `set_playing_data` 




>     def set_playing_data(
>         self,
>         training=False,
>         memorizing=False,
>         learns_in_episode=False,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         target_update_interval=1,
>         tau=1.0,
>         verbose=True
>     )


Sets the playing data.


##### Args
**```training```**
:   A boolean, which determines if the agent
          should be treated as in a training mode


**```memorizing```**
:   A boolean, which determines if the agent
            should be adding the information obtained
            through playing an episode to memory


**```learns_in_episode```**
:   A boolean, which determines if the agent
                   learns during a episode or at the end


**```batch_size```**
:   An integer, which is the size of each batch
            within the mini-batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled and qvalues repredicted)


**```target_update_interval```**
:   An integer, which is the number of
                        complete training instances
                        (repeats do not count) until the
                        target critic model weights are updated


**```tau```**
:   A float, which is the strength of the copy from the
     Actor or Critic model to the target models
     (1.0 is a hard copy and less is softer)


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
##### Method `update_target` 




>     def update_target(
>         self,
>         tau
>     )


Updates the target Actor and Critic Model weights.


##### Args
**```tau```**
:   A float, which is the strength of the copy from the
     Actor or Critic model to the target models
     (1.0 is a hard copy and less is softer)



    
### Class `DQNAgent` 




>     class DQNAgent(
>         policy,
>         qmodel,
>         discounted_rate,
>         create_memory=<function DQNAgent.<lambda>>,
>         enable_target=True,
>         enable_double=False,
>         enable_per=False
>     )


This class is an Agent that uses a Deep Q Network instead of
a table like the QAgent. This allows for generalizations and
large environment states.

Initalizes the Deep Q Network Agent.


##### Args
**```policy```**
:   A policy instance


**```qmodel```**
:   A keras model, which takes the state as input and outputs
        Q Values


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```create_memory```**
:   A function, which returns a Memory instance


**```enable_target```**
:   A boolean, which determines if a target model
               should be used


**```enable_double```**
:   A boolean, which determiens if the Double Deep Q
               Network should be used


**```enable_per```**
:   A boolean, which determines if prioritized experience
            replay should be used (The implementation for this is
            not the normal tree implementation, and only weights
            the probabilily of being choosen and not also the
            gradient)




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)


    
#### Descendants

* [paiutils.reinforcement_agents.DQNPGAgent](#paiutils.reinforcement_agents.DQNPGAgent)




    
#### Static methods


    
##### `Method get_dueling_output_layer` 




>     def get_dueling_output_layer(
>         action_size,
>         dueling_type='avg'
>     )





    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value, which is the action the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action


**```reward```**
:   A float, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode



    
##### Method `learn` 




>     def learn(
>         self,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         target_update_interval=1,
>         tau=1.0,
>         verbose=True
>     )


Trains the agent on a sample of its experiences.


##### Args
**```batch_size```**
:   An integer, which is the size of each batch
            within the mini-batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled and qvalues repredicted)


**```target_update_interval```**
:   An integer, which is the number of
                        complete training instances
                        (repeats do not count) until the
                        target model weights are updated


**```tau```**
:   A float, which is the strength of the copy from the
     qmodel to the target qmodel (1.0 is a hard copy and
     less is softer)


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
##### Method `load` 




>     def load(
>         self,
>         path,
>         load_model=True,
>         load_data=True
>     )


Loads a save from a folder.


##### Args
**```path```**
:   A string, which is the path to a folder to load


**```load_model```**
:   A boolean, which determines if the model
            architecture and weights
            should be loaded


**```load_data```**
:   A boolean, which determines if the memory
           from a folder should be loaded



##### Returns
A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         save_model=True,
>         save_data=True,
>         note='DQNAgent Save'
>     )


Saves a note, model weights, and memory to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```save_model```**
:   A boolean, which determines if the model
            architecture and weights
            should be saved


**```save_data```**
:   A boolean, which determines if the memory
           should be saved


**```note```**
:   A string, which is a note to save in the folder



##### Returns
A string, which is the complete path of the save

    
##### Method `select_action` 




>     def select_action(
>         self,
>         state,
>         training=False
>     )


Returns the action the Agent "believes" to be
   suited for the given state.


##### Args
**```state```**
:   A value, which is the state to predict
       the Q values for


**```training```**
:   A boolean, which determines if the
          agent is training



##### Returns
A value, which is the selected action

    
##### Method `set_playing_data` 




>     def set_playing_data(
>         self,
>         training=False,
>         memorizing=False,
>         learns_in_episode=False,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         target_update_interval=1,
>         tau=1.0,
>         verbose=True
>     )


Sets the playing data.


##### Args
**```training```**
:   A boolean, which determines if the agent
          should be treated as in a training mode


**```memorizing```**
:   A boolean, which determines if the agent
            should be adding the information obtained
            through playing an episode to memory


**```learns_in_episode```**
:   A boolean, which determines if the agent
                   learns during a episode or at the end


**```batch_size```**
:   An integer, which is the size of each batch
            within the mini-batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled and qvalues repredicted)


**```target_update_interval```**
:   An integer, which is the number of
                        complete training instances
                        (repeats do not count) until the
                        target model weights are updated


**```tau```**
:   A float, which is the strength of the copy from the
     qmodel to the target qmodel (1.0 is a hard copy and
     less is softer)


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
##### Method `update_target` 




>     def update_target(
>         self,
>         tau
>     )


Updates the target Q Model weights.


##### Args
**```tau```**
:   A float, which is the strength of the copy from the
     qmodel to the target qmodel (1.0 is a hard copy and
     less is softer)



    
### Class `Decay` 




>     class Decay(
>         initial_value,
>         constant,
>         min_value=0,
>         step_every_call=True
>     )


This class decays a initial value to a minimum
value through a given number of steps.
(formula: max(initial_value - constant * steps, 0))

Initalizes the state of the decay object.


##### Args
**```initial_value```**
:   A float, which is the starting value to decay


**```constant```**
:   A float, which is the slope/rate that the decay occurs


**```min_value```**
:   A float, which is the minimum value the decay can reach


**```step_every_call```**
:   A boolean, which determines if each call should
                 step the decay





    
#### Descendants

* [paiutils.reinforcement.ExponentialDecay](#paiutils.reinforcement.ExponentialDecay)
* [paiutils.reinforcement.LinearDecay](#paiutils.reinforcement.LinearDecay)





    
#### Methods


    
##### Method `reset` 




>     def reset(
>         self
>     )


Resets the steps.

    
##### Method `step` 




>     def step(
>         self
>     )


Steps the decay forward.

    
### Class `ETDMemory` 




>     class ETDMemory(
>         num_time_steps,
>         void_state,
>         max_len=None
>     )


This class is for the efficient storage of time distributed states.
This type of memory should only be used for states.

Initalizes the memory.


##### Args
**```num_time_steps```**
:   An integer, which is the number of
                states that make up a complete state


**```void_state```**
:   A ndarray, which is used when there is not
            enough states to create a complete state


**```max_len```**
:   An integer, which is the max length of memory
         (if reached, the oldest memory will be removed)




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Memory](#paiutils.reinforcement.Memory)





    
#### Static methods


    
##### `Method create_shuffled_subset` 




>     def create_shuffled_subset(
>         memories,
>         subset_size,
>         weights=None
>     )


Creates a list of numpy arrays of a shuffled subset of memories.


##### Args
**```memories```**
:   A list of Memeory Objects (not asserted but assumed)


**```subset_size```**
:   A integer, which is the size of the
             outer dimension of each ndarray


**```weights```**
:   A list of probabilities that add up to 1



##### Returns
arrays and shuffled indexes


    
### Class `Environment` 




>     class Environment(
>         state_shape,
>         action_size
>     )


This class handles the environment in which the Agent
performs actions in and can get rewards from.

Initalizes state and action shapes and sets the state.


##### Args
**```state_shape```**
:   A tuple of integers, which is the
             expected state shape for the agent,
             or an integer of the discrete state
             space


**```action_size```**
:   An integer which is the discrete size
             of the action space





    
#### Descendants

* [paiutils.reinforcement.GymWrapper](#paiutils.reinforcement.GymWrapper)
* [paiutils.reinforcement.MultiSeqAgentEnvironment](#paiutils.reinforcement.MultiSeqAgentEnvironment)





    
#### Methods


    
##### Method `close` 




>     def close(
>         self
>     )


Closes any threads or loose ends of the environment.

    
##### Method `play_episode` 




>     def play_episode(
>         self,
>         agent,
>         max_steps,
>         random=False,
>         random_bounds=None,
>         render=False,
>         verbose=True
>     )


Plays a single complete episode with the agent.


##### Args
**```agent```**
:   An instance of Agent, which will be used to
       interact in the environment


**```max_steps```**
:   An integer, which is the max steps an episode
           can take before terminating the episode


**```random```**
:   A booelan, which determines if the agent should not
        be used, but instead pick random actions


**```random_bounds```**
:   A tuple of two bounds (lower and upper), which
               are used for random actions that are not onehots


**```render```**
:   A boolean, which determines if the environment should
        be rendered each step


**```verbose```**
:   A boolean, which determines if information should be
         printed to the screen



##### Returns
A tuple of an integer (last step) and a float (total reward)

    
##### Method `play_episodes` 




>     def play_episodes(
>         self,
>         agent,
>         num_episodes,
>         max_steps,
>         random=False,
>         random_bounds=None,
>         render=False,
>         verbose=True,
>         episode_verbose=None,
>         end_episode_callback=None
>     )


Plays atleast 1 complete episode with the agent.


##### Args
**```agent```**
:   An instance of Agent, which will be used to
       interact in the environment


**```num_episodes```**
:   An integer, which is the number of episodes to play


**```max_steps```**
:   An integer, which is the max steps an episode
           can take before terminating the episode


**```random```**
:   A booelan, which determines if the agent should not
        be used, but instead pick random actions


**```random_bounds```**
:   A tuple of two bounds (lower and upper), which
               are used for random actions that are not onehots


**```render```**
:   A boolean, which determines if the environment should
        be rendered each step


**```verbose```**
:   A boolean, which determines if information should be
         printed to the screen


**```episode_verbose```**
:   A boolean, which determines if single episode
                 information should be printed to the screen


**```end_episode_callback```**
:   A function called at the end of each episode
                      with episode count, steps, and total reward
                      from the most recent episode. If True
                      is returned, play_episodes will stop
                      early.



##### Returns
A float, which is the average total reward of all episodes

    
##### Method `render` 




>     def render(
>         self
>     )


Renders the environment.

    
##### Method `reset` 




>     def reset(
>         self
>     )


Resets the environment to its initialized state.


##### Returns
A numpy ndarray, which is the state

    
##### Method `step` 




>     def step(
>         self,
>         action
>     )


Moves the current state one step forward
   with regard to the action.


##### Args
**```action```**
:   An integer or value that determines an action



##### Returns
A tuple of a ndarray (state), a float/integer (reward),
    and a boolean (terminal state)

    
### Class `ExponentialDecay` 




>     class ExponentialDecay(
>         initial_value,
>         rate,
>         min_value=0,
>         step_every_call=True
>     )


This class decays a initial value to a minimum
value exponentially through a given number of steps.
(formula: inital_value * (1 - rate)^steps + min_value)

Initalizes the state of the decay object.


##### Args
**```initial_value```**
:   A float, which is the starting value to decay


**```rate```**
:   A float, which is the slope/rate that the decay occurs


**```min_value```**
:   A float, which is the minimum value the decay can reach


**```step_every_call```**
:   A boolean, which determines if each call should
                 step the decay




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Decay](#paiutils.reinforcement.Decay)






    
### Class `GreedyPolicy` 




>     class GreedyPolicy


This class is used for calling an Agent's action function and
selecting the greediest action.

Initalizes the Policy.


    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Policy](#paiutils.reinforcement.Policy)






    
#### Methods


    
##### Method `select_action` 




>     def select_action(
>         self,
>         action_func,
>         training
>     )


Returns the action the Agent should take.


##### Args
**```action_func```**
:   A function that returns a list of values


**```training```**
:   A boolean, which determines if the
          Agent is in a training states



    
### Class `GymWrapper` 




>     class GymWrapper(
>         genv
>     )


This class is a environment wrapper for OpenAI Gyms.

Initalizes state and action shapes and sets the state.


##### Args
**```genv```**
:   An OpenAI Gym




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Environment](#paiutils.reinforcement.Environment)






    
### Class `LinearDecay` 




>     class LinearDecay(
>         initial_value,
>         total_steps,
>         min_value=0,
>         step_every_call=True
>     )


This class decays a initial value to a minimum
value linearly through a given number of steps.
(formula: max(initial_value - (inital_value - min_value)
              / total_steps * steps, min_value))

Initalizes the state of the decay object.


##### Args
**```initial_value```**
:   A float, which is the starting value to decay


**```total_steps```**
:   An integer, which is the number of steps until
             min_value would be reach


**```min_value```**
:   A float, which is the minimum value the decay
           can reach


**```step_every_call```**
:   A boolean, which determines if each call should
                 step the decay




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Decay](#paiutils.reinforcement.Decay)






    
### Class `Memory` 




>     class Memory(
>         max_len=None
>     )


This class is used by agents to store episode information.
(uses a normal python list)

Initalizes the memory.


##### Args
**```max_len```**
:   An integer, which is the max length of memory
         (if reached, the oldest memory will be removed)





    
#### Descendants

* [paiutils.reinforcement.ETDMemory](#paiutils.reinforcement.ETDMemory)
* [paiutils.reinforcement.RingMemory](#paiutils.reinforcement.RingMemory)




    
#### Static methods


    
##### `Method create_shuffled_subset` 




>     def create_shuffled_subset(
>         memories,
>         subset_size,
>         weights=None
>     )


Creates a list of numpy arrays of a shuffled subset of memories.


##### Args
**```memories```**
:   A list of Memeory Objects


**```subset_size```**
:   A integer, which is the size of the
             outer dimension of each ndarray


**```weights```**
:   A list of probabilities that add up to 1



##### Returns
arrays and shuffled indexes


    
#### Methods


    
##### Method `add` 




>     def add(
>         self,
>         x
>     )


Adds a entry to memory.


##### Args
**```x```**
:   A entry similar to other entries



    
##### Method `array` 




>     def array(
>         self
>     )


Returns a copy of the memory.


##### Returns
A numpy ndarray

    
##### Method `end_episode` 




>     def end_episode(
>         self
>     )


Tells memory an episode ended.

    
##### Method `load` 




>     def load(
>         self,
>         file,
>         name
>     )


Loads a h5py dataset with the saved memory data.


##### Args
**```file```**
:   A h5py open file for reading


**```name```**
:   A string, which is the dataset name



    
##### Method `reset` 




>     def reset(
>         self
>     )


Resets or clears the memory.

    
##### Method `save` 




>     def save(
>         self,
>         file,
>         name
>     )


Creates a h5py dataset with the memory data.


##### Args
**```file```**
:   A h5py open file for writing


**```name```**
:   A string, which is the dataset name



    
### Class `MemoryAgent` 




>     class MemoryAgent(
>         action_size,
>         policy
>     )


This class is the base class for all agent that use memory.
    

Initalizes the agent.


##### Args
**```action_size```**
:   An integer which is the discrete size
             of the action space


**```policy```**
:   A policy instance




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)


    
#### Descendants

* [paiutils.reinforcement.DDPGAgent](#paiutils.reinforcement.DDPGAgent)
* [paiutils.reinforcement.DQNAgent](#paiutils.reinforcement.DQNAgent)
* [paiutils.reinforcement.PGAgent](#paiutils.reinforcement.PGAgent)





    
#### Methods


    
##### Method `load` 




>     def load(
>         self,
>         path,
>         load_data=True
>     )


Loads a save from a folder.


##### Args
**```path```**
:   A string, which is the path to a folder to load


**```load_data```**
:   A boolean, which determines if the memory
           from a folder should be loaded



Returns:
    A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         save_data=True,
>         note='MemoryAgent'
>     )


Saves a note and memory to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```save_data```**
:   A boolean, which determines if the memory
           should be saved


**```note```**
:   A string, which is a note to save in the folder



##### Returns
A string, which is the complete path of the save

    
### Class `MultiSeqAgentEnvironment` 




>     class MultiSeqAgentEnvironment(
>         state_shape,
>         action_size
>     )


This class handles the environment in which multiple agents
can perform actions against eachother in a sequential manner.

Initalizes state and action shapes and sets the state.


##### Args
**```state_shape```**
:   A tuple of integers, which is the
             expected state shape for the agent


**```action_size```**
:   An integer which is the discrete size
             of the action space




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Environment](#paiutils.reinforcement.Environment)






    
#### Methods


    
##### Method `play_episode` 




>     def play_episode(
>         self,
>         agents,
>         max_steps,
>         shuffle=True,
>         random=False,
>         random_bounds=None,
>         render=False,
>         verbose=True
>     )


Plays a single complete episode with the agents.


##### Args
**```agents```**
:   A list of Agent instances, which will be used to
        interact in the environment


**```max_steps```**
:   An integer, which is the max steps an episode
           can take before terminating the episode


**```shuffle```**
:   A boolean, which determines if the agents' positions
         should be shuffled


**```random```**
:   A booelan, which determines if the agent should not
        be used, but instead pick random actions


**```random_bounds```**
:   A tuple of two bounds (lower and upper), which
               are used for random actions that are not onehots


**```render```**
:   A boolean, which determines if the environment should
        be rendered each step


**```verbose```**
:   A boolean, which determines if information should be
         printed to the screen



##### Returns
A tuple of a list of integers (last steps)
    and a list of floats (total rewards)

    
##### Method `play_episodes` 




>     def play_episodes(
>         self,
>         agents,
>         num_episodes,
>         max_steps,
>         shuffle=True,
>         random=False,
>         random_bounds=None,
>         render=False,
>         verbose=True,
>         episode_verbose=None,
>         end_episode_callback=None
>     )


Plays at least 1 complete episode with the agents.


##### Args
**```agents```**
:   A list of Agent instances, which will be used to
        interact in the environment


**```num_episodes```**
:   An integer, which is the number of episodes to play


**```max_steps```**
:   An integer, which is the max steps an episode
           can take before terminating the episode


**```shuffle```**
:   A boolean, which determines if the agents' positions
         should be shuffled


**```random```**
:   A booelan, which determines if the agent should not
        be used, but instead pick random actions


**```random_bounds```**
:   A tuple of two bounds (lower and upper), which
               are used for random actions that are not onehots


**```render```**
:   A boolean, which determines if the environment should
        be rendered each step


**```verbose```**
:   A boolean, which determines if information should be
         printed to the screen


**```episode_verbose```**
:   A boolean, which determines if single episode
                 information should be printed to the screen


**```end_episode_callback```**
:   A function called at the end of each episode
                      with episode count, steps, and total reward
                      from the most recent episode. If True
                      is returned, play_episodes will stop
                      early.



##### Returns
A list of floats, which are the average total reward of all
    episodes for each agent

    
##### Method `reset` 




>     def reset(
>         self,
>         num_agents
>     )


Resets the environment to its initialized state.


##### Args
**```num_agents```**
:   An integer, which is the number of states needed



##### Returns
A numpy ndarray, which is the state

    
##### Method `step` 




>     def step(
>         self,
>         agent_ndx,
>         action
>     )


Moves the current state one step forward
   with regard to the agent's action.


##### Args
**```agent_ndx```**
:   An integer, which is the index of the
           agent taking a step


**```action```**
:   An integer or value that determines an action



##### Returns
A tuple of a ndarray (state), a float/integer (reward),
    and a boolean (terminal state)

    
### Class `NoisePolicy` 




>     class NoisePolicy(
>         noise_scale_decay_training,
>         noise_scale_testing,
>         action_bounds
>     )


This class is used for adding normal noise to an Agent's action.

Initalizes the Noise Policy.


##### Args
**```noise_scale_decay_training```**
:   A decay instance, which decays
                            the noise scale (a fraction of
                            action range) for the policy


**```noise_scale_testing```**
:   A float, which is the noise scale
                     of the policy when the agent is not
                     training


**```action_bounds```**
:   A tuple of two floats/integers, which are
               the lower and upper bounds of the action
               range




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Policy](#paiutils.reinforcement.Policy)


    
#### Descendants

* [paiutils.reinforcement.TemporalNoisePolicy](#paiutils.reinforcement.TemporalNoisePolicy)
* [paiutils.reinforcement.UniformNoisePolicy](#paiutils.reinforcement.UniformNoisePolicy)





    
#### Methods


    
##### Method `end_episode` 




>     def end_episode(
>         self
>     )


Tells the policy the episode ended and steps the decay.

    
##### Method `reset` 




>     def reset(
>         self
>     )


Resets decay state.

    
### Class `PGAgent` 




>     class PGAgent(
>         amodel,
>         discounted_rate,
>         create_memory=<function PGAgent.<lambda>>
>     )


This class is an Agent that uses a Neural Network like the DQN Agent,
but instead of learning to predict Q values, it predicts actions. It
learns to predict these actions through Policy Gradients (PG).

Initalizes the Policy Gradient Agent.


##### Args
**```amodel```**
:   A keras model, which takes the state as input and outputs
        actions (regularization losses are not applied,
        and compiled loss are not used)


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```create_memory```**
:   A function, which returns a Memory instance




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)


    
#### Descendants

* [paiutils.reinforcement_agents.A2CAgent](#paiutils.reinforcement_agents.A2CAgent)
* [paiutils.reinforcement_agents.DQNPGAgent](#paiutils.reinforcement_agents.DQNPGAgent)
* [paiutils.reinforcement_agents.PGCAgent](#paiutils.reinforcement_agents.PGCAgent)





    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value, which is the action the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action (discarded)


**```reward```**
:   A float, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode
          (discarded)



    
##### Method `end_episode` 




>     def end_episode(
>         self
>     )


Ends the episode and creates drewards based
on the episodes rewards.

    
##### Method `learn` 




>     def learn(
>         self,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         entropy_coef=0,
>         verbose=True
>     )


Trains the agent on a sample of its experiences.


##### Args
**```batch_size```**
:   An integer, which is the size of each batch
            within the mini_batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled)


**```entropy_coef```**
:   A float, which is the coefficent of entropy to add
              to the actor loss


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
##### Method `load` 




>     def load(
>         self,
>         path,
>         load_model=True,
>         load_data=True
>     )


Loads a save from a folder.


##### Args
**```path```**
:   A string, which is the path to a folder to load


**```load_model```**
:   A boolean, which determines if the model
            architecture and weights
            should be loaded


**```load_data```**
:   A boolean, which determines if the memory
           from a folder should be loaded



##### Returns
A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         save_model=True,
>         save_data=True,
>         note='PGAgent Save'
>     )


Saves a note, model weights, and memory to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```save_model```**
:   A boolean, which determines if the model
            architecture and weights
            should be saved


**```save_data```**
:   A boolean, which determines if the memory
           should be saved


**```note```**
:   A string, which is a note to save in the folder



##### Returns
A string, which is the complete path of the save

    
##### Method `select_action` 




>     def select_action(
>         self,
>         state,
>         training=False
>     )


Returns the action the Agent "believes" to be
   suited for the given state.


##### Args
**```state```**
:   A value, which is the state to predict
       the action for


**```training```**
:   A boolean, which determines if the
          agent is training (does nothing)



##### Returns
A value, which is the selected action

    
##### Method `set_playing_data` 




>     def set_playing_data(
>         self,
>         training=False,
>         memorizing=False,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         entropy_coef=0,
>         verbose=True
>     )


Sets the playing data.


##### Args
**```training```**
:   A boolean, which determines if the agent
          should be treated as in a training mode


**```memorizing```**
:   A boolean, which determines if the agent
            should be adding the information obtained
            through playing an episode to memory


**```batch_size```**
:   An integer, which is the size of each batch
            within the mini-batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled and qvalues repredicted)


**```entropy_coef```**
:   A float, which is the coefficent of entropy to add
              to the actor loss


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
### Class `PQAgent` 




>     class PQAgent(
>         discrete_state_space,
>         action_size,
>         policy,
>         discounted_rates,
>         learning_rates
>     )


This class is like QAgent, but it uses multiple variables at once,
hince Parallel Q Agent.

Initalizes the Q-learning agent.


##### Args
**```discrete_state_space```**
:   An integer, which


**```action_size```**
:   An integers, which is the
             action size of the environment


**```policy```**
:   A policy instance


**```discounted_rates```**
:   A list of floats within 0.0-1.0, which
                  are the rates that future rewards should
                  be counted for the current reward


**```learning_rate```**
:   A list of floats, which are the rates
               that the table is updated with the currect
               Q reward




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.QAgent](#paiutils.reinforcement.QAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)






    
#### Methods


    
##### Method `learn` 




>     def learn(
>         self,
>         learning_rate_ndx=None,
>         discounted_rate_ndx=None,
>         verbose=True
>     )


Trains the agent on its last experience.


##### Args
**```learning_rate_ndx```**
:   An integer, which is a ndx for
                   the learning rates


**```discounted_rate_ndx```**
:   An integer, which is a ndx for
                     the discounted rate


**```verbose```**
:   A boolean, which determines if information
         should be printed to the screen



    
##### Method `load` 




>     def load(
>         self,
>         path
>     )


Loads a save from a folder.


##### Args
**```path```**
:   A string, which is the path to a folder to load
      from



##### Returns
A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         note='PQAgent Save'
>     )


Saves a note and qtables to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```note```**
:   A string, which is the note to save in the folder



##### Returns
A string, which is the complete path of the save

    
##### Method `set_playing_data` 




>     def set_playing_data(
>         self,
>         training=False,
>         learning_rate_ndx=None,
>         discounted_rate_ndx=None,
>         verbose=False
>     )


Sets the playing data.


##### Args
**```training```**
:   A boolean, which determines if the agent
          should be treated as in a training mode


**```learning_rate_ndx```**
:   An integer, which is a ndx for
                   the learning rates


**```discounted_rate_ndx```**
:   An integer, which is a ndx for
                     the discounted rate


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
### Class `PlayingData` 




>     class PlayingData(
>         training,
>         memorizing,
>         epochs,
>         learns_in_episode,
>         learning_params
>     )


This class is used for containing data
that the environment needs to know, but the agent has.

Initalizes the data.


##### Args
**```training```**
:   A boolean, which determines if the agent
          should be treated as in a training mode


**```memorizing```**
:   A boolean, which determines if the agent
            should be adding the information obtained
            through playing an episode to memory


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```learns_in_episode```**
:   A boolean, which determines if the agent
                   learns during a episode or at the end


**```learning_Args```**
:   A dictionary of parameters for the agent's
                 learn method









    
### Class `Policy` 




>     class Policy


This class is used for calling an Agent's action function.

Initalizes the Policy.



    
#### Descendants

* [paiutils.reinforcement.AsceticPolicy](#paiutils.reinforcement.AsceticPolicy)
* [paiutils.reinforcement.GreedyPolicy](#paiutils.reinforcement.GreedyPolicy)
* [paiutils.reinforcement.NoisePolicy](#paiutils.reinforcement.NoisePolicy)
* [paiutils.reinforcement.StochasticPolicy](#paiutils.reinforcement.StochasticPolicy)





    
#### Methods


    
##### Method `end_episode` 




>     def end_episode(
>         self
>     )


Tells the policy the episode ended.

    
##### Method `reset` 




>     def reset(
>         self
>     )


Resets any states.

    
##### Method `select_action` 




>     def select_action(
>         self,
>         action_func,
>         training
>     )


Returns the action the Agent should take.


##### Args
**```action_func```**
:   A function that returns a value


**```training```**
:   A boolean, which determines if the
          Agent is in a training states



    
### Class `QAgent` 




>     class QAgent(
>         discrete_state_space,
>         action_size,
>         policy,
>         discounted_rate
>     )


This class is a Q-learning Agent. It does not uses a neural network,
but instead uses a table.

Initalizes the Q-learning agent.


##### Args
**```discrete_state_space```**
:   An integer, which is the size of
                      the state space


**```action_size```**
:   An integers, which is the
             action size of the environment


**```policy```**
:   A policy instance


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)


    
#### Descendants

* [paiutils.reinforcement.PQAgent](#paiutils.reinforcement.PQAgent)





    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value, which is the action the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action


**```reward```**
:   A float, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode



    
##### Method `learn` 




>     def learn(
>         self,
>         learning_rate,
>         verbose=True
>     )


Trains the agent on its last experience.


##### Args
**```learning_rate```**
:   A float, which is the rate that the table
               is updated with the currect Q reward


**```verbose```**
:   A boolean, which determines if information
         should be printed to the screen



    
##### Method `save` 




>     def save(
>         self,
>         path,
>         note='QAgent Save'
>     )


Saves a note and qtable to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```note```**
:   A string, which is the note to save in the folder



##### Returns
A string, which is the complete path of the save

    
##### Method `select_action` 




>     def select_action(
>         self,
>         state,
>         training=False
>     )


Returns the action the Agent "believes" to be
   suited for the given state.


##### Args
**```state```**
:   A value or list of values, which is the
       state to look up the action for in the table


**```training```**
:   A boolean, which determines if the
          agent is training



##### Returns
A value, which is the selected action

    
##### Method `set_playing_data` 




>     def set_playing_data(
>         self,
>         training=False,
>         learning_rate=None,
>         verbose=False
>     )


Sets the playing data.


##### Args
**```training```**
:   A boolean, which determines if the agent
          should be treated as in a training mode


**```learning_rate```**
:   A float, which is the rate that the table
               is updated with the currect Q reward
               (Must be provided if training is True)


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
### Class `RingMemory` 




>     class RingMemory(
>         max_len
>     )


This class is used by agents to store episode information.
(uses a normal python list)

Initalizes the memory.


##### Args
**```max_len```**
:   An integer, which is the max length of memory
         (if reached, the oldest memory will be removed)




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Memory](#paiutils.reinforcement.Memory)






    
### Class `StochasticPolicy` 




>     class StochasticPolicy(
>         policy,
>         stochasticity_decay_training,
>         stochasticity_testing,
>         action_size
>     )


This class is used for calling an Agent's action function.

Initalizes the Policy's states.


##### Args
**```policy```**
:   A policy instance


**```stochasticity_decay_training```**
:   A decay instance which decays
                              the stochasticity of the policy


**```stochasticity_testing```**
:   A float, which is the stochasticity
                       of the policy when the agent is not
                       training


**```action_size```**
:   An integer, which is the size of the action ndarray




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.Policy](#paiutils.reinforcement.Policy)






    
#### Methods


    
##### Method `end_episode` 




>     def end_episode(
>         self
>     )


Tells the policy the episode ended and steps the decay.

    
##### Method `reset` 




>     def reset(
>         self
>     )


Resets state of the stochasticity decay instance.

    
##### Method `select_action` 




>     def select_action(
>         self,
>         action_func,
>         training
>     )


Returns the action the Agent should take.


##### Args
**```action_func```**
:   A function that returns a list of values


**```training```**
:   A boolean, which determines if the
          Agent is in a training states



    
### Class `TemporalNoisePolicy` 




>     class TemporalNoisePolicy(
>         noise_scale_decay_training,
>         noise_scale_testing,
>         action_bounds,
>         sigma=0.3,
>         theta=0.15,
>         dt=0.01,
>         init_noise=None
>     )


This class is used for adding temporal noise to an Agent's action.

Initalizes the Temporal Noise Policy.


##### Args
**```noise_scale_decay_training```**
:   A decay instance, which decays
                            the noise scale (a fraction of
                            action range) for the policy


**```noise_scale_testing```**
:   A float, which is the noise scale
                     of the policy when the agent is not
                     training


**```action_bounds```**
:   A tuple of two floats/integers, which are
               the lower and upper bounds of the action
               range




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.NoisePolicy](#paiutils.reinforcement.NoisePolicy)
* [paiutils.reinforcement.Policy](#paiutils.reinforcement.Policy)






    
#### Methods


    
##### Method `reset` 




>     def reset(
>         self
>     )


Resets decay state and initial actions.

    
### Class `UniformNoisePolicy` 




>     class UniformNoisePolicy(
>         noise_scale_decay_training,
>         noise_scale_testing,
>         action_bounds,
>         additive=False
>     )


This class is used for adding noise to an Agent's action.

Initalizes the Uniform Noise Policy.


##### Args
**```noise_scale_decay_training```**
:   A decay instance, which decays
                            the noise scale (a fraction of
                            action range) for the policy


**```noise_scale_testing```**
:   A float, which is the noise scale
                     of the policy when the agent is not
                     training


**```action_bounds```**
:   A tuple of two floats/integers, which are
               the lower and upper bounds of the action
               range


**```additive```**
:   A boolean, which determines if the noise should be
          added or replace the action value completely




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.NoisePolicy](#paiutils.reinforcement.NoisePolicy)
* [paiutils.reinforcement.Policy](#paiutils.reinforcement.Policy)








    
# Module `paiutils.reinforcement_agents` 

Author: Travis Hammond
Version: 12_21_2020





    
## Classes


    
### Class `A2CAgent` 




>     class A2CAgent(
>         amodel,
>         cmodel,
>         discounted_rate,
>         lambda_rate=0,
>         create_memory=<function A2CAgent.<lambda>>
>     )


This class (Advantage Actor-Critic) is like the PGAgent, but it also
has a critic network which is used to estimate the value function
in order to train the Actor network on the advantages instead of
the discounted rewards.

Initalizes the Policy Gradient Agent.


##### Args
**```amodel```**
:   A keras model, which takes the state as input and outputs
        actions (regularization losses are not applied,
        and compiled loss are not used)


**```cmodel```**
:   A keras model, which takes the state as input and outputs
        the value of that state


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```lambda_rate```**
:   A float within 0.0-1.0, which if nonzero will enable
             generalized advantage estimation


**```create_memory```**
:   A function, which returns a Memory instance




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.PGAgent](#paiutils.reinforcement.PGAgent)
* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)


    
#### Descendants

* [paiutils.reinforcement_agents.A2CCAgent](#paiutils.reinforcement_agents.A2CCAgent)
* [paiutils.reinforcement_agents.PPOAgent](#paiutils.reinforcement_agents.PPOAgent)





    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value, which is the action the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action


**```reward```**
:   A float/integer, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode



    
##### Method `end_episode` 




>     def end_episode(
>         self
>     )


Ends the episode, and creates drewards based
on the episodes rewards.

    
##### Method `load` 




>     def load(
>         self,
>         path,
>         load_model=True,
>         load_data=True,
>         custom_objects=None
>     )


Loads a save from a folder.


##### Args
**```path```**
:   A string, which is the path to a folder to load


**```load_model```**
:   A boolean, which determines if the model
            architectures and weights
            should be loaded


**```load_data```**
:   A boolean, which determines if the memory
           from a folder should be loaded


**```custom_objects```**
:   A dictionary mapping to custom classes
                or functions for loading the model



##### Returns
A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         save_model=True,
>         save_data=True,
>         note='A2CAgent Save'
>     )


Saves a note, models, weights, and memory to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```save_model```**
:   A boolean, which determines if the model
            architectures and weights
            should be saved


**```save_data```**
:   A boolean, which determines if the memory
           should be saved


**```note```**
:   A string, which is a note to save in the folder



##### Returns
A string, which is the complete path of the save

    
### Class `A2CCAgent` 




>     class A2CCAgent(
>         amodel,
>         cmodel,
>         discounted_rate,
>         lambda_rate=0,
>         create_memory=<function A2CCAgent.<lambda>>
>     )


This class is a continuous action space variant of the A2CAgent.
    

Initalizes the Policy Gradient Agent.


##### Args
**```amodel```**
:   A keras model, which takes the state as input and outputs
        actions (regularization losses are not applied,
        and compiled loss are not used)


**```cmodel```**
:   A keras model, which takes the state as input and outputs
        the value of that state


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```lambda_rate```**
:   A float within 0.0-1.0, which if nonzero will enable
             generalized advantage estimation


**```create_memory```**
:   A function, which returns a Memory instance




    
#### Ancestors (in MRO)

* [paiutils.reinforcement_agents.A2CAgent](#paiutils.reinforcement_agents.A2CAgent)
* [paiutils.reinforcement.PGAgent](#paiutils.reinforcement.PGAgent)
* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)
* [paiutils.reinforcement_agents.Continuous](#paiutils.reinforcement_agents.Continuous)






    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value or list of values, which is the action
        the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action


**```reward```**
:   A float/integer, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode



    
### Class `Continuous` 




>     class Continuous


This interface is used for the continuous action space
variants of algorithms.



    
#### Descendants

* [paiutils.reinforcement_agents.A2CCAgent](#paiutils.reinforcement_agents.A2CCAgent)
* [paiutils.reinforcement_agents.PGCAgent](#paiutils.reinforcement_agents.PGCAgent)




    
#### Static methods


    
##### `Method clip` 




>     def clip(
>         lower_bound,
>         upper_bound,
>         name=None
>     )




    
##### `Method sample` 




>     def sample(
>         name=None
>     )




    
##### `Method scale` 




>     def scale(
>         lower_bound,
>         upper_bound,
>         name=None
>     )





    
### Class `DQNPGAgent` 




>     class DQNPGAgent(
>         policy,
>         qmodel,
>         amodel,
>         discounted_rate,
>         create_memory=<function DQNPGAgent.<lambda>>,
>         enable_target=True,
>         enable_double=False,
>         enable_per=False
>     )


This class is an Agent that uses a Deep Q Network by default to
select actions, but can be easily be changed to a policy gradient
based network to predict actions.

Initalizes the Deep Q Network and Policy Gradient Agent.


##### Args
**```policy```**
:   A policy instance (for DQN Agent)


**```qmodel```**
:   A keras model, which takes the state as input and outputs
        Q Values


**```amodel```**
:   A keras model, which takes the state as input and outputs
        actions (regularization losses are not be applied,
        and compiled loss are not used)


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```create_memory```**
:   A function, which returns a Memory instance


**```enable_target```**
:   A boolean, which determines if a target model
               should be used


**```enable_double```**
:   A boolean, which determiens if the Double Deep Q
               Network should be used


**```enable_per```**
:   A boolean, which determines if prioritized experience
            replay should be used
            (The implementation for this is not the normal tree
             implementation, and only weights the probabilily of
             being choosen)




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.DQNAgent](#paiutils.reinforcement.DQNAgent)
* [paiutils.reinforcement.PGAgent](#paiutils.reinforcement.PGAgent)
* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)






    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value, which is the action the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action


**```reward```**
:   A float/integer, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode



    
##### Method `end_episode` 




>     def end_episode(
>         self
>     )


Ends the episode, and creates drewards based
on the episodes rewards.

    
##### Method `learn` 




>     def learn(
>         self,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         target_update_interval=1,
>         tau=1.0,
>         verbose=True
>     )


Trains the agent on a sample of its experiences.


##### Args
**```batch_size```**
:   An integer, which is the size of each batch
            within the mini-batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled and qvalues repredicted)


**```target_update_interval```**
:   An integer, which is the number of
                        complete training instances
                        (repeats do not count) until the
                        target qmodel weights are updated


**```tau```**
:   A float, which is the strength of the copy from the
     qmodel to the target qmodel (1.0 is a hard copy and
     less is softer)


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
##### Method `load` 




>     def load(
>         self,
>         path,
>         load_model=True,
>         load_data=True,
>         custom_objects=None
>     )


Loads a save from a folder.


##### Args
**```path```**
:   A string, which is the path to a folder to load


**```load_model```**
:   A boolean, which determines if the model
            architectures and weights
            should be loaded


**```load_data```**
:   A boolean, which determines if the memory
           from a folder should be loaded


**```custom_objects```**
:   A dictionary mapping to custom classes
                or functions for loading the model



##### Returns
A string of note.txt

    
##### Method `save` 




>     def save(
>         self,
>         path,
>         save_model=True,
>         save_data=True,
>         note='DQNPGAgent Save'
>     )


Saves a note, model weights, and memory to a new folder.


##### Args
**```path```**
:   A string, which is the path to a folder to save within


**```save_model```**
:   A boolean, which determines if the model
            architectures and weights
            should be saved


**```save_data```**
:   A boolean, which determines if the memory
           should be saved


**```note```**
:   A string, which is a note to save in the folder



##### Returns
A string, which is the complete path of the save

    
##### Method `use_dqn` 




>     def use_dqn(
>         self
>     )




    
##### Method `use_pg` 




>     def use_pg(
>         self
>     )




    
### Class `PGCAgent` 




>     class PGCAgent(
>         amodel,
>         discounted_rate,
>         create_memory=<function PGCAgent.<lambda>>
>     )


This class is a continuous action space variant of the PGAgent.
    

Initalizes the Policy Gradient Agent.


##### Args
**```amodel```**
:   A keras model, which takes the state as input and outputs
        actions (regularization losses are not applied,
        and compiled loss are not used)


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```create_memory```**
:   A function, which returns a Memory instance




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.PGAgent](#paiutils.reinforcement.PGAgent)
* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)
* [paiutils.reinforcement_agents.Continuous](#paiutils.reinforcement_agents.Continuous)






    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value or list of values, which is the action
        the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action (discarded)


**```reward```**
:   A float, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode
          (discarded)



    
### Class `PPOAgent` 




>     class PPOAgent(
>         amodel,
>         cmodel,
>         discounted_rate,
>         lambda_rate=0,
>         clip_ratio=0.2,
>         create_memory=<function PPOAgent.<lambda>>
>     )


This class (Proximal Policy Optimization) is like the A2CAgent
but attempts to avoid taking large gradient steps that would
collapse the performacne of the agent. (this is the clip variant)

Initalizes the Policy Gradient Agent.


##### Args
**```amodel```**
:   A keras model, which takes the state as input and outputs
        actions (regularization losses are not applied,
        and compiled loss are not used)


**```cmodel```**
:   A keras model, which takes the state as input and outputs
        the value of that state


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```lambda_rate```**
:   A float within 0.0-1.0, which if nonzero will enable
             generalized advantage estimation


**```clip_ratio```**
:   A float, which is the ratio to clip the differences
            between new and old action probabilities


**```create_memory```**
:   A function, which returns a Memory instance




    
#### Ancestors (in MRO)

* [paiutils.reinforcement_agents.A2CAgent](#paiutils.reinforcement_agents.A2CAgent)
* [paiutils.reinforcement.PGAgent](#paiutils.reinforcement.PGAgent)
* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)






    
#### Methods


    
##### Method `add_memory` 




>     def add_memory(
>         self,
>         state,
>         action,
>         new_state,
>         reward,
>         terminal
>     )


Adds information from one step in the environment to the agent.


##### Args
**```state```**
:   A value or list of values, which is the
       state of the environment before the
       action was performed


**```action```**
:   A value, which is the action the agent took


**```new_state```**
:   A value or list of values, which is the
           state of the environment after performing
           the action (discarded)


**```reward```**
:   A float/integer, which is the evaluation of
        the action performed


**```terminal```**
:   A boolean, which determines if this call to
          add memory is the last for the episode
          (discarded)



    
##### Method `select_action` 




>     def select_action(
>         self,
>         state,
>         training=False
>     )


Returns the action the Agent "believes" to be
   suited for the given state.


##### Args
**```state```**
:   A value, which is the state to predict
       the action for


**```training```**
:   A boolean, which determines if the
          agent is training



##### Returns
A value, which is the selected action

    
### Class `TD3Agent` 




>     class TD3Agent(
>         policy,
>         amodel,
>         cmodel,
>         discounted_rate,
>         create_memory=<function TD3Agent.<lambda>>
>     )


This class (Twin Delayed DDPG Agent) attempts to mitigate
the problems that a DDPGAgent faces through clipping Q targets
between two Q models, delaying policy updates, and adding noise
to target actions.

Initalizes the DDPG Agent.


##### Args
**```policy```**
:   A noise policy instance, which used for exploring


**```amodel```**
:   A keras model, which takes the state as input and outputs
        actions (regularization losses are not applied,
        and compiled loss are not used)


**```cmodel```**
:   A keras model, which takes the state and a action as input
         and outputs two seperate Q Values (a judgement)


**```discounted_rate```**
:   A float within 0.0-1.0, which is the rate that
                 future rewards should be counted for the current
                 reward


**```create_memory```**
:   A function, which returns a Memory instance




    
#### Ancestors (in MRO)

* [paiutils.reinforcement.DDPGAgent](#paiutils.reinforcement.DDPGAgent)
* [paiutils.reinforcement.MemoryAgent](#paiutils.reinforcement.MemoryAgent)
* [paiutils.reinforcement.Agent](#paiutils.reinforcement.Agent)






    
#### Methods


    
##### Method `learn` 




>     def learn(
>         self,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         target_update_interval=1,
>         tau=1.0,
>         policy_noise_std=0.2,
>         policy_noise_clip=0.5,
>         actor_update_infreq=2,
>         verbose=True
>     )


Trains the agent on a sample of its experiences.


##### Args
**```batch_size```**
:   An integer, which is the size of each batch
            within the mini_batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled and predictions are
        repredicted)


**```target_update_interval```**
:   An integer, which is the number of
                        complete training instances
                        (repeats do not count) until the
                        target critic model weights are updated


**```tau```**
:   A float, which is the strength of the copy from the
     Actor or Critic model to the target models
     (1.0 is a hard copy and less is softer)


**```policy_noise_std```**
:   A float, which is the standard deviation of the
                  noise to add to the target actions for gradient
                  steps


**```policy_noise_clip```**
:   A float, which is the min and max value of
                   the normal noise added to target actions
                   for gradient steps


**```actor_update_infreq```**
:   An integer, which is the infrequency that
                     the actor is updated compared to the critic


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)



    
##### Method `set_playing_data` 




>     def set_playing_data(
>         self,
>         training=False,
>         memorizing=False,
>         learns_in_episode=False,
>         batch_size=None,
>         mini_batch=0,
>         epochs=1,
>         repeat=1,
>         target_update_interval=1,
>         tau=1.0,
>         policy_noise_std=0.2,
>         policy_noise_clip=0.5,
>         actor_update_infreq=2,
>         verbose=True
>     )


Sets the playing data.


##### Args
**```training```**
:   A boolean, which determines if the agent
          should be treated as in a training mode


**```memorizing```**
:   A boolean, which determines if the agent
            should be adding the information obtained
            through playing an episode to memory


**```learns_in_episode```**
:   A boolean, which determines if the agent
                   learns during a episode or at the end


**```batch_size```**
:   An integer, which is the size of each batch
            within the mini-batch during one training instance


**```mini_batch```**
:   An integer, which is the entire batch size for
            one training instance


**```epochs```**
:   An integer, which is the number of epochs to train
        in one training instance


**```repeat```**
:   An integer, which is the times to repeat a training
        instance in one training instance (similar to epochs,
        but mini_batch is resampled and qvalues repredicted)


**```target_update_interval```**
:   An integer, which is the number of
                        complete training instances
                        (repeats do not count) until the
                        target critic model weights are updated


**```tau```**
:   A float, which is the strength of the copy from the
     Actor or Critic model to the target models
     (1.0 is a hard copy and less is softer)


**```policy_noise_std```**
:   A float, which is the standard deviation of the
                  noise to add to the target actions for gradient
                  steps


**```policy_noise_clip```**
:   A float, which is the min and max value of
                   the normal noise added to target actions
                   for gradient steps


**```actor_update_infreq```**
:   An integer, which is the infrequency that
                   the actor is updated compared to the critic


**```verbose```**
:   A boolean, which determines if training
         should be verbose (print information to the screen)





    
# Module `paiutils.util_funcs` 

Author: Travis Hammond
Version: 12_21_2020




    
## Functions


    
### Function `load_datasets` 




>     def load_datasets(
>         path,
>         file_loader
>     )




    
### Function `load_directory_database` 




>     def load_directory_database(
>         path,
>         file_loaders
>     )


Loads groups of datasets from a directory.
        Directory
         //   \
        //     \
       //       \
    train_x     train_y
     //           \
   // \         // \
 dogs  cats    dogs  cats
  ||    ||      ||    ||
files  files  files  files <- Groups should share file names


##### Args
**```path```**
:   A string, which is a path to the datasets


**```file_loaders```**
:   A dictionary of postfix identifiers as keys
              and file loader functions for values



##### Returns
A dictionary, which contains the dataset groups

    
### Function `load_directory_datasets` 




>     def load_directory_datasets(
>         path,
>         file_loader
>     )


Loads datasets from a directory.
        Directory
         //   \
        //     \
       //       \
     dogs       cats
     //           \
data files     data files


##### Args
**```path```**
:   A string, which is a path to the datasets


**```file_loader```**
:   A function for loading each file, or
             a dictionary of postfix identifiers as keys
             and file loader functions for values



##### Returns
A dictionary, which contains the datasets

    
### Function `load_file_datasets` 




>     def load_file_datasets(
>         path
>     )


Loads datasets from a file.


##### Args
**```path```**
:   A string, which is a path to the datasets



##### Returns
A dictionary, which contains the datasets

    
### Function `read` 




>     def read(
>         path,
>         **kwargs
>     )


Loads a mapping file.


##### Args
**```path```**
:   A string from which to load the mappings.csv file



##### Returns
A list of dictionaries

    
### Function `save_datasets` 




>     def save_datasets(
>         path,
>         datasets,
>         file_saver
>     )




    
### Function `save_directory_database` 




>     def save_directory_database(
>         path,
>         database,
>         file_savers
>     )


Saves groups of datasets from a directory.
        Directory
         //   \
        //     \
       //       \
    train_x     train_y
     //           \
   // \         // \
 dogs  cats    dogs  cats
  ||    ||      ||    ||
files  files  files  files <- Groups should share order


##### Args
**```path```**
:   A string, which is a path to the datasets


**```database```**
:   A dictionary, which contains the groups of datasets


**```file_loaders```**
:   A dictionary of postfix identifiers as keys
              and file loader functions for values



##### Returns
A dictionary, which contains the dataset groups

    
### Function `save_directory_datasets` 




>     def save_directory_datasets(
>         path,
>         datasets,
>         file_saver
>     )


Saves datasets to a directory.
        Directory
         //   \
        //     \
       //       \
     dogs       cats
     //           \
data files     data files


##### Args
**```path```**
:   A string, which is a path to the datasets


**```datasets```**
:   A dictionary, which contains the datasets


**```file_saver```**
:   A function for saving each file, or a
            dictionary of postfix identifiers as keys
            and file loader functions for values



    
### Function `save_file_datasets` 




>     def save_file_datasets(
>         path,
>         datasets
>     )


Saves datasets to a file.


##### Args
**```path```**
:   A string, which is a path to the datasets


**```datasets```**
:   A dictionary, which contains the datasets



    
### Function `write` 




>     def write(
>         mappings,
>         path,
>         **kwargs
>     )


Creates or appends a mappings csv file.


##### Args
**```mappings```**
:   A dictionary


**```path```**
:   A string for the path to save the csv file to





-----
Generated by *pdoc* 0.9.2 (<https://pdoc3.github.io>).
