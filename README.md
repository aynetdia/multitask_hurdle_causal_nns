# Multitask Hurdle Causal Neural Networks
Submission of the seminar paper (jupyter notebook) for the Applied Predictive Analytics seminar

The project's main goal was to implement a multitask hurdle causal network that would be able to
handle classical hurdle models while dealing with the causal aspect in the data.

Folder structure:

    .
    ├── scripts                              # Implementations of the NN models, tuning func, etc.
    │   ├── baseline_nns.py                  # Baseline simple NNs       
    │   ├── dataloaders.py                   # PyTorch Dataloaders
    │   ├── helper_funcs.py                  # Functions for calculation of the analytical targeting policy and the transformed
    │   │                                    # outcome loss, seed function
    │   ├── mt_residual_nns.py               # Multitask Hurdle Residual NNs
    │   ├── mt_residual_nns.py               # Multitask Hurdle NNs   
    │   └── tuning.py                        # Functions for tuning hyperparams of the models
    │
    ├── tuning_results                       # All files needed for running the notebook as is
    │   ├── daw_tune                         
    │   │   ├── models                       # Saved model checkpoints of the best models tuned with the 
    │   │   │                                # Dynamic Weight Average approach
    │   │   ├── training_history             # CSV files with training histories for all hyperparameter combinations
    │   │   └── daw_tune_results.npy         # nparray with training results of the Dynamic Weight Average approach
    │   ├── hyperparam_tune                  # Contains files relevant for MTNet, MTXNet and MTCat
    │   │   └── ...                          # Similar to daw_tune
    │   ├── residual_model                   # Contains files relevant for the residual models
    │   │   └── ...                          # Similar to daw_tune
    │   ├── separate_lr_tune                 # Contains files relevant for MTNets trained with multiple learning rates
    │   │   └── ...                          # Similar to daw_tune
    │   └── simple_nn_tune                   
    │       ├── checkout_nn                  # Saved CheckoutNN model checkpoints
    │       ├── conversion_nn                # Saved ConversionNN model checkpoints
    │       ├── checkout_nn_results.npy      # nparray with CheckoutNN training results
    │       └── conversion_nn_results.npy    # nparray with ConversionNN training results
    │
    ├── data                                 # Dataset Folder
    │
    ├── submission.ipynb                     # Main Notebook detailing the project and its results
    └── README.md                            # Readme file
