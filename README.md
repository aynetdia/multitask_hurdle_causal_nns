# Multitask Hurdle Causal Neural Networks
Submission of the seminar paper (jupyter notebook) for the Applied Predictive Analytics seminar

Authored by Ansar Aynetdinov and Duygu Ider

Since we were working on this project in Google Colab, the `dir_path` specified in the third code cell of the notebook and leading to the `apa` folder needs to be changed accordingly, depeding on where this repo is accessed from.

Folder structure:


    .
    ├── apa                                  # All files needed for running the notebook as is
    │   ├── daw_tune                         
    │       ├── models                       # Saved model checkpoints of the best models tuned with the 
                                             # Dynamic Weight Average approach
    │       ├── training_history             # CSV files with training histories for all hyperparameter combinations
    │       └── daw_tune_results.npy         # nparray with training results of the Dynamic Weight Average approach
    │   ├── hyperparam_tune2                 # Contains files relevant for MTNet, MTXNet and MTCat
    │       └── ...                          # Similar to daw_tune
    │   ├── residual_model                   # Contains files relevant for the residual models
    │       └── ...                          # Similar to daw_tune
    │   ├── separate_lr_tune                 # Contains files relevant for MTNets trained with multiple learning rates
    │       └── ...                          # Similar to daw_tune
    │   ├── simple_nn_tune                   
    │       ├── checkout_nn                  # Saved CheckoutNN model checkpoints
    │       ├── conversion_nn                # Saved ConversionNN model checkpoints
    │       ├── checkout_nn_results.npy      # nparray with CheckoutNN training results
    │       └── conversion_nn_results.npy    # nparray with ConversionNN training results
    │   └── helper_functions.py              # PyTorch Dataloaders and functions for calculation of the analytical
    │                                        # targeting policy and the transformed outcome loss written by Johannes Haupt
    ├── group4_submission.ipynb              # Main Notebook
    └── README.md                            # Readme file
