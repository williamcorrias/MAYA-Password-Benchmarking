# MAYA: Addressing Inconsistencies in Generative Password Guessing through a Unified Benchmark

This repository contains the official implementation of the paper [*MAYA: Addressing Inconsistencies in Generative Password Guessing through a Unified Benchmark*](https://www.arxiv.org/abs/2504.16651).

## Authors & Contacts

- **William Corrias**: | [ORCID](https://orcid.org/0009-0006-2270-9266) | [corrias@di.uniroma1.it](mailto:corrias@di.uniroma1.it) - Sapienza University of Rome. 
- **Fabio De Gaspari**: | [ORCID](https://orcid.org/0000-0001-9718-1044) | [degaspari@di.uniroma1.it](mailto:degaspari@di.uniroma1.it) - Sapienza University of Rome. 
- **Dorjan Hitaj**: | [ORCID](https://orcid.org/0000-0001-5686-3831) | [hitaj.d@di.uniroma1.it](mailto:hitaj.d@di.uniroma1.it) - Sapienza University of Rome. 
- **Luigi V. Mancini**: | [ORCID](https://orcid.org/0000-0003-4859-2191) | [mancini@di.uniroma1.it](mailto:mancini@di.uniroma1.it) - Sapienza University of Rome. 

## Citation

```
@article{corrias2025maya,
  title={MAYA: Addressing Inconsistencies in Generative Password Guessing through a Unified Benchmark},
  author={Corrias, William and De Gaspari, Fabio and Hitaj, Dorjan and Mancini, Luigi V},
  journal={arXiv preprint arXiv:2504.16651},
  year={2025}
}
```

## Overview

MAYA is a unified, customizable, plug-and-play benchmarking framework designed to facilitate the systematic characterization and benchmarking of generative password-guessing models in the context of trawling attacks. 

## Getting Started

### Pre-Requisites

Make sure you have Python 3.8 or newer installed. Earlier versions might not be compatible.

### Dependencies 

All necessary packages are listed in the requirements.txt file in the root directory.

```
accelerate
argparse
gdown
datasets
py7zr
pyyaml
pyzstd
rarfile
matplotlib
numpy
numpy-ringbuffer
torch
torchvision
torchaudio
tqdm
transformers
seaborn
scipy
scikit-learn
```

To install all of them at once, simply run:

```
pip install -r requirements.txt
```

Alternatively, you can install individual packages as needed.

### Reproducing Our Environment

For full compatibility and to reproduce the exact environment used in our experiments, use the requirements_versioned.txt file:

```
pip install -r requirements_versioned.txt
```

### Datasets

This repository doesn't directly include the datasets. Instead, the framework automatically downloads them as needed using the script/utils/download_raw_data.py script.

If a dataset is missing when you run an experiment, the system will try to download it for you. However, you can also trigger the download manually.

#### Downloading Datasets

To download the datasets, navigate to the project's root directory and run the following script:

```
python3 script/utils/download_raw_data.py
```

Running the script without any arguments will automatically download all available datasets and place them into their respective subdirectories within the datasets/ folder.

#### Available Datasets

* Rockyou
* myspace
* phpbb
* LinkedIn
* Hotmail
* Mail.ru
* Yandex
* Yahoo
* Faith Writer
* Hak5
* Zomato
* 000webhost
* Singles.org
* Taobao
* Gmail
* Mate1.com
* Twitter
* Ashley Madison
* Libero

#### Downloading Specific Datasets

If you only need certain datasets, use the --datasets option followed by a space-separated list of dataset names. For example:

```
python3 script/utils/download_raw_data.py --datasets rockyou libero gmail
```

## Quick Testing

To quickly reproduce the experiments presented in the paper, you can use the predefined configuration files located in the config/test/ directory. Each file corresponds to a specific research question (RQ) scenario.

### How to Run a Predefined Experiment

Use the following command to run a predefined experiment:

```
python3 main.py --test_config config/test/rqX.yaml
```

Replace X with the number of the research question you want to reproduce (e.g., rq1, rq2, rq3, rq4.1, rq4.2, rq5.1, rq5.2, rq5.3, rq6.1-jaccard, rq6.1-mergeability, rq6.2, rq7.2, rq7.3).

Each configuration file contains all the necessary information to reproduce a specific experimental setting. 
    
You can also use these files as templates to create your own custom experiments.

### Automatic Plotting

Each .yaml scenario file is automatically linked to a dedicated Python script that:

- Generates the corresponding plot or table based on the results.
- Saves the output in the figures/ directory.

You do not need to manually call the plotting script, it is triggered automatically once the experiment completes!

## Parameters

MAYA offers a modular and flexible configuration system. You can control experiments and various settings using a wide range of parameters.

### How to Configure Parameters

Parameters can be passed in two ways:

- YAML configuration files (defined by --test_config)
- Command-line arguments (CLI), which override YAML values.

It's mandatory to provide a valid --test_config YAML file. This file is crucial as it defines the core setup for your experiment, including:

- The preprocessing functions to use (e.g., filtering, splitting, chunking).
- The specific inputs for each preprocessing function.
- The type and scope of each parameter (e.g., general_params, split_params, post_split_params, etc.)
- The Python script responsible for generating the corresponding plot or figure from the results.

CLI arguments are primarily used to override values defined in the --test_config file. This is particularly useful for quickly testing different settings or making minor adjustments without modifying the YAML files directly.

**NOTE**: If a parameter is defined in both the YAML file and as a CLI argument, the CLI argument will always take precedence and override the value in the YAML configuration.

### Available Parameters

```
Usage: ./main.py  
            --test_config PATH  
            [--models STR [STR ...]] 
            [--train_datasets STR [STR ...]] 
            [--test_datasets STR [STR ...]] 
            [--n_samples INT [INT ...]] 
            [--max_length INT [INT ...]] 
            [--display_logs {0,1}] 
            [--autoload {0,1}] 
            [--overwrite {0,1}] 
            [--save_guesses {0,1] Default: 1
            [--save_matches {0,1] Default: 1
            [--path_to_checkpoint PATH] 
            [--char_bag STR [STR ...]] 
            [--train_split_percentage INT [INT ...]] 
            [--train_chunk_percentage INT [INT ...]] 
            [--test_frequency INT [INT ...]] 
            [--guesses_file PATH [PATH ...]] 
            [--sub_samples_from_file PATH [PATH ...]] 
            [--data_to_embed PATH [PATH ...]] 
```

Many parameters support multiple values. When you pass multiple values (e.g., --models passgan passflow), the program will automatically generate all valid combinations and run each experiment sequentially.

A special note on --n_samples: The model will generate only the maximum number of passwords requested. The evaluation is then performed independently for each --n_samples threshold by slicing the generated output. In short, password generation happens once, but results are sliced for different evaluation thresholds.

Here's a detailed explanation of each parameter:

- **--test_config PATH**: Path to a test configuration file (YAML). The file must follow the expected schema (see Scenarios section).
- **--models STR [STR ...]**: One or more models to run. Each model name must match a key defined in config/model/model_settings.yaml.
- **--train_datasets STR [STR ...]**: One or more dataset names to be used for training.
- **--test_datasets STR [STR ...]**: One or more dataset names to be used for testing. Do not pass this option if you intend to use the same dataset for both training and testing.
- **--n_samples INT [INT ...]**: Thresholds for evaluation. The model will generate the largest value, then slice the output for the others.
- **--max_length INT [INT ...]**: Maximum password length. 
- **--display_logs {0,1}**: Flag. Show logs in the console if set. Otherwise, logs are redirected to the logs/ directory. 
- **--autoload {0,1}**: Flag. Automatically loads the latest available checkpoint (checkpointX.pt, highest X). Use only if you’re not specifying --path_to_checkpoint.
- **--overwrite {0,1}**: Flag. If set, reruns tests even if results already exist.
- **--save_guesses {0,1}**: Flag. If set to 1, all generated passwords will be saved to disk after sampling. Default: 1.
- **--save_matches {0,1}**: Flag. If set to 1, all successfully guessed passwords (i.e., those matching the test set) will be saved. Default: 1.
- **--path_to_checkpoint PATH**: Manually specify a model checkpoint file to load.
- **--char_bag STR [STR ...]**: One or more character sets to use.
- **--train_split_percentage INT [INT ...]**: Percentage(s) of the dataset to be used for training.
- **--train_chunk_percentage INT [INT ...]**: Portion of the training dataset to use. Values ≤ 100 → treated as percentages. Values > 100 → treated as absolute counts.
- **--test_frequency INT [INT ...]**: Filter test set by password frequency. Positive values → top % most frequent. Negative values → bottom %.
- **--guesses_file PATH [PATH ...]**: Use pre-generated password files instead of generating new ones.
- **--sub_samples_from_file PATH [PATH ...]**: Subsample passwords from given file(s), according to the values in --n_samples.
- **--data_to_embed PATH [PATH ...]**: Path(s) to the data to be embedded using PassFlow’s encoder. Requires a valid PassFlow checkpoint.

### CLI Usage Examples

CLI arguments are specifically designed to override values defined in your --test_config file. This is highly useful for running quick tests or experimenting with different settings without the need to modify your YAML configuration files.

Here are some examples illustrating how to use CLI arguments to customize predefined scenarios:

#### Example 1 – Modify RQ1: Run PassGAN and PassFlow on RockYou, generating 10M, 100M, and 500M passwords, with maximum password lengths of 10 and 12.

```
python3 ./main.py --test_config config/test/rq1.yaml --models passgan passflow --train_datasets rockyou --n_samples 10000000 100000000 500000000 --max_length 10 12 --autoload 1
```

#### Example 1 – Modify RQ2. Running only PassGPT. #### 
```
python3 ./main.py --test_config config/test/rq2.yaml --models passgpt
```

#### Example 2 - Modify RQ3. Train PLR-GAN on 1M, 2M samples and 50% of LinkedIn. Generate 10M passwords, evaluate also at 1M, and fix max length to 12: #### 
```
python3 ./main.py --test_config config/test/rq3.yaml --models plrgan --train_datasets linkedin --train_chunk_percentage 50 1000000 2000000 --n_samples 10000000 1000000 --max_length 12 --autoload 1
```

#### Example 3 - Modify RQ4.1. Train PLR-GAN on RockYou and test on LinkedIn. Generate 500M passwords and fix max length to 8: #### 
```
python3 ./main.py --test_config config/test/rq4.1.yaml --models plrgan --train_datasets rockyou --test_datasets linkedin --n_samples 500000000 --max_length 8 --autoload 1
```

#### Example 4 - Modify RQ5.1. Train PassGPT on Libero and 000webhost, test on the top 5%, 10%, and bottom 90% frequent passwords. Generate 500M passwords, max length 12: #### 
```
python3 ./main.py --test_config config/test/rq5.1.yaml --models passgpt --train_datasets 000webhost libero --test_frequency 5 10 -90 --n_samples 500000000 --max_length 12 --autoload 1
```

#### Example 5 - Modify RQ6.1-jaccard. Compute the Jaccard index only between PassGAN, PassFlow, and FLA: #### 
```
python3 ./main.py --test_config config/test/rq6.1-jaccard.yaml --models passgan passflow fla 
```

#### Example 6 - Modify RQ6.2. Perform a multi-model attack using PassGAN, PassFlow, FLA, and PassGPT, evaluating only on RockYou, on maximum password length 8 and 10: #### 
```
python3 ./main.py --test_config config/test/rq6.2.yaml --models passgan passflow fla passgpt --test_datasets rockyou --max_length 8 10
```

### YAML Configuration Files

YAML configuration files are central to defining and managing evaluation scenarios within MAYA. These files allow you to create reproducible experimental setups by specifying:

- Preprocessing functions: Which functions to execute, in what order, and with what inputs. 
- Parameters: All the necessary parameters for the experiment and their types.

These files are passed to the program using the --test_config option and are essential for reproducing the evaluation setups reported in our paper.

MAYA distinguishes between five types of parameters, categorized by when and how they impact the pipeline:

1) **general_params**: These are general-purpose parameters that are independent of both training and test datasets: models, display_logs, autoload, overwrite, path_to_checkpoint, n_samples, test_config, guesses_file, sub_samples_from_file, and data_to_embed.
2) **pre_split_params**: Parameters applied before the dataset is split, thus affecting both the training and test sets: max_length, char_bag, and train_datasets.
3) **split_params**: These control the logic of dataset splitting and influence both the training and test sets, such as train_split_percentage.
4) **post_split_params**: Applied after the data has been split, still may impact both sets. An example is train_chunk_percentage, which in this case only impacts the training.
5) **test_params**: These parameters affect only the test dataset and are ignored during training. Examples include test_datasets and test_frequency.

Scenario-specific configuration files are stored in the config/test/ directory. You can specify which file to use by passing its path via the --test_config parameter:

```
python3 main.py --test_config config/test/rq1.yaml
```

#### Example: Training and Testing Scenario (RQ1, RQ3-RQ5.1)

Here's an example of the rq1.yaml configuration file, explained with comments. This structure applies to all scenarios that involve both training and testing phases (e.g., RQ1, RQ3, RQ4.1, RQ4.2, and RQ5.1).

```
# Name of the test scenario
rq1:

    # Path to the script used to generate the final plot based on the results.
    figure_script: script/plotters/rq1.py

    # General parameters that do not require specific pre-processing functions.
    general_params:

        # Automatically load the latest available model checkpoint (1 for true, 0 for false).
        autoload: 1

        # List of models to be included in this evaluation scenario.
        models:
          - passgan
          - plrgan
          - passflow
          - passgpt
          - vgpt2
          - fla

        # Thresholds for evaluation (number of generated passwords to consider).
        n_samples:
          - 1000000
          - 2500000
          - 5000000
          - 7500000
          - 10000000
          - 25000000
          - 50000000
          - 75000000
          - 100000000
          - 250000000
          - 500000000

    # From here, you must specify the preprocessing functions to be executed.

    pre_split_params:

        # Format: filename.function_name
        # The line below calls the `read_train_passwords()` function
        # from `script/dataset/preprocessing/standard_preprocessing.py`.
        standard_preprocessing.read_train_passwords:
            # Arguments for the function
            train_datasets:
              - rockyou
              - 000webhost
              - linkedin
              - mailru
              - libero
              - ashleymadison
              - taobao
              - gmail

        # Filters passwords by their maximum length.
        standard_preprocessing.filter_by_length:
            # Arguments for the function
            max_length:
              - 8
              - 10
              - 12

        # Filters passwords based on a defined character set.
        standard_preprocessing.filter_by_char_bag:
            # Arguments for the function
            char_bag:
              - "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `"

    # Parameters controlling the dataset splitting logic.
    split_params:

        # Performs a test-centric split of the dataset.
        standard_preprocessing.test_centric_split:
            # Arguments for the function
            train_split_percentage:
              - 80

    # Parameters applied after the dataset has been split.
    post_split_params:

        # Chunks the training dataset.
        standard_preprocessing.chunk_train_dataset:
            # Arguments for the function
            train_chunk_percentage:
              - 100
              
```

#### Example: Evaluation-Only Scenario (RQ2, RQ5.2-RQ7.3)

When a scenario only computes evaluation metrics on previously generated passwords—without retraining or regenerating—the configuration files follow a slightly different format. This applies to scenarios like RQ2, RQ5.2, RQ5.3, RQ6.1, RQ6.2, RQ7.2, and RQ7.3 (which reuse passwords generated in RQ1).

Here's an example for rq5.2.yaml:

```
rq5.2:

    # Script that performs the evaluation by computing the desired statistics.
    evaluation_script: script/test/rqs/match_per_length.py

    # Path to the script used to generate the final plot.
    figure_script: script/plotters/rq5.2.py

    # Specifies which previous scenario's generated passwords to use as reference.
    test: rq1

    # Below, list all combinations for which the statistic will be computed.
    # Note: In evaluation-only scenarios, there is no need to define preprocessing functions
    # or separate parameter categories like `general_params`, `pre_split_params`, etc.
    
    overwrite: 0
    display_logs: 0

    models:
      - passgan
      - plrgan
      - passflow
      - passgpt
      - vgpt2
      - fla

    train_datasets:
        - rockyou
        - linkedin
        - 000webhost
        - gmail
        - libero
        - ashleymadison
        - taobao
        - mailru

    char_bag:
        - "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `"

    max_length:
        - 12

    train_chunk_percentage:
        - 100

    train_split_percentage:
        - 80

    n_samples:
        - 500000000
```

## Customization
MAYA is designed to be customizable, providing an interface that allows you to easily extend its capabilities. You can integrate your own:

- Models: Add new generative password-guessing models.
- Datasets: Incorporate custom datasets for benchmarking.
- Scenarios: Define entirely new evaluation scenarios tailored to your specific research needs.

### How To Add a Custom Dataset

To integrate your own dataset into MAYA, simply follow these steps:

1) **Insert the `.pickle` File**  
   Save your dataset as a .pickle file, ensuring it contains only passwords, with each password separated by a newline character (\n). Place this file into the appropriate subdirectory within the datasets/ folder, organized by its language and service type.  Example Path: datasets/en/social-net/DatasetX.pickle

3) **Register the Dataset**  
   Add an entry inside the dict_datasets dictionary in `script/utils/download_raw_data.py`.  
   The only required parameters are:
   - `filename`: The name of the file (without the `.pickle` extension)
   - `language`: The language subdirectory (e.g., `"en"`)
   - `service`: The service-type subdirectory (e.g., `"social-net"`)

   **Example:** You want to add a dataset named `DatasetX`, which is in English and represents a social network:
   ```
   datasetx : {
       "filename": "DatasetX",
       "ext": "",
       "format": "",
       "language": "en",
       "service": "social-net",
   },

### How To Add an Evaluation Scenario

To define a new evaluation scenario, simply create a .yaml configuration file within the config/test/ directory. You can use any of the existing files in that folder as a helpful reference.

Each configuration file must adhere to the following structure:
1. The name of your test should be defined as the top-level key.
2. Parameters must be grouped according to their type (e.g., pre_split_params, split_params, post_split_params).
3. The path to the preprocessing function needs to be specified using the format: filename.function_name. Where filename is a file inside the directory script/dataset/preprocessing/.

### Adding or Modifying Preprocessing Functions

If you need to add new preprocessing functions or modify existing ones, you have two options:

- Modify the existing `standard_preprocessing.py` file located at `script/dataset/preprocessing/standard_preprocessing.py` (**not recommended**), or  
- Create a new file inside `script/dataset/preprocessing/` containing your custom preprocessing functions.

Example: If you create a new file named personalized_preprocessing.py containing your functions, you would reference them in your YAML config file like this:

```
    ...
        personalized_preprocessing.your_custom_function:
            parameter1:
              - value1
              
            ...
    ...
```

### How To Add a Model
If you desire to integrate a custom model into MAYA, follow these steps:

1. **Subclass the Base Model**
   Your custom model must inherit from the Model base class defined in script/test/model.py.
   You are required to implement the following abstract methods:
   - prepare_data 
   - load 
   - train 
   - eval_init 
   - sample 
   - guessing_strategy 
   - post_sampling
   
   Each of these functions is documented in the base class with usage instructions.
   You can also refer to existing models such as PassGAN for concrete implementations.
   
   Note: The only partially integrated model is PassGPT, which uses a custom evaluate function and differs slightly from the standard interface.

2. **Register Your Model**  
   Add an entry for your model in the `config/model/model_settings.yaml` file.  
   You need to specify:
   - `path_to_class`: The Python import path to your model class
   - `class_name`: The name of the model class
   - `path_to_config`: The path to your model's configuration file

   **Example**: Suppose you created a model named `ModelX` inside the `models/ModelX` directory.  
   You would add the following entry to `model_settings.yaml`:

   ```
   modelx:
     path_to_class: models.ModelX.ModelX
     class_name: ModelX
     path_to_config: ./models/ModelX/CONF/config.yaml
   ```

   Once "registered", your model will be automatically available for use in the framework through the --models command-line flag or via configuration files.

3. **Create the Config File**
   The file specified in path_to_config must exist and should define all training and evaluation parameters required by your model.
   
   Typical parameters include (but are not limited to):
   - batch_size
   - learning_rate
   - max_epochs
   - layer_dim
   - Any other architecture-specific hyperparameters
   
   These parameters will be automatically loaded into self.params within your model, making them accessible during both training and evaluation phases.