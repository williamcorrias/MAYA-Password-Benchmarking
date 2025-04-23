# MAYA: Addressing Inconsistencies in Generative Password Guessing through a Unified Benchmark

This repository is the official implementation of *MAYA: Addressing Inconsistencies in Generative Password Guessing through a Unified Benchmark*.

## Authors & Contacts

- **William Corrias**: | [ORCID](https://orcid.org/0009-0006-2270-9266) | [corrias@di.uniroma1.it](mailto:corrias@di.uniroma1.it) - Sapienza University of Rome. 
- **Fabio De Gaspari**: | [ORCID](https://orcid.org/0000-0001-9718-1044) | [degaspari@di.uniroma1.it](mailto:degaspari@di.uniroma1.it) - Sapienza University of Rome. 
- **Dorjan Hitaj**: | [ORCID](https://orcid.org/0000-0001-5686-3831) | [hitaj.d@di.uniroma1.it](mailto:hitaj.d@di.uniroma1.it) - Sapienza University of Rome. 
- **Luigi Vincenzo Mancini**: | [ORCID](https://orcid.org/0000-0003-4859-2191) | [mancini@di.uniroma1.it](mailto:mancini@di.uniroma1.it) - Sapienza University of Rome. 

## Overview

MAYA, is a **customizable**, **plug-and-play password benchmarking framework**, that adopts a **standardized and comprehensive testing methodology** for assessing generative password-guessing models across an extensive set of advanced testing scenarios. 

## Pre-Requisites

Before running this project, make sure you are using **Python >= 3.8**, as functionality with earlier versions is not guaranteed.

## Dependencies 

Dependencies are listed in the **requirements.txt** file located inside the root directory:

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

You can either install them one by one or all at once by running:

```
pip install -r requirements.txt
```

## Dataset

This repository does not directly contain the datasets. Instead, **datasets must be downloaded using the `download_raw_data.py` script**.

To integrate your own dataset into the framework, you’ll need to manually insert it following the expected format. This will be explained in the Customization section of the documentation.

### Downloading datasets

To download the datasets, **stay within the project's main folder and run:**

```
./script/utils/download_raw_data.py
```

Running the script **without any options** will download and save the following datasets into their respective directories within the `datasets` folder:

Currently, these are the available datasets:

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

If you want to **download only specific datasets**, run the script with the **`--datasets`** option, specifying the names of the datasets you want to download. For example:
```
./script/utils/download_raw_data.py --datasets rockyou, libero, gmail
```


## Usage

To run the framework, simply execute the **main.py** file.

Before doing so, **you can configure a lot of parameters**. There are two main ways to specify them: (1) via **command line** arguments, and (2) using **configuration files**.

**Note**: Unless otherwise specified, command line values always take precedence over those specified in configuration files. If a parameter is set in both places, the command line value will override the config file.

### Via Command Line
```
Usage: ./main.py    
            [--models STR [STR ...]]
            [--test_config PATH [PATH ...]]
            [--train_datasets STR [STR ...]]
            [--test_datasets STR [STR ...]]
            [--n_samples INT [INT ...]]
            [--max_length INT [INT ...]]
            [--autoload {0,1}]
            [--path_to_checkpoint PATH [PATH ...]]
            [--char_bag STR [STR ...]]
            [--train_split_percentage INT [INT ...]]
            [--train_chunk_percentage INT [INT ...]]
            [--test_frequency INT [INT ...]]
            [--test_reference PATH [PATH ...]]
            [--use_existing_samples INT [INT ...]]
            [--data_to_embed PATH [PATH ...]]
           
**NOTE**: All parameters, except for `--autoload`, can accept multiple 
values. When multiple values are provided, the program will generate 
all possible combinations and run them sequentially. For simplicity, the 
following options are explained assuming only a single value is passed.

Options:
    --models STR [STR ...]: Name of the model to run. The provided
        string must match a key defined in config/model/model_settings.yaml
        e.g.: --models passflow
    
    --test_config PATH [PATH ...]: PATH to a scenario config file. Note
        that the config file must match a specific format, which is 
        described in detail later in the documentation.
        e.g.: --test_config config/test/test_chunked_data_training.yaml 
    
    --train_datasets STR [STR ...]: Name of the training dataset. 
        e.g.: --test_config rockyou
    
    --test_datasets DATASET [DATASET ...]: Name of the testing dataset.
        NOTE: If you want to use the same dataset for both training and 
        testing, **do not pass this option**.
    
        Examples: 
            (1) Train on `linkedin` and test on `rockyou`:  
                --train_datasets linkedin --test_datasets rockyou
                
            (2) Train and test on `rockyou`:
                --train_datasets rockyou
            
    --n_samples INT [INT ...]: Number of passwords to generate.  
        The selected model will generate exactly this number of samples.  
        e.g.: --n_samples 500000000
    
    --max_length INT [INT ...]: Maximum length of passwords.  
        Passwords longer than this value will be discarded from the dataset.  
        The model will also generate passwords up to this maximum length.  
        e.g.: --max_length 12
    
    --autoload {0,1}: Whether to automatically load a model checkpoint. 
        If set to `0` and `path_to_checkpoint` is null, a new checkpoint 
        will be created. It will be named `checkpointX.pt`, where `X` 
        is one greater than the highest existing checkpoint. 
    
        If set to `1`, the system will automatically attempt to load 
        a checkpoint from the corresponding folder. If multiple checkpoints 
        are available (named using the format `checkpointX.pt`), 
        it will load the one with the highest `X`.  
    
    --path_to_checkpoint PATH [PATH ...]: Specifies the exact path to a 
        checkpoint file to load, instead of relying on automatic checkpoint 
        loading. This option overrides `--autoload` if both are provided.
        e.g.: --path_to_checkpoint checkpoints/passflow/checkpoint3.pt
        
    --char_bag STR [STR ...]: Specifies the set of characters to consider 
        for password generation. Any password containing characters not 
        included in the `char_bag` will be discarded.
        e.g.: --char_bag "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU
        VWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `" (default value).
    
    --train_split_percentage INT [INT ...]: Specifies the percentage of 
        the dataset to be used for training, with the remaining portion 
        used for testing. The value should be an integer between 0 and 100.  
        e.g.: --train_split_percentage 80 (default value)
    
    --train_chunk_percentage INT [INT ...]: Specifies the percentage or 
        exact number of passwords to select from the training dataset.  
        
        If the value is between 1 and 100, it selects that percentage 
        of passwords.  
        
        e.g.: --train_chunk_percentage 50
         
        If the value is greater than 100, it selects that exact number 
        of passwords.

        e.g.: --train_chunk_percentage 1000   
     
    --test_frequency INT [INT ...]: Selects the top or bottom percentage 
        of passwords from the testing dataset, based on their frequency.  
        
        A positive value selects the most frequent passwords. 
        e.g.: --test_frequency 10
        
        While a negative value selects the least frequent ones.  
        e.g.: --test_frequency -10 
        
    --test_reference PATH [PATH ...]:  Instead of generating new passwords,
        the model will use the passwords located at the specified PATH.
        
        Option 1 (faster method):
        If a path to a scenario config file (.yaml) is provided, the model 
        will automatically derive the passwords from the respective file,
        based on the settings in the config. 
        Any missing settings will be taken from the command-line arguments.  
        e.g.: using `--test_reference config/test/x.yaml` and 
        `--test_config config/test/y.yaml`, the program will evaluate the model 
        on test X using passwords generated during test Y.

        Option 2 (more flexible): If a path to a directory containing guesses
        is provided, the model will use the passwords from the first file 
        found in that directory.
    
    --use_existing_samples INT [INT ...]: Subsamples `--n_samples` passwords
        from the --use_existing_samples previously generated ones.
    
        NOTE: All settings (except for `n_samples`) must exactly match those 
        used to generate the original samples.   

    --data_to_embed PATH [PATH ...]: Path to the data to be embedded. 
        The embedding process will be handled solely by PassFlow's encoder. 
              
```
### Through Config Files
** We highly recommend passing options via the command line, as it is much easier **. Alternatively, all options can be specified in the corresponding config files, where they will be saved for subsequent runs.

**Note**: Unless otherwise specified, command line values always take precedence over those specified in configuration files. If a parameter is set in both places, the command line value will override the config file.

In this context, things are slightly different and a bit more complex.

We define five types of parameters:

1) general_params: These are general parameters that do not affect either the training or test dataset. eg.: autoload, models, path_to_checkpoint, test_config, test_reference, n_samples, use_existing_samples and data_to_embed.
2) pre_split_params: Parameters that apply before the dataset is split, affecting both the training and test datasets. e.g.: max_length, and char_bag.
3) split_params: Parameters related to the splitting procedure, impacting both training and test datasets. e.g: train_split_percentage, train_datasets, and test_datasets.
4) post_split_params: Parameters that affect both datasets but are applied after the data is split. e.g: train_chunk_percentage.
5) test_params: Parameters that affect only the test dataset. e.g: test_frequency.

To specify a parameter through config files, define it under the appropriate section in `config/general_settings.yaml` and assign a value.

**Example 1**: Suppose you want to run PassGAN. You would specify the `models` parameter under the `general_params` section and set its value to `passgan`:

```
general_params:
  ...
  models:
    - passgan
  ...
```

**Example 2**: Want to run both PassGAN and PassFlow? Just list both models:

```
general_params:
  ...
  models:
    - passgan
    - passflow
  ...
```

**Note**: All parameters—except for `autoload`—must be passed as a list, even if you’re only providing a single value (as shown in Example 1). In YAML, you can do this by preceding each value with a `-`.

Parameters specified in `config/general_settings.yaml` can be overridden not only by command-line arguments, but also by parameters defined in the selected test configuration file.  
For example, if you run the script with `--test_config config/test/test_chunked_data_training.yaml`, then any overlapping parameters defined in `test_chunked_data_training.yaml` will override those in `general_settings.yaml`.

In this case, you not only need to specify the parameter type (e.g., `split_params`), but also the pre-processing function where the parameter will be applied. This is done using the format: `filename.function_name`.

```
test_chunked_data_training:
    ...
    split_params:

        standard_preprocessing.test_centric_split:  # format: filename.function
           train_split_percentage:
              - 80
    ...
```

The `standard_preprocessing` file, which contains all pre-processing functions, is located at: `script/dataset/preprocessing/standard_preprocessing.py`. 
From there, you can check which parameters need to be passed to a function by inspecting the `kwargs` used within each function.

## Quick Testing
Evaluation scenarios are divided according to the Research Questions (RQs), exactly as structured in the paper.

**Note**: We are still actively working on the code of this repository. Upcoming updates will further simplify the benchmarking process.

You can evaluate your model on one of our standardized scenarios using:

```
./main.py --models <your_model> --train_datasets <dataset> --test_config config/test/<chosen_scenario>.yaml
```

### RQ1: 
```
./main.py --models <your_model> --train_datasets rockyou --test_config config/test/test_chunked_data_training.yaml
```

### RQ2: 
```
./main.py --models <your_model> --train_datasets rockyou --test_config config/test/test_chunked_data_training.yaml --max_length 12
```

### RQ3:
```
./main.py --models <your_model> --train_datasets rockyou --test_config config/test/test_chunked_data_training.yaml --max_length 12 --train_chunk_percentage 850000 1785681 2700000 4710736 11802325 100
```

### RQ4:
```
./main.py --models <your_model> --train_datasets rockyou --test_datasets linkedin --test_config config/test/test_cross_datasets.yaml
```

### RQ5:
```
./main.py --models <your_model> --train_datasets rockyou --test_config config/test/test_on_frequency.yaml
```

## Customization
MAYA, is fully-customizable, offering an easy interface to add your models, datasets and scenarios.

### How To Add a Model
To integrate a custom model into MAYA, follow these steps:

1. **Subclass the Base Model**  
   Your custom model **must** inherit from the `Model` base class defined in `script/test/model.py`.  
   Make sure to implement any methods not already defined in the base class: `prepare_data`, `prepare_model`, `save`, `load`, `train`, and `evaluate`.  
   For better clarity, you can refer to the PassGAN implementation as a guide.

2. **Register the Model**  
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

   Once registered, your model will be recognized and available for use within the framework via the `--models` command-line flag or config file.

### How To Add an Evaluation Scenario

To add a new evaluation scenario, create a `.yaml` configuration file inside the `config/test` directory. You can use any of the existing files in that folder as a reference.

Each config file must:
1. Define the name of the test as the top-level key.
2. Group parameters according to their type (`pre_split_params`, `split_params`, `post_split_params`, etc.).
3. Specify the path to the preprocessing function using the format: `filename.function_name`.

**Example:**

```
test_chunked_data_training:

    pre_split_params:

        standard_preprocessing.filter_by_length:
            max_length:
              - 12

        standard_preprocessing.filter_by_char_bag:
            char_bag:
              - "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `"

    split_params:

        standard_preprocessing.test_centric_split:
            train_split_percentage:
              - 80

    post_split_params:

        standard_preprocessing.chunk_train_dataset:
            train_chunk_percentage:
              - 100
```

#### Adding or Modifying Preprocessing Functions

You can either:

- Modify the existing `standard_preprocessing.py` file located at `script/dataset/preprocessing/standard_preprocessing.py` (**not recommended**), or  
- Create a new file inside `script/dataset/preprocessing/` containing your custom preprocessing functions.

**Example:**  
Suppose you create a new file named `personalized_preprocessing.py`. In your config file, you would reference a function like this:

```
    ...
        personalized_preprocessing.yourfunction:
            your_parameter:
              - your_value
    ...
```
   
### How To Add a Dataset

To add a custom dataset to MAYA, follow these steps:

1. **Insert the `.pickle` File**  
   Place a `.pickle` file containing only passwords (each separated by a newline `\n`) into the appropriate directory inside `datasets/`, according to its **language** and **service type**.

   Example path: datasets/en/social-net/DatasetX.pickle


2. **Register the Dataset**  
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
   ```