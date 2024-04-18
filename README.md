# explainable-methods-in-exercise-analysis
Data processing and models training for explainable pose estimation models

## Introduction
Project takes into account four methods:
- LSTM
- GraphCNN
- SignalCNN
- Statistical

to classify and  explain performed errors in exercise performance. EC3D dataset of 3D joints representations was obtained from https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises?tab=readme-ov-file.

## Installation
```git clone https://github.com/hubert/explainable-methods-in-exercise-analysis.git```

## Set environement
* locally: ```pip install -e .```
* docker: ```docker run```

## Process data and train models
1. Preprocess the data:
```bash
python scripts/process_data.py --data_path $PATH_TO_EC3D_DATASET --output_dir $DESIRED_DATA_OUTPUT
```

2. Train VartiationalAutoEncoder:
```bash
python train_vae.py --dataset_dir $DESIRED_DATA_OUTPUT --exercise $DESIRED_EXERCISE --model $ARCHITECTURE_NAME --weights_dir $VAE_OUTPUT_DIR 
```
3. Train Latent Classifier:
```bash
python train_latent_classifier.py --autoencoder $VAE_OUTPUT_DIR --dataset_dir $DESIRED_DATA_OUTPUT --exercise $DESIRED_EXERCISE --weights_dir $CLF_OUTPUT_DIR
```

## Explain mistakes

1. Generate proper version:
```bash 
python explain.py --autoencoder $VAE_OUTPUT_DIR --classifier $CLF_OUTPUT_DIR --dataset_dir $DESIRED_DATA_OUTPUT --exercise $DESIRED_EXERCISE --output_dir $COMPARISON_VIDEO_OUTPUT --sample_num $DATASET_SAMPLE_TO_EXPLAIN
```

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.