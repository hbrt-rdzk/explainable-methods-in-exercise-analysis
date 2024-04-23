# explainable-methods-in-exercise-analysis
This project presents generative method of properly executed exercise, based on Counterfactual Explainations and Variational AutoEncoders.

![knee_passes_toe_fixed-ezgif com-video-to-gif-converter](https://github.com/hbrt-rdzk/explainable-methods-in-exercise-analysis/assets/123837698/aedc9149-38e6-4695-bffa-34b38f0c3dcb)


## Introduction
To learn models proper time series data EC3D dataset of 3D joints representations was obtained from https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises?tab=readme-ov-file.

![deep](https://github.com/hbrt-rdzk/explainable-methods-in-exercise-analysis/assets/123837698/4f43b6c7-d168-4e73-9e0e-4f2ec75e18ae)
The designed framework takes a 3D pose representation as input and converts its signal into the first K Discrete Cosine Transform coefficients. A VAE then encodes this representation into a latent space. Perturbation of the sample is performed here to find the proper version of the exercise using counterfactual explanations. After applying corrections in the latent space, the sample is decoded in reverse order to obtain the appropriate version of the performed exercise.

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
