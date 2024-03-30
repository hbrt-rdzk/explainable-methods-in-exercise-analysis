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
1. Clone the repository:
    ```bash
    git clone https://github.com/hubert/explainable-methods-in-exercise-analysis.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -e .
    ```

## Usage
1. Preprocess the data:
    ```bash
    python scripts/process_data.py --data_path $PATH_TO_EC3D_DATASET --output_dir $DESIRED_OUTPUT_PATH
    ```

2. Train the model:
    ```bash
    python train.py
    ```

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.