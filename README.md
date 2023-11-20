# IDS706-Week12-Mini-Project: Use MLflow to Manage an ML Project

This project demonstrates the use of MLflow to manage and track a machine learning project. We implement a simple Linear Regression model using the California Housing dataset to predict housing values.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Conda (Anaconda or Miniconda)

## Setup

1. **Clone the Repository**
   
   Clone this repository to your local machine.

2. **Create and Activate a Conda Environment**

   Navigate to the project directory and create a conda environment using the `conda.yaml` file:
   
   ```bash
   conda env create -f conda.yaml
   conda activate mlflow-california-housing
   ```

## Running the Project

To run the project and train the machine learning model, execute the `main.py` script:

```bash
python main.py
```

## MLflow Tracking

This project uses MLflow for tracking experiments. To view the experiments:

1. **Launch MLflow UI**

   After running the `main.py` script, start the MLflow UI:

   ```bash
   mlflow ui
   ```

2. **Viewing the Results**

   Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000). Here, you can view the logged experiments, including parameters, metrics, and model artifacts.
   [!ml flow](pic.png)

## Project Structure

- `MLproject`: MLflow project configuration file.
- `conda.yaml`: Conda environment file.
- `main.py`: Main script for the machine learning model using the California Housing dataset.
- `models/`: Directory for storing model artifacts (managed by MLflow).