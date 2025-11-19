# Deep Learning Assignment 1: Linear & Logistic Regression

Implementation of Linear and Logistic Regression from scratch using NumPy for mobile price prediction and classification.

## 📋 Assignment Overview

This project implements two fundamental machine learning algorithms:
1. **Linear Regression** - For predicting continuous mobile prices
2. **Logistic Regression** - For binary classification of mobile prices (High/Low)

## 📊 Dataset

- **Source**: Mobile Prices 2023 Dataset (`dataset/mobile_prices_2023.csv`)
- **Total Examples**: 1,647 mobile phones
- **Features**: 5 numerical features
  - Rating (out of 5)
  - Number of Ratings
  - RAM (GB)
  - ROM/Storage (GB)
  - Battery (mAh)
- **Target**: Price in INR

## 🎯 Features Implemented

### Core Functions

1. **`load_data(file_path)`**
   - Loads CSV dataset
   - Preprocesses numerical features
   - Extracts RAM, ROM, and Battery from text columns
   - Returns feature matrix X and target vector Y

2. **`split_data(X, Y, test_size=0.3)`**
   - Splits data into 70% training and 30% testing sets
   - Uses sklearn's train_test_split

3. **`normalize(trainX, testX)`**
   - Standardizes features using StandardScaler
   - Prevents feature dominance due to different scales

4. **`add_bias(X)`**
   - Adds bias term (column of ones) to feature matrix

### Linear Regression Functions

5. **`h_linear(X, theta)`**
   - Hypothesis function: h(x) = X · θ
   - Computes predicted values

6. **`cost_linear(X, Y, theta)`**
   - Mean Squared Error (MSE) cost function
   - J(θ) = (1/2m) Σ(h(x) - y)²

### Logistic Regression Functions

7. **`sigmoid(z)`**
   - Sigmoid activation: σ(z) = 1 / (1 + e^(-z))
   - Clips values to prevent overflow

8. **`h_logistic(X, theta)`**
   - Hypothesis function: h(x) = σ(X · θ)

9. **`cost_logistic(X, Y, theta)`**
   - Cross-Entropy cost function
   - J(θ) = -(1/m) Σ[y·log(h(x)) + (1-y)·log(1-h(x))]

### Training Function

10. **`grad_desc(X, Y, theta, learning_rate, num_epoch, X_test, Y_test, cost_func, h_func)`**
    - Implements gradient descent optimization
    - Updates parameters: θ := θ - α·∇J(θ)
    - Records training and testing costs per epoch
    - Returns optimized parameters and cost history

11. **`plot_data(train_costs, test_costs, title, filename)`**
    - Visualizes cost convergence over epochs
    - Plots both training and testing costs
    - Saves plot to file

## 📈 Results

### Problem 1: Linear Regression (Price Estimation)

- **Best Learning Rate**: 0.01
- **Number of Epochs**: 2,000
- **Final Test Cost (MSE)**: 50,070,213.81
- **Convergence**: Model converges smoothly as shown in `linear_regression_cost_plot.png`

### Problem 2: Logistic Regression (Binary Classification)

- **Classification Task**: High Price (>median) vs Low Price (≤median)
- **Median Price Threshold**: ₹14,999.00
- **Best Learning Rate**: 0.1
- **Number of Epochs**: 2,000
- **Final Test Cost (Cross-Entropy)**: 0.3761
- **Convergence**: Model converges effectively as shown in `logistic_regression_cost_plot.png`

## 🚀 How to Run

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) as the package manager for faster and more reliable dependency management.

#### Install uv (if not already installed)
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

#### Setup Project
```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-directory>

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### Execution
```bash
# Run Jupyter notebook
uv run jupyter notebook abdullah_54.ipynb

# Or if virtual environment is activated
jupyter notebook abdullah_54.ipynb
```

Or execute all cells in sequence to:
1. Load and preprocess the dataset
2. Train Linear Regression model
3. Train Logistic Regression model
4. Generate cost plots

## 📁 Project Structure

```
.
├── abdullah_54.ipynb              # Main implementation notebook
├── dataset/
│   └── mobile_prices_2023.csv     # Dataset file
├── linear_regression_cost_plot.png    # Linear regression results
├── logistic_regression_cost_plot.png  # Logistic regression results
├── README.md                      # This file
├── pyproject.toml                 # Project dependencies (uv)
├── uv.lock                        # Lock file for reproducible builds
└── .venv/                         # Virtual environment (created by uv)
```

## 🔍 Key Implementation Details

### Data Preprocessing
- Removed currency symbols and commas from price data
- Extracted numerical values from text columns (RAM, ROM, Battery)
- Handled missing values by dropping incomplete rows
- Normalized features to zero mean and unit variance

### Model Training
- Initialized parameters to zero
- Used batch gradient descent
- Monitored both training and testing costs to detect overfitting
- Experimented with different learning rates and epochs

### Hyperparameter Tuning
- **Linear Regression**: Lower learning rate (0.01) for stability
- **Logistic Regression**: Higher learning rate (0.1) for faster convergence
- Both models trained for 2,000 epochs

## 🔧 Package Management

This project uses **uv** for dependency management, which offers:
- ⚡ **10-100x faster** than pip
- 🔒 **Reproducible builds** with lock files
- 🎯 **Better dependency resolution**
- 📦 **Unified tool** for package and project management

All dependencies are specified in `pyproject.toml` and locked in `uv.lock` for reproducibility.

## 📊 Visualizations

The project generates two plots:
1. **Linear Regression Cost Plot**: Shows MSE decreasing over epochs
2. **Logistic Regression Cost Plot**: Shows cross-entropy loss decreasing over epochs

Both plots display training and testing costs to monitor model performance.

## ⚠️ Important Notes

- No external ML libraries used for core algorithms (only NumPy)
- sklearn used only for data splitting and normalization
- All gradient descent and cost calculations implemented from scratch
- Code follows assignment requirements strictly

## 👨‍💻 Author

**Abdullah** (Roll No: 54)

## 📝 Submission Date

November 18, 2025

## 🎓 Course

Deep Learning - Assignment 1

---

**Note**: This implementation is for educational purposes as part of a Deep Learning course assignment.
