# OPAM (Online Purchasing-behavior Analysis & Management)

OPAM is a comprehensive machine learning pipeline designed for personal finance management. It leverages advanced algorithms to analyze transaction data, predict expenses, detect anomalies and fraud, cluster users, and recommend budgets.

## Features

*   **Expense Prediction**: Predicts future spending using an ensemble of ML models (Linear Regression, Random Forest, XGBoost, etc.).
*   **Anomaly Detection**: Identifies unusual spending patterns using statistical methods and isolation forests.
*   **Fraud Detection**: Scores transactions for fraud risk based on multiple risk factors.
*   **User Clustering**: Segments users into behavioral personas using K-Means clustering.
*   **Budget Recommendation**: Suggests optimized budgets based on historical spending and savings goals.
*   **Visualization**: Generates insightful charts and graphs for all analysis modules.

## Setup and Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd OPAM
    ```

2.  **Create and Activate Virtual Environment**:
    *   **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare Data
You can either generate synthetic data or use your own.

*   **Generate Synthetic Data**:
    ```bash
    cd scripts
    python generate_sample_data.py
    ```
    This will create `../data/transactions.csv`.

*   **Use Real Data**:
    Place your CSV file in `data/transactions.csv` and run the fixer tool if needed:
    ```bash
    cd scripts
    python fix_data.py
    ```

### 2. Run the Pipeline
You can run the project using the provided automation scripts or manually.

**Option A: Using Automation Script (Recommended)**
*   **Windows**: Double-click `run.bat` or run in terminal:
    ```powershell
    .\run.bat
    ```
*   **Mac/Linux**: Run the bash script:
    ```bash
    chmod +x run.sh
    ./run.sh
    ```
    This script will automatically set up the environment, install dependencies, check for data, and run the system.

**Option B: Manual Execution**
To run the complete analysis pipeline manually:

```bash
cd src
python run_all_systems.py
```

Follow the on-screen prompts. The system will execute all modules sequentially.

### 3. View Results
*   **Analysis Reports**: Check the `results/` directory for CSV files containing detailed analysis.
*   **Visualizations**: Check the `charts/` directory for PNG images of the generated insights.

## How It Works

The system operates through several interconnected modules:

1.  **Expense Predictor**: This module uses historical transaction data to forecast next month's total and category-wise spending. It employs an ensemble approach, combining predictions from Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, and XGBoost to achieve high accuracy.

2.  **Anomaly Detector**: This component flags transactions that deviate significantly from normal spending behavior. It utilizes statistical thresholds and isolation forest algorithms to identify outliers that may indicate errors or unusual activity.

3.  **Fraud Detector**: This module assigns a risk score to transactions based on multiple risk factors such as location, amount, and frequency. It helps in identifying potentially fraudulent activities by analyzing patterns that differ from the user's standard behavior.

4.  **User Clusterer**: This feature groups users based on their spending habits using K-Means clustering. It identifies distinct behavioral personas to provide personalized insights and recommendations tailored to specific user types.

5.  **Budget Recommender**: This system calculates safe spending limits and savings potential. It analyzes past spending trends and financial goals to suggest optimized budgets that encourage saving without compromising essential needs.

## License
[MIT License](LICENSE)
