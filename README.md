# Load Type Prediction for Power System Optimization

## Problem Statement
The primary objective of this project is to develop a machine learning model capable of predicting the load type of a power system based on historical data. The "Load_Type" categorization includes "Light_Load", "Medium_Load", and "Maximum_Load". This classification problem requires candidates to apply their skills in data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, and model evaluation to predict the load type accurately.

## Dataset Details
The dataset provided for this task contains several features that are crucial for understanding and predicting the load type of a power system. These features include:
- Date_Time: Continuous-time data taken on the first of the month
- Usage_kWh: Industry Energy Consumption (Continuous kWh)
- Lagging_Current_Reactive.Power_kVarh: Continuous kVarh
- Leading_Current_Reactive_Power_kVarh: Continuous kVarh
- CO2(tCO2): Continuous ppm
- Lagging_Current_Power_Factor: Continuous
- Leading_Current_Power_Factor: Continuous
- NSM: Number of Seconds from midnight (Continuous S)
- Load_Type: Categorical (Light Load, Medium Load, Maximum Load)

## Evaluation Criteria
To assess the model's performance, an appropriate validation strategy will be implemented, using the last month of data as the test set. Metrics specific to classification problems, such as accuracy, precision, recall, and F1-score, will be used for evaluation.

## Code
The code for this project is available in the `notebooks/` directory. The main notebook, `Load Type Prediction for Power System Optimization.ipynb`, details the entire project workflow, including data preprocessing, exploratory data analysis, feature engineering, model selection, and evaluation.

## File Structure
- `data/`: Contains the dataset (`load_data.csv`) used for training and testing the model.
- `scripts/`: Contains Python scripts for various utility functions or preprocessing steps.
- `notebooks/`: Contains Jupyter notebooks, including the main notebook `Load Type Prediction for Power System Optimization.ipynb`, which details the entire project workflow.
- `README.md`: This file, providing an overview of the project and instructions for navigating the repository.
- `.gitignore`: Specifies files and directories to be ignored by Git, such as `.pyc` files or temporary files.
- `requirements.txt`: Lists the Python dependencies required to run the project, ensuring reproducibility.

## Usage
To reproduce the results or further explore the project:
1. Clone this repository to your local machine.
2. Navigate to the `notebooks/` directory and open [Load Type Prediction for Power System Optimization.ipynb](notebooks/Load%20Type%20Prediction%20for%20Power%20System%20Optimization.ipynb) in Jupyter Notebook or JupyterLab.
3. Follow the step-by-step instructions provided in the notebook to understand the project workflow, execute code cells, and analyze results.
4. Ensure that the required dependencies listed in `requirements.txt` are installed. You can install them using pip:

## Acknowledgements
The authors would like to acknowledge the following individuals and organizations for their contributions and support:
- **OpenAI**: For providing cutting-edge language models and advancing the field of artificial intelligence.
- **GitHub**: For providing a collaborative platform for hosting and sharing code repositories.
- **All the open-source libraries**: For their contributions to the field of machine learning, data analysis, and scientific computing.

## Contributions
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

## License
This project is licensed under the [Creative Commons](LICENSE).
