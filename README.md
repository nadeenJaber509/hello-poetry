# 📊 Data Explorer

An interactive CSV Data Explorer built with **Python, Streamlit, Pandas, and Poetry** as part of the ProGineer AI Training Program.

The application allows users to upload CSV files, explore datasets, view summary statistics, identify missing values, and generate interactive visualizations.

## Features

- **CSV Upload:** Upload a CSV file or use the built-in sample dataset.
- **Data Preview:** Display the first five rows of the dataset.
- **Dataset Overview:** Show the number of rows and columns.
- **Column Types:** Display the data type of each column.
- **Summary Statistics:** Explore descriptive statistics for numerical and categorical columns.
- **Missing Values:** Identify missing values in each column.
- **Interactive Charts:** Select a numerical column and chart type (Histogram, Line Chart, or Bar Chart) from the sidebar.
- **Error Handling:** Display user-friendly messages when invalid or empty CSV files are uploaded.
- **Automated Testing:** Test data processing and application functionality using pytest.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.14 | Core programming language |
| Poetry | Dependency and environment management |
| Streamlit | Interactive web application |
| Pandas | Data loading and analysis |
| pytest | Automated testing |

## Project Structure

```text
hello-poetry/
├── app.py
├── src/
│   └── hello_poetry/
│       ├── __init__.py
│       ├── hello.py
│       └── data_utils.py
├── tests/
│   ├── __init__.py
│   ├── test_hello.py
│   ├── test_app.py
│   └── test_data_utils.py
├── sample.csv
├── invalid.csv
├── pyproject.toml
├── poetry.lock
├── README.md
└── .gitignore
```

## Getting Started

### 1. Clone the Repository

```bash
git clone -b nadeen-jaber https://github.com/nadeenJaber509/hello-poetry.git
cd hello-poetry
```

### 2. Install Dependencies

Make sure Python 3.14 and Poetry are installed.

```bash
poetry install
```

### 3. Run the Application

```bash
poetry run streamlit run app.py
```

Open the application at:

http://localhost:8501

## Using the Data Explorer

1. Launch the Streamlit application.
2. Upload a CSV file or explore the built-in sample dataset.
3. View dataset information, column types, statistics, and missing values.
4. Use the sidebar to select a numerical column.
5. Choose a chart type to update the visualization interactively.

## Sample CSV Files

Two example files are included for testing:

- **sample.csv:** A valid dataset containing sample records and missing values, useful for testing data analysis.
- **invalid.csv:** An intentionally invalid dataset used to verify error handling.

The application also includes a built-in sample dataset that loads automatically when no CSV file is uploaded.

## Running Tests

Run all automated tests using:

```bash
poetry run pytest -v
```

The project includes tests for:

- The original Hello World functions
- CSV loading and data analysis
- Streamlit application functionality

**Latest local test result: 10 passed.**

## Cloud Deployment

The application is hosted on Streamlit Community Cloud.

**Live Application:**  
https://hello-poetry-n3g8uw4ccm6afnqqothxus.streamlit.app

**GitHub Repository (nadeen-jaber branch):**  
https://github.com/nadeenJaber509/hello-poetry/tree/nadeen-jaber

The cloud application is connected to the `nadeen-jaber` branch, allowing changes pushed to GitHub to be deployed through Streamlit Community Cloud.

## Development

Developed as part of the **ProGineer AI Training Program**, focusing on Python project management, data analysis, automated testing, GitHub collaboration, and cloud deployment.

## Task 3 — Data Cleaning

The Clean Data page allows users to inspect, clean, and prepare
CSV datasets for machine learning.

### Features

- Detect missing values, duplicate rows, and column data types.
- Identify potential outliers using the IQR method.
- Convert numeric-looking text columns into numeric values.
- Handle missing values using drop, mean, median, or mode.
- Remove duplicate rows.
- Encode categorical features using One-Hot Encoding.
- Scale numerical features using StandardScaler.
- Split datasets into training and testing sets.
- Compare datasets before and after cleaning.
- Download the cleaned dataset as CSV.

### Data Leakage Prevention

The machine learning preparation workflow separates data cleaning
from fitted preprocessing.

The dataset is split into training and testing sets before
fitting imputers, encoders, or scalers.

Preprocessing is fitted on training data only and then applied
to testing data using the same learned parameters.

### Example Dataset

The included `messy_customers.csv` contains 515 rows,
including 15 duplicates and missing values.

Using Fill with Mode and Remove Duplicates produces:

- 500 rows
- 0 missing values
- 0 duplicates

### Tests

Run all tests with:

```bash
poetry run pytest -v