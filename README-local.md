# Progineer Streamlit Training — Tasks 1 to 4

This is a combined Streamlit app that brings together:
1. **Task 1:** Hello World
2. **Task 2:** Data Explorer (CSV upload, preview, dimensions, data types, summary statistics, missing values, and a simple column chart)
3. **Task 3:** Data Cleaning (fill or drop missing values, remove duplicates, download cleaned CSV)
4. **Task 4:** Train Model (Logistic Regression, Decision Tree, Random Forest; train/test accuracy, confusion matrix, and majority-class baseline)

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not recognized in PowerShell, try:

```bash
python -m streamlit run app.py
```

## Run tests

```bash
python -m pytest -v
```

## Deploy to Streamlit Community Cloud

1. Push this project folder to a GitHub repository.
2. Open https://share.streamlit.io/
3. Select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

The app is ready to deploy, but this archive does **not** create a live `streamlit.app` URL automatically. You must deploy it through your Streamlit account.

## Notes

- The included student CSV is a small demonstration dataset, not a meaningful model-training dataset.
- For Task 4, upload a dataset with a categorical target column and enough examples in each class.
- Numeric scaling is fitted inside the training pipeline, avoiding fitting the scaler on test data.
