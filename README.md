# CreditScore - Give Me Some Credit

Django web app for credit default risk prediction using three machine learning models:
- Logistic Regression
- XGBoost
- Balanced Random Forest

The app accepts borrower inputs, returns probability and risk annotation, and stores recent predictions in SQLite.

## Dashboard Analysis (What We Implemented)

The web app includes an interactive ML dashboard that combines prediction, monitoring, and analysis in one place.

- **Live KPI cards**  for total applicants, low/high-risk mix, and headline model score.
- **Model comparison panel** showing performance across Logistic Regression, XGBoost, and Balanced Random Forest (with and without optimization), including Accuracy, ROC-AUC, and Weighted Precision.
- **Risk distribution visualization** (doughnut chart) based on current database records.
- **Recent predictions table** (latest 5 requests) pulled from `PredictedData` so users can review recent scoring activity.
- **Analytics section** with data-driven charts for:
  - late payment frequencies (30-59, 60-89, 90+ days),
  - income distribution buckets,
  - age distribution buckets.
- **Summary statistics table** (min, max, mean, std, null%) generated from `CreditData` features.
- **Prediction result card** that returns:
  - primary decision from XGBoost,
  - probability + risk annotation,
  - side-by-side outputs from all three models.

In short, the dashboard is not only a prediction form; it also provides model performance reporting, portfolio-level risk analytics, and recent-inference tracking for the ML project.

## Project Structure

- `manage.py` - Django entry point
- `app1/` - main Django app (views, models, urls)
- `templates/` - frontend templates
- `models/` - trained model artifacts (`*.joblib`)
- `db.sqlite3` - local development database

## Requirements

- Python 3.13+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

From the `CreditScore` folder:

```bash
python manage.py migrate
python manage.py runserver
```

Open:

- http://127.0.0.1:8000/

## Main Endpoint

- `POST /predict/`
  - Input: credit feature fields from the form
  - Output: primary prediction + all model outputs (`prediction`, `prob`, `anno`)

## Notes

- This repository uses SQLite for local development.
- Keep the model files inside `models/` available, otherwise prediction endpoints will fail at startup.
- `scikit-learn==1.6.1` is intentionally pinned for model compatibility.
# CreditScore-ML-
# CreditScore-ML-
