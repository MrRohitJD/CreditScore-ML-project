from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.db.models import Sum
from pathlib import Path
import pickle
import joblib
import numpy as np
import pandas as pd
import sqlite3

# Load model artifacts once during startup.
XGBOOST_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgboost_wt.joblib"
modelXgboost = joblib.load(XGBOOST_MODEL_PATH)

LOGISTIC_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logistic_regression_wt.joblib"
modelLogistic = joblib.load(LOGISTIC_MODEL_PATH)

RANDOM_FOREST_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "brfc_wt.joblib"
modelRandomForest = joblib.load(RANDOM_FOREST_MODEL_PATH)

from .models import CreditData, PredictedData



# Create your views here.


def _risk_label_from_probability(probability_percent):
    if probability_percent <= 20:
        return "Very Safe"
    if probability_percent <= 50:
        return "Medium Risk"
    return "High Risk"


def _predict_from_model(model_obj, features_df):
    prediction = int(model_obj.predict(features_df)[0])

    probability_percent = None
    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(features_df)
        probability_percent = float(proba[0][1]) * 100

    if probability_percent is None:
        annotation = "High Risk" if prediction == 1 else "Very Safe"
    else:
        annotation = _risk_label_from_probability(probability_percent)

    return {
        'prediction': prediction,
        'prob': probability_percent,
        'anno': annotation
    }


# def home(request):
#     return HttpResponse("<h2>Welcome to the Credit Score Prediction App</h2>")

def _build_late_payment_context():
    totals = CreditData.objects.aggregate(
        NumberOfTime30_59DaysPastDueNotWorse=Sum('NumberOfTime30_59DaysPastDueNotWorse'),
        NumberOfTime60_89DaysPastDueNotWorse=Sum('NumberOfTime60_89DaysPastDueNotWorse'),
        NumberOfTimes90DaysLate=Sum('NumberOfTimes90DaysLate')
    )
    return {
        'NumberOfTime30_59DaysPastDueNotWorse': int(totals['NumberOfTime30_59DaysPastDueNotWorse'] or 0),
        'NumberOfTime60_89DaysPastDueNotWorse': int(totals['NumberOfTime60_89DaysPastDueNotWorse'] or 0),
        'NumberOfTimes90DaysLate': int(totals['NumberOfTimes90DaysLate'] or 0)
    }


def _format_stat_value(value):
    if pd.isna(value):
        return '-'
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric):,}"
    if abs(numeric) >= 1000:
        return f"{numeric:,.2f}"
    return f"{numeric:.3f}"


def _build_summary_stats():
    feature_map = [
        ('Revolving Utilization', 'RevolvingUtilizationOfUnsecuredLines'),
        ('Age', 'age'),
        ('30-59 Days Past Due', 'NumberOfTime30_59DaysPastDueNotWorse'),
        ('Debt Ratio', 'DebtRatio'),
        ('Monthly Income', 'MonthlyIncome'),
        ('Open Credit Lines', 'NumberOfOpenCreditLinesAndLoans'),
        ('90 Days Late', 'NumberOfTimes90DaysLate'),
        ('Real Estate Loans', 'NumberRealEstateLoansOrLines'),
        ('60-89 Days Past Due', 'NumberOfTime60_89DaysPastDueNotWorse'),
        ('Dependents', 'NumberOfDependents')
    ]

    df = pd.DataFrame.from_records(CreditData.objects.values(*(f for _, f in feature_map)))
    if df.empty:
        return [
            {'feature': label, 'min': '-', 'max': '-', 'mean': '-', 'std': '-', 'null_pct': '0.0%'}
            for label, _ in feature_map
        ]

    rows = []
    for label, field in feature_map:
        series = pd.to_numeric(df[field], errors='coerce')
        valid = series.dropna()
        if valid.empty:
            min_val = max_val = mean_val = std_val = '-'
        else:
            min_val = _format_stat_value(valid.min())
            max_val = _format_stat_value(valid.max())
            mean_val = _format_stat_value(valid.mean())
            std_raw = valid.std()
            std_val = _format_stat_value(std_raw) if not pd.isna(std_raw) else '0'

        rows.append({
            'feature': label,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'null_pct': f"{series.isna().mean() * 100:.1f}%"
        })
    return rows


def _build_analytics_distribution_context():
    df = pd.DataFrame.from_records(CreditData.objects.values('age', 'MonthlyIncome'))

    age_labels = ['18-24', '25-30', '31-36', '37-42', '43-48', '49-54', '55-60', '61-66', '67-72', '73-78', '79+']
    income_labels = ['0-2k', '2k-4k', '4k-6k', '6k-8k', '8k-10k', '10k+']

    if df.empty:
        return {
            'age_distribution_labels': age_labels,
            'age_distribution_counts': [0] * len(age_labels),
            'income_distribution_labels': income_labels,
            'income_distribution_counts': [0] * len(income_labels)
        }

    age_series = pd.to_numeric(df['age'], errors='coerce')
    age_bins = [18, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, float('inf')]
    age_groups = pd.cut(age_series, bins=age_bins, labels=age_labels, right=False, include_lowest=True)
    age_counts = age_groups.value_counts(sort=False).fillna(0).astype(int).tolist()

    income_series = pd.to_numeric(df['MonthlyIncome'], errors='coerce')
    income_bins = [0, 2000, 4000, 6000, 8000, 10000, float('inf')]
    income_groups = pd.cut(income_series, bins=income_bins, labels=income_labels, right=False, include_lowest=True)
    income_counts = income_groups.value_counts(sort=False).fillna(0).astype(int).tolist()

    return {
        'age_distribution_labels': age_labels,
        'age_distribution_counts': age_counts,
        'income_distribution_labels': income_labels,
        'income_distribution_counts': income_counts
    }


def _build_dashboard_context():
    total_applicants = CreditData.objects.count()
    low_risk_count = CreditData.objects.filter(SeriousDlqin2yrs=0).count()
    high_risk_count = CreditData.objects.filter(SeriousDlqin2yrs=1).count()

    boxData_1 = {
        'total': total_applicants,
        'low_risk': round(low_risk_count / total_applicants * 100, 2) if total_applicants > 0 else 0,
        'high_risk': round(high_risk_count / total_applicants * 100, 2) if total_applicants > 0 else 0,
        'Accuracy': 86.4
    }
    boxData_2 = {
        'low_risk': low_risk_count,
        'high_risk': high_risk_count
    }

    Recent_Predictions = PredictedData.objects.all().order_by('-pk')[:5]
    boxData_3 = {
        'Recent_Predictions': Recent_Predictions
    }
    return {
        'boxData_1': boxData_1,
        'boxData_2': boxData_2,
        'boxData_3': boxData_3,
        'summary_stats': _build_summary_stats(),
        **_build_analytics_distribution_context(),
        **_build_late_payment_context()
    }


def stats(request):
    data = CreditData.objects.all().values()
    df = pd.DataFrame(data)
    context = df.describe().to_dict() if not df.empty else {}
    return render(request, 'index.html', {'stats': context, 'summary_stats': _build_summary_stats()})


def home(request):
    return render(request, 'home.html', _build_dashboard_context())

def predict(request):
    if request.method == 'POST':
        try:
            # 🔹 Get input values (safe parsing)
            data = {
                'RevolvingUtilizationOfUnsecuredLines': float(request.POST.get('RevolvingUtilizationOfUnsecuredLines', 0)),
                'age': int(request.POST.get('age', 0)),
                'NumberOfTime30-59DaysPastDueNotWorse': int(
                    request.POST.get('NumberOfTime30_59DaysPastDueNotWorse',
                    request.POST.get('NumberOfTime30-59DaysPastDueNotWorse', 0))
                ),
                'DebtRatio': float(request.POST.get('DebtRatio', 0)),
                'MonthlyIncome': float(request.POST.get('MonthlyIncome', 0)),
                'NumberOfOpenCreditLinesAndLoans': int(request.POST.get('NumberOfOpenCreditLinesAndLoans', 0)),
                'NumberOfTimes90DaysLate': int(request.POST.get('NumberOfTimes90DaysLate', 0)),
                'NumberRealEstateLoansOrLines': int(request.POST.get('NumberRealEstateLoansOrLines', 0)),
                'NumberOfTime60-89DaysPastDueNotWorse': int(
                    request.POST.get('NumberOfTime60_89DaysPastDueNotWorse',
                    request.POST.get('NumberOfTime60-89DaysPastDueNotWorse', 0))
                ),
                'NumberOfDependents': int(request.POST.get('NumberOfDependents', 0))
            }

        except (TypeError, ValueError) as e:
            context = {
                'prediction': 'NA',
                'probability': 'NA',
                'annotation': f'Invalid input: {e}'
            }
            return JsonResponse(context, status=400)

      
        columns = [
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]

        df = pd.DataFrame([data], columns=columns)

        logistic_output = _predict_from_model(modelLogistic, df)
        xgboost_output = _predict_from_model(modelXgboost, df)
        random_forest_output = _predict_from_model(modelRandomForest, df)

        # Keep xgboost as the primary score used by the headline card.
        prediction = xgboost_output['prediction']
        prob = xgboost_output['prob']
        anno = xgboost_output['anno']
        prob_val = prob


        PredictedData.objects.create(
            PredictedSeriousDlqin2yrs_lr=logistic_output['prediction'],
            PredictedSeriousDlqin2yrs_xgb=xgboost_output['prediction'],
            PredictedSeriousDlqin2yrs_rf=random_forest_output['prediction'],

            RevolvingUtilizationOfUnsecuredLines=data['RevolvingUtilizationOfUnsecuredLines'],
            age=data['age'],
            NumberOfTime30_59DaysPastDueNotWorse=data['NumberOfTime30-59DaysPastDueNotWorse'],
            DebtRatio=data['DebtRatio'],
            MonthlyIncome=data['MonthlyIncome'],
            NumberOfOpenCreditLinesAndLoans=data['NumberOfOpenCreditLinesAndLoans'],
            NumberOfTimes90DaysLate=data['NumberOfTimes90DaysLate'],
            NumberRealEstateLoansOrLines=data['NumberRealEstateLoansOrLines'],
            NumberOfTime60_89DaysPastDueNotWorse=data['NumberOfTime60-89DaysPastDueNotWorse'],
            NumberOfDependents=data['NumberOfDependents']
        )
        context = {
            'prediction': prediction,
            'probability': f"{prob_val:.2f}%" if prob_val is not None else 'NA',
            'annotation': anno,
            'prob': prob,
            'anno': anno,
            'prob_val': prob_val,
            'models': {
                'logistic_regression': logistic_output,
                'xgboost': xgboost_output,
                'random_forest': random_forest_output
            }
        }
        return JsonResponse(context)

    context = {
        'prediction': 'NA',
        'probability': 'NA',
        'annotation': 'Invalid request method'
    }
    accepts_json = 'application/json' in request.headers.get('Accept', '').lower()
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    if accepts_json or is_ajax:
        return JsonResponse(context, status=405)
    return redirect('home')

def analysis(request):
    context = _build_late_payment_context()
    return render(request, 'index.html', context)

def dashboardBox1(request):
    return render(request, 'home.html', _build_dashboard_context())