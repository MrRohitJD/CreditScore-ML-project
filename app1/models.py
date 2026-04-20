from django.db import models

# Create your models here.
from django.db import models

class CreditData(models.Model):
    SeriousDlqin2yrs = models.IntegerField()
    RevolvingUtilizationOfUnsecuredLines = models.FloatField()
    age = models.IntegerField()
    NumberOfTime30_59DaysPastDueNotWorse = models.IntegerField()
    DebtRatio = models.FloatField()
    MonthlyIncome = models.FloatField(null=True, blank=True)
    NumberOfOpenCreditLinesAndLoans = models.IntegerField()
    NumberOfTimes90DaysLate = models.IntegerField()
    NumberRealEstateLoansOrLines = models.IntegerField()
    NumberOfTime60_89DaysPastDueNotWorse = models.IntegerField()
    NumberOfDependents = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return str(self.age)
    

class PredictedData(models.Model):
    PredictedDataId = models.IntegerField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID', null=False, blank=False)
    PredictedSeriousDlqin2yrs_lr = models.IntegerField(null=True, blank=True)
    PredictedSeriousDlqin2yrs_xgb = models.IntegerField(null=True, blank=True)
    PredictedSeriousDlqin2yrs_rf = models.IntegerField(null=True, blank=True)

    RevolvingUtilizationOfUnsecuredLines = models.FloatField(null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    NumberOfTime30_59DaysPastDueNotWorse = models.IntegerField(null=True, blank=True)
    DebtRatio = models.FloatField()
    MonthlyIncome = models.FloatField(null=True, blank=True)
    NumberOfOpenCreditLinesAndLoans = models.IntegerField(null=True, blank=True)
    NumberOfTimes90DaysLate = models.IntegerField(null=True, blank=True)
    NumberRealEstateLoansOrLines = models.IntegerField(null=True, blank=True)
    NumberOfTime60_89DaysPastDueNotWorse = models.IntegerField(null=True, blank=True)
    NumberOfDependents = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return str(self.age)
