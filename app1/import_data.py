import csv
import os
from django.conf import settings
from django.db import transaction
from app1.models import CreditData

BATCH_SIZE = 5000

def to_int(val):
    if val in ("", "NA"):
        return None
    return int(float(val))

def to_float(val):
    if val in ("", "NA"):
        return None
    return float(val)

def run():
    file_path = os.path.join(settings.BASE_DIR, 'data.csv')

    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        batch = []
        total = 0

        with transaction.atomic():
            for row in reader:

                # Skip invalid rows
                if not row or len(row) < 12:
                    continue

                # Skip repeated headers
                if row[1] == "SeriousDlqin2yrs":
                    continue

                try:
                    obj = CreditData(
                        SeriousDlqin2yrs=to_int(row[1]),
                        RevolvingUtilizationOfUnsecuredLines=to_float(row[2]),
                        age=to_int(row[3]),
                        NumberOfTime30_59DaysPastDueNotWorse=to_int(row[4]),
                        DebtRatio=to_float(row[5]),
                        MonthlyIncome=to_float(row[6]),
                        NumberOfOpenCreditLinesAndLoans=to_int(row[7]),
                        NumberOfTimes90DaysLate=to_int(row[8]),
                        NumberRealEstateLoansOrLines=to_int(row[9]),
                        NumberOfTime60_89DaysPastDueNotWorse=to_int(row[10]),
                        NumberOfDependents=to_int(row[11])
                    )

                    batch.append(obj)

                except Exception as e:
                    print(f"❌ Skipped row: {row} → {e}")
                    continue

                if len(batch) >= BATCH_SIZE:
                    CreditData.objects.bulk_create(batch)
                    total += len(batch)
                    print(f"Inserted {total} rows...")
                    batch.clear()

            if batch:
                CreditData.objects.bulk_create(batch)
                total += len(batch)

    print(f"✅ DONE! Total inserted: {total}")