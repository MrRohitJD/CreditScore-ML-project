from django.contrib import admin

from import_export.admin import ImportExportModelAdmin
from .models import CreditData, PredictedData

# @admin.register(CreditData)
# class CreditDataAdmin(admin.ModelAdmin):
#     list_display = ('age', 'SeriousDlqin2yrs')

@admin.register(CreditData)
class CreditDataAdmin(ImportExportModelAdmin):
    pass

@admin.register(PredictedData)
class PredictedDataAdmin(ImportExportModelAdmin):
    pass
