from rest_framework import serializers
from .models import DomesticInvoice, ImportInvoice


# class InvoiceSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Invoice
#         fields = '__all__'


class DomesticInvoiceSerializer(serializers.ModelSerializer):
    class Meta:
        model = DomesticInvoice
        fields = '__all__'

class ImportInvoiceSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImportInvoice
        fields = '__all__'

