import os
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import csv
from .management.commands.run_extraction import Command as ExtractionCommand
import threading
import logging

logger = logging.getLogger(__name__)

@staff_member_required
def log_viewer(request):
    log_file_path = os.path.join(settings.BASE_DIR, 'logs', 'app.log')
    if not os.path.exists(log_file_path):
        return HttpResponse("Log file not found.", status=404)
    try:
        with open(log_file_path, 'r') as f:
            logs = f.readlines()
    except Exception as e:
        return HttpResponse(f"Error reading log file: {str(e)}", status=500)
    # Show last 100 lines
    logs = logs[-100:]
    return render(request, 'testapp/log_viewer.html', {'logs': logs})

@staff_member_required
@csrf_exempt
def trigger_revalidation(request):
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")
    # Run extraction in a separate thread to avoid blocking
    def run_extraction():
        try:
            cmd = ExtractionCommand()
            cmd.handle()
            logger.info("Re-validation (extraction) triggered manually.")
        except Exception as e:
            logger.error(f"Error during manual re-validation: {e}")

    threading.Thread(target=run_extraction).start()
    return JsonResponse({'message': 'Re-validation triggered successfully.'})

@staff_member_required
def export_invoices_csv(request):
    from .models import DomesticInvoice, ImportInvoice

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="invoices_export.csv"'

    writer = csv.writer(response)
    # Write header
    header = [
        'Invoice Number', 'Invoice Date', 'Invoice Amount', 'Vendor Name', 'PO No',
        'HSN Code', 'IRN No', 'GRN No', 'GRN Date', 'Basic Amount', 'CGST', 'IGST', 'SGST',
        'PAN Number', 'GSTIN Number', 'Udyog Aadhar Registration Number', 'ECC Number',
        'Street/HouseNo.', 'City', 'District', 'State', 'Country', 'Payment Terms', 'Total Invoice Value'
    ]
    writer.writerow(header)

    # Write Domestic Invoices
    for invoice in DomesticInvoice.objects.all():
        writer.writerow([
            invoice.invoice_number, invoice.invoice_date, invoice.invoice_amount, invoice.vendor_name, invoice.po_no,
            invoice.hsn_code, invoice.irn_no, invoice.grn_no, invoice.grn_date, invoice.basic_amount, invoice.cgst,
            invoice.igst, invoice.sgst, invoice.pan_number, invoice.gstin_number, invoice.udyog_aadhar_registration_number,
            invoice.ecc_number, invoice.street_house_no, invoice.city, invoice.district, invoice.state, invoice.country,
            invoice.payment_terms, invoice.total_invoice_value
        ])

    # Write Import Invoices
    for invoice in ImportInvoice.objects.all():
        writer.writerow([
            invoice.invoice_number, invoice.invoice_date, invoice.invoice_amount, invoice.vendor_name, invoice.po_no,
            invoice.hsn_code, invoice.irn_no, invoice.grn_no, invoice.grn_date, invoice.basic_amount, invoice.cgst,
            invoice.igst, invoice.sgst, invoice.pan_number, '', '',  # GSTIN and Udyog Aadhar not applicable
            invoice.ecc_number, invoice.street_house_no, invoice.city, invoice.district, invoice.state, invoice.country,
            invoice.payment_terms, invoice.total_invoice_value
        ])

    return response

@staff_member_required
def admin_dashboard_view(request):
    return render(request, 'testapp/admin_dashboard.html')
