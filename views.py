import os
import re
import json
import tempfile
import pdfplumber
from io import BytesIO
import base64
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotAllowed, HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import TemplateView
from rest_framework import viewsets, filters, permissions
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import viewsets, filters, permissions
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from .models import DomesticInvoice, ImportInvoice, CustomUser
from .serializers import DomesticInvoiceSerializer, ImportInvoiceSerializer
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv
from pdf2image import convert_from_path
from pathlib import Path
import logging
from rest_framework.permissions import IsAuthenticated
import requests
import time
from django.conf import settings
from django.core.cache import cache
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from .save_extracted_data import save_extracted_data
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from difflib import SequenceMatcher
import logging
import os
import string

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'app.log')

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler with UTF-8 encoding if possible
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    try:
        console_handler.stream.reconfigure(encoding='utf-8')
    except Exception:
        # For Python versions < 3.7 or if reconfigure is not available, fallback to no reconfiguration
        pass

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from datetime import datetime, timedelta

from django.views.decorators.http import require_POST
from django.http import JsonResponse, HttpResponseRedirect
from django.urls import reverse 
@login_required
@require_POST
def manual_trigger_job(request, job_id):
    from django.utils import timezone
    job = scheduler.get_job(job_id)
    if job:
        job.modify(next_run_time=timezone.now())
        # If AJAX, return JSON
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({"status": f"Job '{job_id}' triggered manually"})
        # If form POST, redirect back to cron job page with a message
        return HttpResponseRedirect(reverse('cron_job_page') + '?message=Job triggered')
    else:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({"error": f"Job '{job_id}' not found"}, status=404)
        return HttpResponseRedirect(reverse('cron_job_page') + '?message=Job not found')
 
@login_required
def manual_extraction_run(request):
    job_id = 'apscheduler_daily_extraction_job'
    job = scheduler.get_job(job_id)
    if job:
        # Set status to processing
        cache.set('extraction_job_status', 'processing')
        job.modify(next_run_time=datetime.now())
        return JsonResponse({"status": f"Job '{job_id}' triggered to run immediately"})
    else:
        return JsonResponse({"error": f"Job '{job_id}' not found"}, status=404)


# manual_comparison_run triggers the comparison job immediately (no delay)
@login_required
def manual_comparison_run(request):
    job_id = 'apscheduler_comparison_job'
    job = scheduler.get_job(job_id)
    if job:
        # Set status to processing
        cache.set('comparison_job_status', 'processing')
        job.modify(next_run_time=datetime.now())
        return JsonResponse({"status": f"Job '{job_id}' triggered to run immediately"})
    else:
        return JsonResponse({"error": f"Job '{job_id}' not found"}, status=404)

@login_required
def comparison_status_view(request):
    status = cache.get('comparison_job_status')
    logger.debug(f"comparison_job_status from cache: {status}")
    if status is None:
        status = 'unknown'
    return JsonResponse({'status': status})

@login_required
def extraction_status_view(request):
    status = cache.get('extraction_job_status')
    logger.debug(f"Fetched extraction_job_status from cache: {status}")
    if status is None:
        status = 'unknown'
    return JsonResponse({'status': status})

@login_required
def stop_cron_job(request):
    if request.method == 'POST':
        # Clear the status of both extraction and comparison jobs
        cache.delete('extraction_job_status')
        cache.delete('comparison_job_status')
        return JsonResponse({'status': 'Cron jobs stopped'})
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def log_test_messages():
    logger.info("Test info log: Application is running smoothly.")
    logger.warning("Test warning log: This is a sample warning message.")
    logger.error("Test error log: This is a sample error message.")

# Call the test log function at module load to generate sample logs
# Add these near your other imports at the top
import re
from datetime import datetime
import logging

def clean_and_parse_json_response(raw_response):
    """
    Robustly clean and parse JSON responses from AI
    """
    try:
        # First try to parse directly in case it's already clean JSON
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass
    
    # Clean the response
    cleaned = raw_response.strip()
    
    # Remove everything before the first { and after the last }
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}') + 1
    
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx:end_idx]
    
    # Remove code block markers if present
    cleaned = cleaned.replace('```json', '').replace('```', '')
    
    # Remove any trailing characters after the last }
    cleaned = re.sub(r'}[^}]*$', '}', cleaned)
    
    # Remove any notes or explanations after the JSON
    cleaned = re.sub(r'}\s*###.*$', '}', cleaned, flags=re.DOTALL)
    
    # Fix common JSON issues
    cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
    cleaned = re.sub(r',\s*}', '}', cleaned)  # Remove trailing commas
    cleaned = re.sub(r',\s*]', ']', cleaned)  # Remove trailing commas in arrays
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse cleaned JSON: {e}\nCleaned text: {cleaned[:500]}...")
        return None
    
def extract_json_from_response(text):
    """Extracts the JSON object from a markdown-style code block like ```json ... ```."""
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # fallback: try to find a pure JSON object if not wrapped in code block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0).strip()
    raise ValueError("No JSON object found in the response.")
def parse_date(date_str):
    """
    Robust date parsing for various formats including:
    - "221024" (DDMMYY)
    - "07.10.2024" (DD.MM.YYYY)
    - And all previously supported formats
    """
    if not date_str or str(date_str).lower().strip() in ['empty', '']:
        return None

    # Clean the date string first - remove quotes, commas, whitespace
    date_str = str(date_str).strip(' "\',')
    
    # Try to extract date from complex strings
    date_patterns = [
        r'(\d{1,2}-\w{3}-\d{2,4})',    # 1-Nov-24 or 1-Nov-2024
        r'(\d{1,2}/\d{1,2}/\d{2,4})',  # 08/11/24 or 08/11/2024
        r'(\d{1,2}\.\d{1,2}\.\d{2,4})', # 07.10.2024
        r'(\d{6})',                     # 221024 (DDMMYY)
        r'(\d{8})',                     # 08112024 (DDMMYYYY)
        r'(\d{1,2} \w{3,} \d{2,4})'    # 1 November 24
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            date_str = match.group(1)
            break
    
    # Define all possible date formats to try
    formats = [
        ('%d%m%y', lambda s: len(s) == 6),       # 221024 → 22-10-2024
        ('%d.%m.%Y', lambda s: '.' in s),        # 07.10.2024
        ('%d%m%Y', lambda s: len(s) == 8),       # 08112024
        ('%d-%b-%y', lambda s: '-' in s),        # 1-Nov-24
        ('%d-%b-%Y', lambda s: '-' in s),        # 1-Nov-2024
        ('%d/%m/%y', lambda s: '/' in s),        # 08/11/24
        ('%d/%m/%Y', lambda s: '/' in s),        # 08/11/2024
        ('%d %B %y', lambda s: ' ' in s),        # 1 November 24
        ('%d %B %Y', lambda s: ' ' in s),        # 1 November 2024
        ('%Y-%m-%d', lambda s: s.count('-') == 2), # ISO format
    ]
    
    for fmt, condition in formats:
        if condition(date_str):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None
# Configure logging at module level (top of views.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
import pandas as pd
import re
from django.shortcuts import render
from .models import DomesticInvoice, ImportInvoice
from django.contrib.auth.decorators import login_required
import logging

logger = logging.getLogger(__name__)
# @role_required(allowed_roles=['admin', 'manager'])
from testapp.scheduler_engine import scheduler
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.http import JsonResponse
from testapp.tasks import apscheduler_daily_extraction_job, apscheduler_comparison_job

@login_required
@csrf_exempt
def cron_job_page_view(request):
    message = None
    if request.method == 'POST':
        if 'extraction' in request.POST:
            try:
                job = scheduler.get_job('apscheduler_daily_extraction_job')
                if job:
                    job.modify(next_run_time=timezone.now())
                    message = "Extraction is triggered. Please check Extraction Dashboard."
                else:
                    message = "Extraction job not found in scheduler. Please ensure APScheduler is running."
            except Exception as e:
                message = f"Extraction failed: {str(e)}"
        elif 'comparison' in request.POST:
            try:
                job = scheduler.get_job('apscheduler_comparison_job')
                if job:
                    job.modify(next_run_time=timezone.now())
                    message = "Comparison is triggered. Please check Comparison Dashboard."
                else:
                    message = "Comparison job not found in scheduler. Please ensure APScheduler is running."
            except Exception as e:
                message = f"Comparison failed: {str(e)}"
    return render(request, 'testapp/cron_job.html', {'message': message})


def normalize_column_name(column_name):
    """Normalize column names to match database fields."""
    return re.sub(r'[^a-zA-Z0-9]', '_', column_name).lower().strip('_')

def normalize_value(value):
    """Normalize values for comparison."""
    if pd.isna(value):
        return ''
    value_str = str(value).lower()
    # Remove punctuation
    value_str = value_str.translate(str.maketrans('', '', string.punctuation))
    # Expand common company suffix abbreviations
    abbreviations = {
        'pvt ltd': 'private limited',
        'pvt. ltd.': 'private limited',
        'pvt. ltd': 'private limited',
        'pvt ltd.': 'private limited',
        'ltd': 'limited',
        'ltd.': 'limited',
        'pty ltd': 'proprietary limited',
        'pty. ltd.': 'proprietary limited',
        'inc': 'incorporated',
        'corp': 'corporation',
        'co': 'company',
        'co.': 'company',
        'aust': 'australia',
        'aust.': 'australia',
        'australia pty ltd': 'australia proprietary limited',
        'pty': 'proprietary',
        'llc': 'limited liability company',
        'llp': 'limited liability partnership',
    }
    # Replace abbreviations with full forms
    for abbr, full in abbreviations.items():
        # Use regex word boundaries to avoid partial replacements
        value_str = re.sub(r'\b' + re.escape(abbr) + r'\b', full, value_str)
    # Normalize whitespace
    value_str = ' '.join(value_str.split())
    return value_str

def enhanced_fuzzy_match(a, b, threshold=0.85):
    ratio = SequenceMatcher(None, a, b).ratio()
    # Removed verbose logging of fuzzy match ratio to reduce log noise
    # logger.info(f"Fuzzy match ratio between '{a}' and '{b}': {ratio}")
    return ratio >= threshold

def run_comparison():
    excel_file_path = os.getenv("EXCEL_FILE_PATH")
    try:
        sheets = pd.read_excel(excel_file_path, sheet_name=None, header=2)
        logger.info(f"Excel file read successfully with header at row 2. Sheets: {list(sheets.keys())}")

        domestic_mapping = {
            'GSTIN': 'gstin_number',
            'PAN NO.': 'pan_number',
            'AADHAR(Indv.)': 'aadhar_individual',
            'UDYAM': 'udyog_aadhar_certificate_date',
            'Region': 'state',
            'Email ID1':'email_ids',
            'Email ID2':'email_ids',
            'Email ID3':'email_ids',
            'Email ID4':'email_ids',
            'Email ID5':'email_ids',
            'Telephone No.':'mobile_landline_numbers',
            'Telephone No.2':'mobile_landline_numbers'



        }
        import_mapping = {
            'Region': 'state',
            'PAN NO.': 'pan_number',
            'Email ID1':'email_ids',
            'Email ID2':'email_ids',
            'Email ID3':'email_ids',
            'Email ID4':'email_ids',
            'Email ID5':'email_ids',
            'Telephone No.':'mobile_landline_numbers',
            'Telephone No.2':'mobile_landline_numbers'

        }

        domestic_model_fields = [field.name for field in DomesticInvoice._meta.get_fields() if field.concrete and not field.many_to_many and not field.auto_created]
        import_model_fields = [field.name for field in ImportInvoice._meta.get_fields() if field.concrete and not field.many_to_many and not field.auto_created]

        results = []
        for sheet_name, df in sheets.items():
            logger.info(f"Processing sheet: {sheet_name}")

            df.columns = [normalize_column_name(col) for col in df.columns]
            logger.info(f"Normalized columns for sheet {sheet_name}: {df.columns.tolist()}")

            if sheet_name.lower() == 'domestic':
                model_fields = domestic_model_fields
                field_mapping = domestic_mapping
            elif sheet_name.lower() == 'import':
                model_fields = import_model_fields
                field_mapping = import_mapping
            else:
                logger.warning(f"Unknown sheet name: {sheet_name}, skipping.")
                continue

            mapped_columns = {}

            def best_fuzzy_match_column(col, candidates, threshold=0.8):
                best_match = None
                best_ratio = 0
                for candidate in candidates:
                    ratio = SequenceMatcher(None, col, candidate).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = candidate
                if best_ratio >= threshold:
                    return best_match
                return None

            # List of sensitive fields that should only be mapped by exact match
            sensitive_fields = ['bank_account_name']

            for col in df.columns:
                matched_field = None
                for key, mapped_field in field_mapping.items():
                    if col.lower() == normalize_column_name(key):
                        # Block mapping between bic_code and ifsc_code for import sheets
                        if sheet_name.lower() == 'import':
                            if (mapped_field == 'bic_code' and 'ifsc' in col.lower()) or (mapped_field == 'ifsc_code' and 'bic' in col.lower()):
                                continue
                        matched_field = mapped_field
                        break
                if matched_field and matched_field in model_fields:
                    if matched_field in mapped_columns:
                        if isinstance(mapped_columns[matched_field], list):
                            mapped_columns[matched_field].append(col)
                        else:
                            mapped_columns[matched_field] = [mapped_columns[matched_field], col]
                    else:
                        mapped_columns[matched_field] = col
                    continue

                # Block fuzzy matching for sensitive fields
                for sensitive_field in sensitive_fields:
                    if sensitive_field in model_fields and sensitive_field not in mapped_columns:
                        # Only allow exact match, skip fuzzy for this field
                        continue

                matched_col = best_fuzzy_match_column(col, model_fields)
                # Block mapping between bic_code and ifsc_code for import sheets in fuzzy matching as well
                if matched_col:
                    if sheet_name.lower() == 'import':
                        if (matched_col == 'bic_code' and 'ifsc' in col.lower()) or (matched_col == 'ifsc_code' and 'bic' in col.lower()):
                            continue
                    # Block fuzzy for sensitive fields
                    if matched_col in sensitive_fields:
                        continue
                    mapped_columns[matched_col] = col

            logger.info(f"Matched columns in sheet {sheet_name}: {mapped_columns}")

            matching_columns = list(mapped_columns.keys())
            logger.info(f"Matching columns in sheet {sheet_name}: {matching_columns}")

            if 'vendor_name' not in mapped_columns:
                logger.warning(f"'vendor_name' column missing in sheet: {sheet_name}")
                continue

            df['normalized_vendor_name'] = df[mapped_columns['vendor_name']].apply(normalize_value)

            domestic_invoices = list(DomesticInvoice.objects.all())
            import_invoices = list(ImportInvoice.objects.all())

            domestic_map = {normalize_value(inv.vendor_name): inv for inv in domestic_invoices}
            import_map = {normalize_value(inv.vendor_name): inv for inv in import_invoices}

            for idx, row in df.iterrows():
                normalized_vendor = row['normalized_vendor_name']

                domestic_invoice = None
                for db_vendor, inv in domestic_map.items():
                    if enhanced_fuzzy_match(normalized_vendor, db_vendor):
                        domestic_invoice = inv
                        break
                if domestic_invoice:
                    comparison_result = compare_row_with_invoice(row, domestic_invoice, 'domestic', mapped_columns)
                    results.append(comparison_result)
                    matched_cols = list(mapped_columns.keys())
                    logger.info(f"Domestic Invoice {domestic_invoice.invoice_number}: {len(matched_cols)} matching columns: {matched_cols}")
                    logger.info(f"Domestic Invoice {domestic_invoice.invoice_number}: comparison_result: {comparison_result}")

                    # Ensure mismatched_invoice_data is populated correctly
                    if comparison_result.get('mismatches'):
                        mismatched_fields = [m['field'] for m in comparison_result['mismatches']]
                        domestic_invoice.Description = ", ".join(mismatched_fields)

                        # Extract mismatched invoice data and master data, preserving '@' in emails
                        invoice_values = [str(m.get('db_value', 'No data')).replace(' at ', '@') for m in comparison_result['mismatches']]
                        master_values = [str(m.get('excel_value', 'No data')).replace(' at ', '@') for m in comparison_result['mismatches']]
                        logger.info(f"Domestic Invoice {domestic_invoice.invoice_number}: invoice_values: {invoice_values}")
                        logger.info(f"Domestic Invoice {domestic_invoice.invoice_number}: master_values: {master_values}")

                        # Join and truncate values to fit model field constraints
                        domestic_invoice.mismatched_invoice_data = ", ".join(invoice_values)[:500]
                        domestic_invoice.mismatched_master_data = ", ".join(master_values)[:500]
                        domestic_invoice.status = "miss matched"
                        logger.info(f"Domestic Invoice {domestic_invoice.invoice_number}: mismatched_invoice_data: {domestic_invoice.mismatched_invoice_data}")
                        logger.info(f"Domestic Invoice {domestic_invoice.invoice_number}: mismatched_master_data: {domestic_invoice.mismatched_master_data}")
                    else:
                        domestic_invoice.status = "matched"
                        domestic_invoice.Description = ""
                        domestic_invoice.mismatched_invoice_data = ""
                        domestic_invoice.mismatched_master_data = ""
                    domestic_invoice.save()

                import_invoice = None
                for db_vendor, inv in import_map.items():
                    if enhanced_fuzzy_match(normalized_vendor, db_vendor):
                        import_invoice = inv
                        break
                if import_invoice:
                    comparison_result = compare_row_with_invoice(row, import_invoice, 'import', mapped_columns)
                    results.append(comparison_result)
                    matched_cols = list(mapped_columns.keys())
                    logger.info(f"Import Invoice {import_invoice.invoice_number}: {len(matched_cols)} matching columns: {matched_cols}")

                    if comparison_result.get('mismatches'):
                        import_invoice.status = "miss matched"
                        mismatched_fields = [m['field'] for m in comparison_result['mismatches']]
                        import_invoice.Description = ", ".join(mismatched_fields)
                        # Save detailed mismatch values
                        # Filter out empty or None values before joining
                        invoice_values = [str(m.get('db_value', '')).strip() for m in comparison_result['mismatches'] if m.get('db_value', '') not in [None, '', 'None']]
                        master_values = [str(m.get('excel_value', '')).strip() for m in comparison_result['mismatches'] if m.get('excel_value', '') not in [None, '', 'None']]
                        invoice_data = ', '.join(invoice_values)
                        master_data = ', '.join(master_values)
                        # Set default string if empty
                        if not invoice_data:
                            invoice_data = "No mismatches"
                        if not master_data:
                            master_data = "No mismatches"
                        # Truncate to 500 characters to fit model field max_length
                        invoice_data_truncated = invoice_data[:500]
                        master_data_truncated = master_data[:500]
                        logger.info(f"Import Invoice {import_invoice.invoice_number} - invoice_data to save (truncated): {invoice_data_truncated}")
                        logger.info(f"Import Invoice {import_invoice.invoice_number} - master_data to save (truncated): {master_data_truncated}")
                        import_invoice.mismatched_invoice_data = invoice_data_truncated
                        import_invoice.mismatched_master_data = master_data_truncated
                    else:
                        import_invoice.status = "matched"
                        import_invoice.Description = ""
                        import_invoice.mismatched_invoice_data = ""
                        import_invoice.mismatched_master_data = ""
                    import_invoice.save()

        if not results:
            return {'results': [], 'message': 'No matching records found.'}

        domestic_results = []
        import_results = []

        for res in results:
            mismatched_fields = []
            invoice_data = []
            master_data = []

            for mismatch in res.get('mismatches', []):
                mismatched_fields.append(mismatch['field'])
                # Preserve '@' in emails for output
                invoice_data.append(str(mismatch.get('db_value', '')).replace(' at ', '@'))
                master_data.append(str(mismatch.get('excel_value', '')).replace(' at ', '@'))

            formatted_result = {
                'invoice_number': res.get('invoice_number', ''),
                'vendor_name': res.get('vendor_name', ''),
                'mismatched_fields': ', '.join(mismatched_fields),
                'mismatched_count': len(mismatched_fields),
                'invoice_data': ', '.join(invoice_data),
                'master_data': ', '.join(master_data),
            }

            if res.get('invoice_type') == 'domestic':
                domestic_results.append(formatted_result)
            elif res.get('invoice_type') == 'import':
                import_results.append(formatted_result)

        domestic_mismatched_count_total = sum(len(res.get('mismatches', [])) for res in results if res.get('invoice_type') == 'domestic')
        import_mismatched_count_total = sum(len(res.get('mismatches', [])) for res in results if res.get('invoice_type') == 'import')

        return {
            'domestic_results': domestic_results,
            'import_results': import_results,
            'domestic_mismatched_count_total': domestic_mismatched_count_total,
            'import_mismatched_count_total': import_mismatched_count_total,
        }

    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        return {'error': f'Error processing Excel file: {str(e)}'}
@login_required
def match_excel_with_database(request):
    if request.method == 'GET':
        return render(request, 'testapp/upload_excel.html')
    elif request.method == 'POST':
        result = run_comparison()
        if 'error' in result:
            return render(request, 'testapp/upload_excel.html', {'error': result['error']})
        else:
            return render(request, 'testapp/compare_result.html', result)

def compare_row_with_invoice(row, invoice, invoice_type, mapped_columns):
    """Compare a row from Excel with a database invoice using mapped columns."""
    mismatches = []
 
    def display_value(val, field_name=None):
        norm = normalize_value(val)
        # Always show 'empty' for missing/empty values
        if not norm:
            return 'empty'
        # For email fields, ensure '@' is present if it was in the original value
        if field_name and 'email' in field_name.lower():
            val_str = str(val) if val is not None else ''
            if '@' in val_str:
                # Return the original value, lowercased, with all whitespace trimmed
                return val_str.strip().lower()
            else:
                # If not present, return normalized value
                return norm
        return norm
 
    # Special handling for email_ids: collect all Excel columns mapped to email_ids (flatten if needed)
    email_excel_columns = []
    for db_col, col in mapped_columns.items():
        if db_col == 'email_ids':
            if isinstance(col, list):
                email_excel_columns.extend(col)
            else:
                email_excel_columns.append(col)
    processed_email_ids = False
 
    # Special handling for mobile_landline_numbers: collect all Excel columns mapped to mobile_landline_numbers (flatten if needed)
    mobile_excel_columns = []
    for db_col, col in mapped_columns.items():
        if db_col == 'mobile_landline_numbers':
            if isinstance(col, list):
                mobile_excel_columns.extend(col)
            else:
                mobile_excel_columns.append(col)
    processed_mobile_numbers = False
 
    for db_column, excel_column in mapped_columns.items():
        # Skip vendor_name since it's already matched
        if db_column.lower() == 'vendor_name':
            continue
 
        # Special logic for email_ids
        if db_column == 'email_ids' and not processed_email_ids:
            db_email = getattr(invoice, 'email_ids', None)
            db_email_norm = display_value(db_email, 'email_ids')
            # Collect all Excel email values
            excel_emails = [display_value(row.get(col, ''), 'email_ids') for col in email_excel_columns]
            # Remove empty values
            excel_emails = [e for e in excel_emails if e != 'empty']
            # Check if db_email is present in any of the Excel emails
            if db_email_norm in excel_emails:
                # Matched, do nothing
                pass
            else:
                # Mismatch: show all excel emails as the excel_value
                mismatches.append({
                    'field': 'email_ids',
                    'excel_value': ', '.join(excel_emails) if excel_emails else 'empty',
                    'db_value': db_email_norm,
                    'invoice_type': invoice_type,
                    'invoice_number': invoice.invoice_number
                })
            processed_email_ids = True
            continue
 
        # Special logic for mobile_landline_numbers
        if db_column == 'mobile_landline_numbers' and not processed_mobile_numbers:
            db_mobile = getattr(invoice, 'mobile_landline_numbers', None)
            db_mobile_norm = display_value(db_mobile, 'mobile_landline_numbers')
            # Collect all Excel mobile/landline values
            excel_mobiles = [display_value(row.get(col, ''), 'mobile_landline_numbers') for col in mobile_excel_columns]
            # Remove empty values
            excel_mobiles = [e for e in excel_mobiles if e != 'empty']
            # Check if db_mobile is present in any of the Excel mobiles
            if db_mobile_norm in excel_mobiles:
                # Matched, do nothing
                pass
            else:
                # Mismatch: show all excel mobiles as the excel_value
                mismatches.append({
                    'field': 'mobile_landline_numbers',
                    'excel_value': ', '.join(excel_mobiles) if excel_mobiles else 'empty',
                    'db_value': db_mobile_norm,
                    'invoice_type': invoice_type,
                    'invoice_number': invoice.invoice_number
                })
            processed_mobile_numbers = True
            continue
 
        # If excel_column is a list (multi-mapped field), skip default logic
        if isinstance(excel_column, list):
            continue
 
        # Special logic for bank_account_number: allow match if any db value matches excel value
        if db_column == 'bank_account_number':
            excel_raw_value = row.get(excel_column, '')
            excel_display_value = display_value(excel_raw_value, db_column)
            db_raw_value = getattr(invoice, db_column, None)
            # If db value is a string with multiple account numbers separated by comma/semicolon/space
            db_values = []
            if db_raw_value is not None:
                if isinstance(db_raw_value, str):
                    # Split on comma, semicolon, or whitespace
                    db_values = [v.strip() for v in re.split(r'[;,\s]+', db_raw_value) if v.strip()]
                else:
                    db_values = [str(db_raw_value)]
            db_display_values = [display_value(v, db_column) for v in db_values if v]
            # If any db_display_value matches excel_display_value, treat as match
            if excel_display_value != 'empty' and db_display_values:
                if any(excel_display_value == dbv for dbv in db_display_values):
                    continue  # Matched, skip mismatch
                else:
                    # If both are empty, skip (treat as match)
                    if all(dbv == 'empty' for dbv in db_display_values) and excel_display_value == 'empty':
                        continue
                    mismatches.append({
                        'field': db_column,
                        'excel_value': excel_display_value,
                        'db_value': ', '.join(db_display_values),
                        'invoice_type': invoice_type,
                        'invoice_number': invoice.invoice_number
                    })
            else:
                # Fallback to original logic if no db values
                if excel_display_value != display_value(db_raw_value, db_column):
                    if excel_display_value == 'empty' and display_value(db_raw_value, db_column) == 'empty':
                        continue
                    mismatches.append({
                        'field': db_column,
                        'excel_value': excel_display_value,
                        'db_value': display_value(db_raw_value, db_column),
                        'invoice_type': invoice_type,
                        'invoice_number': invoice.invoice_number
                    })
            continue
 
        # Default logic for other fields
        excel_raw_value = row.get(excel_column, '')
        db_raw_value = getattr(invoice, db_column, None)
        excel_display_value = display_value(excel_raw_value, db_column)
        db_display_value = display_value(db_raw_value, db_column)
        # Only flag as mismatch if values are not equal and both are not empty
        if excel_display_value != db_display_value:
            # If both are empty, skip (treat as match)
            if excel_display_value == 'empty' and db_display_value == 'empty':
                continue
            mismatches.append({
                'field': db_column,
                'excel_value': excel_display_value,
                'db_value': db_display_value,
                'invoice_type': invoice_type,
                'invoice_number': invoice.invoice_number
            })
    return {
        'vendor_name': invoice.vendor_name,
        'invoice_type': invoice_type,
        'invoice_number': invoice.invoice_number,
        'mismatches': mismatches
    }
 

# Role-based login view
def role_based_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None and user.role == 'admin':
            login(request, user)
            return HttpResponseRedirect(reverse('admin_dashboard'))
        else:
            return render(request, 'testapp/login.html', {'error': 'Invalid credentials or not authorized'})
    return render(request, 'testapp/login.html')

# Role-specific dashboards
class AdminDashboardView(LoginRequiredMixin, UserPassesTestMixin, TemplateView):
    template_name = 'testapp/admin_dashboard.html'

    def test_func(self):
        return self.request.user.role == 'admin'

class ManagerDashboardView(LoginRequiredMixin, UserPassesTestMixin, TemplateView):
    template_name = 'testapp/manager_dashboard.html'

    def test_func(self):
        return self.request.user.role == 'manager'

class UserDashboardView(LoginRequiredMixin, UserPassesTestMixin, TemplateView):
    template_name = 'testapp/user_dashboard.html'

    def test_func(self):
        return self.request.user.role == 'user'

# Role-based access decorator example
def role_required(allowed_roles=[], task_name="this task"):
    def decorator(view_func):
        def _wrapped_view(request, *args, **kwargs):
            if request.user.role in allowed_roles:
                return view_func(request, *args, **kwargs)
            else:
                return render(request, 'testapp/access_denied.html', {'task_name': task_name})
        return _wrapped_view
    return decorator

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def validate_environment_variables():
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", 
                    "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

endpoint = os.environ.get('AZURE_OCR_ENDPOINT')
key = os.environ.get('AZURE_OCR_API_KEY')

import time
import random

def azure_ocr_extract_text(pdf_path):
    try:
        if not key or not endpoint:
            logger.error("Azure OCR credentials are not set in environment variables.")
            return ""
        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

        with open(pdf_path, "rb") as pdf:
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-document", pdf)
            result = poller.result()
        extracted_text = result.content
        return extracted_text

    except Exception as e:
        logger.error(f"Azure OCR request failed: {e}")
        return ""
    return ""

# def identify_pdf_type(text, country=None):
#     if country and country.strip().lower() == "india":
#         return "domestic"
    
#     if text:
#         lower_text = text.lower()
#         if any(keyword in lower_text for keyword in ["incoterms","swift code","bic code","iban number"]):
#             return "domestic"
    
#     return "import"

def identify_pdf_type_by_folder(pdf_path):
    # Extract the folder name from the file path
    folder_name = os.path.basename(os.path.dirname(pdf_path))
 
    # Determine the PDF type based on the folder name
    if folder_name.lower() == "domestic":
        return "domestic"
    elif folder_name.lower() == "import":
        return "import"
 
    # Default to "unknown" if the folder name doesn't match
    return "unknown"
   
def generate_prompt(pdf_type, text=None, is_image=False, custom_prompt=None):
    base_prompt = (
        "You are an expert at analyzing {} invoice documents{}.\n"
        "Extract ONLY the following details if they exist in the document:\n"
         "1. Only the specified fields are extracted.\n"
        "2. Follow OCR text and extract the OCR text data.\n"
        "3. Every time check the country in OCR and follow that OCR data.\n"
        "4. If a field is not present in the text, leave it empty.\n"
        "5. Do not infer or include any irrelevant or non-explicitly mentioned data.\n"
        "6. Provide the output strictly in the JSON format specified.\n"

        "7. DO NOT SEPARATE any extracted values with commas(,) implicitly.\n"
        "8. Only the vendor's details are extracted. Do NOT gather any Biological E (BE) related data, such as BE email, BE PAN number, BE Bank details or any other BE-specific information.\n"
        "9. When extracting bank details, ignore headings like 'Moldev Bank' as actual bank names. The correct bank name should be explicitly stated. If the actual bank name is not mentioned, leave it blank.\n"
        "10. If email does not relate to BE, extract that email.\n"
        "11. If a field has no data in the document, set its value to 'Empty'.\n"
        "12. Format the output as a JSON object with the exact field names shown above.\n"
        "13. Do not include any additional fields beyond the list provided.\n"
        "14. If a mandatory field like Vendor Name cannot be found, add a note or warning for review instead of proceeding silently.\n"
        "15. For \"Invoice Number\", extract the value exactly as it appears in the document, including all special characters such as slashes (/), dashes (-), and spaces. Do not remove or alter any characters.\n"
    )
    
   
    if pdf_type == "import":
        fields = ["Invoice Number", "Invoice Date", "PO No", "Invoice Amount", "IRN No", "GRN No", "GRN Date", "Description of Goods or Services",
                  "Basic Amount", "CGST", "SGST", "IGST", "Total Invoice Value",
                  "Vendor Name", "PAN Number", "ECC Number",
                  "Origin Country", "Port of Loading", "Port of Discharge",
                  "HSN Code", "Customs Declaration Number", "Bill of Lading Number",
                  "Street/HouseNo.", "Street", "Building",
                  "City", "District", "State",
                  "Country", "PO Box",
                  "Email Id's", "Mobile/Landline numbers", "FAX No.",
                  "Region", "Country Code",
                  "Bank account name/Beneficiary Name", "Bank Account Number",
                  "IBAN Number", "Bank Name", "SWIFT Code", "BIC Code",
                  "SORT Code", "Bank Key", "Bank Address", "Country",
                  "Invoice Currency", "Payment Terms"]
    else:  # Domestic
        fields = ["Invoice Number", "Invoice Amount", "Invoice Date", "PO No", "IRN No", "GRN No", "GRN Date", "Description of Goods or Services", "Basic Amount", "CGST", "SGST", "IGST", "Total Invoice Value",
                  "Vendor Name", "PAN Number", "GSTIN Number",
                  "Udyog AADHAR Registration Number (MSME)",
                  "Udyog AADHAR Certificate date (MSME)", "ECC Number",
                  "AADHAR(Indv.)",
                  "Street/HouseNo.", "Street", "Building",
                  "City", "District", "State",
                  "Country", "PO Box",
                  "Email Id's", "Mobile/Landline numbers", "FAX No.",
                  "Region", "Country Code",
                  "Bank account name/Beneficiary Name", "Bank Account Number",
                  "Bank Name", "IFSC Code", "Bank Address", "Country",
                  "Payment Terms"]
 
    prompt = base_prompt.format(
        pdf_type,
        " from images" if is_image else ""
    ) + "\n".join(f"• {field}" for field in fields) + (
        "\n\nRules:\n"
        "1. Only the specified fields are extracted.\n"
        "2. Follow OCR text and extract the OCR text data.\n"
        "3. Every time check the country in OCR and follow that OCR data.\n"
        "4. If a field is not present in the text, leave it empty.\n"
        "5. Do not infer or include any irrelevant or non-explicitly mentioned data.\n"
        "6. Provide the output strictly in the JSON format specified.\n"

        "7. DO NOT SEPARATE any extracted values with commas(,) implicitly.\n"
        "8. Only the vendor's details are extracted. Do NOT gather any Biological E (BE) related data, such as BE email, BE PAN number, BE Bank details or any other BE-specific information.\n"
        "9. When extracting bank details, ignore headings like 'Moldev Bank' as actual bank names. The correct bank name should be explicitly stated. If the actual bank name is not mentioned, leave it blank.\n"
        "10. If email does not relate to BE, extract that email.\n"
        "11. If a field has no data in the document, set its value to 'Empty'.\n"
        "12. Format the output as a JSON object with the exact field names shown above.\n"
        "13. Do not include any additional fields beyond the list provided.\n"
        "14. If a mandatory field like Vendor Name cannot be found, add a note or warning for review instead of proceeding silently.\n"
    )
 
 

    if text and not is_image:
        prompt += f"\nDocument text:\n{text}\n"

    # Log the generated prompt for debugging
    logger.info(f"Generated prompt: {prompt[:500]}...")  # Log first 500 chars

    return prompt

REQUIRED_FIELDS_DOMESTIC = [
    "Invoice Number", "Invoice Amount", "Invoice Date", "PO No", "IRN No", "GRN No", "GRN Date", "Description of Goods or Services", "Basic Amount", "CGST", "SGST", "IGST", "Total Invoice Value",
    "Vendor Name", "PAN Number", "GSTIN Number","Street/House No"
    "Udyog AADHAR Registration Number (MSME)",
    "Udyog AADHAR Certificate date (MSME)", "ECC Number",
    "AADHAR(Indv.)",
    "Street/HouseNo.", "Street", "Building",
    "City", "District", "State",
    "Country", "PO Box",
    "Email Id's", "Mobile/Landline numbers", "FAX No.",
    "Region", "Country Code",
    "Bank account name/Beneficiary Name", "Bank Account Number",
    "Bank Name", "IFSC Code", "Bank Address", "Country",
    "Payment Terms","path"
]

REQUIRED_FIELDS_IMPORT = [
    "Invoice Number", "Invoice Date", "PO No", "Invoice Amount", "IRN No", "GRN No", "GRN Date", "Description of Goods or Services",
    "Basic Amount", "CGST", "SGST", "IGST", "Total Invoice Value","Street/House No"
    "Vendor Name", "PAN Number", "ECC Number",
    "Origin Country", "Port of Loading", "Port of Discharge",
    "HSN Code", "Customs Declaration Number", "Bill of Lading Number",
    "Street/HouseNo.", "Street", "Building", 
    "City", "District", "State", 
    "Country", "PO Box",
    "Email Id's", "Mobile/Landline numbers", "FAX No.", 
    "Region", "Country Code",
    "Bank account name/Beneficiary Name", "Bank Account Number",
    "IBAN Number", "Bank Name", "SWIFT Code", "BIC Code",
    "SORT Code", "Bank Key", "Bank Address", "Country","Routing/Intermediate Bank Account Number",
    "Routing/Intermediate Bank Name","Routing/Intermediate SWIFT Code",
    "Routing/Intermediate BIC Code","Routing/Intermediate SORT Code",
    "Invoice Currency", "Payment Terms","path"
]
 
def remove_text_before_marker(text, marker):
    index = text.find(marker)
    if index != -1:
        return text[index + len(marker):]
    return text


def remove_string(text, string_to_remove):
    if string_to_remove in text:
        text = text.replace(string_to_remove, "")
    return text

def process_pdf_and_extract_data(pdf_file, custom_prompt=None, dpi=300, use_cache=True):
    cache_key = f"extracted_data_{pdf_file}_{custom_prompt}"
    if use_cache:
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached extraction result.")
            return cached_result

    """Process PDF and extract data based on type (domestic/import) and custom prompt."""
    validate_environment_variables()

    client = AzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
    )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    if hasattr(pdf_file, 'chunks'):
        for chunk in pdf_file.chunks():
            temp_file.write(chunk)
    else:
        with open(pdf_file, 'rb') as f:
            temp_file.write(f.read())
    temp_file.close()

    # Extract text using OCR first if necessary
    full_text = azure_ocr_extract_text(pdf_file)

    # Identify PDF type based on the extracted text
    pdf_type = identify_pdf_type_by_folder(pdf_file) if pdf_file else "import"
    if pdf_type == "unknown":
        pdf_type = "import"

    if full_text.strip():
        print("Entered in actula prompt")
        # enhanced_prompt = generate_prompt(pdf_type, full_text, False, custom_prompt)
        initial_prompt = ""
        domestic_prompt = """
                        Extract the following vendor-related fields from the above text. The data is extracted from **invoice documents** and pertains specifically to **vendor details, addresses, contact information, bank details, and payment terms**.  
                    
                
                    ### Strict Extraction Rules:  
                
                    #### 1. **Mandatory Fields (Must Not Be Skipped):**  
                    - **Vendor Name:**  
                        - It is mandatory to extract the **Vendor Name** as it is always present in the invoice text.  
                        - Perform a thorough search in the text for terms like "Vendor Name," "Supplier," or similar identifiers.
                        - **Critical Rule:** If the Vendor Name contains unusual symbols (e.g., "$" instead of "S"), always cross-check with the OCR data to ensure that the vendor name is accurately reflected, e.g., "S S printpack" instead of "S $ printpack." If a discrepancy is found, **always prioritize the OCR data** and correct the vendor name accordingly.
                        - If the Vendor Name is not extracted, flag it as a critical error.  
                        - **Bank Account Number**(will always be present in the invoice text)
                            - **Bank Name**(will always be present in the invoice text)
                            - **PAN Number** (only if present in the invoice text)  
                            - **GSTIN Number**
                            - **When extracting GSTIN and PAN, ensure that variations like "GSTIN NO", "GST NO", "GSTIN Number", "PAN No", "PAN Number", or similar phrases are also matched. Do not restrict the search to only exact terms like "GSTIN" or "PAN".**

                    #### 2. **Comprehensive Extraction:**  
                    - For all other fields, ensure no present data is skipped. If a field exists in the text, it must be included in the output.  
                
                    #### 3. **Exclude Biological E Limited Data (Must be skipped):**  
                        - **Do NOT extract or include any Biological E Limited related data**. This includes, but is not limited to:  
                        - BE PAN number or Biological E Limited PAN Number (e.g, PAN: AAACB7XXXX which is BE's you have to ignore)
                        - BE GSTIN number  or Biological E Limited GSTIN number
                        - BE email addresses (e.g., be@email.com)  
                        - Any BE-related address, contact, or bank information  
                        - If any BE-specific or Biological E Limited data is found in the text, **skip these fields completely** and do not include them in the output.  
                        - The tool should **explicitly filter out** any instances of BE-related data during the extraction process, flagging them as “excluded” rather than attempting to extract or store them.
            
                    #### 4. **Empty Fields:**  
                        - If a field is not present in the text, leave it as an empty string in the output JSON.  
                
                    #### 5. **Validation for Missing Vendor Name:**  
                        - If the Vendor Name field is empty in the output JSON, the tool must flag this as an error for immediate review. 

                    #### 6. **Wrong PO number Extraction:**
                        - Do NOT extract from headings like 'Order No', 'Buyer's Order No.', or 'Sales Order'.
                        - It must be under the VENDOR's section only.

                    #### 7. **Wrong IRN No Extraction:**
                        - Consider all IRN-related formats, including "IRN", "Irn", "IRN NO", "IRN Number", with or without a colon (e.g., "IRN:", "IRN NO:", "IRN Number:"). Treat all these variations as valid references to the IRN.
                        - Do not extract data from the 'EINV AckNo' heading.
                    
                    8. - Extract only the actual CGST, SGST, and IGST amounts if explicitly mentioned; ignore percentages and leave the field blank if no amount is provided.
                    9. - Extract the Country Code only if it is explicitly provided; otherwise, leave the field blank.
                    #### 10. **GRN Date Extraction:**
                       - Capture the GRN date related to the GRN details; this may include cases where it appears as "Signature & Date" or explicitly as "GRN Date" but exclude any dates labeled as "Date of Receipt."
            
                    JSON Output Template:
                    {
                        "Vendor Name": "",  
                        "PAN Number": "",  
                        "GSTIN Number": "",  
                        "Udyog AADHAR Registration Number (MSME)": "",  
                        "Udyog AADHAR Certificate Date (MSME)": "",  
                        "ECC Number": "",  
                        "AADHAR (Individual)": "",  
                        "Street/HouseNo.": "",  
                        "Street": "",  
                        "Building": "",  
                        "City": "",  
                        "District": "",  
                        "State"F: "",  
                        "Country": "",  
                        "PO No": "",
                        "PO Box": "",  
                        "Email Id's": "",  
                        "Mobile/Landline numbers": "",  
                        "FAX No.": "",  
                        "Region": "",  
                        "Country Code": "",  
                        "Bank Account Name/Beneficiary Name": "", 
                        "Bank Account Number":"",
                        
                        "Bank Name": "", 
                        "IFSC Code": "",  
                        "SWIFT Code":"",
                        "IBAN Number":"",
                        "Bank Address": "",  
                        "Bank Country": "",  
                        "Payment Terms": "",
                        "Invoice Number":"",  
                        "Invoice Date":"",
                        "Invoice Currency":"",
                        "HSN Code":"",
                        "IRN No":"",
                        "GRN No":"",
                        "GRN Date":"",
                        "Description of Goods or Services":"",
                        "Basic Amount":"",
                        "CGST":"",
                        "SGST":"",
                        "IGST":"",
                        "Total Invoice Value":""
                    }
            
                
                
                
                    ### Additional Extraction Guidelines:  
                
                    1. **Prioritize the Vendor Name Extraction:**  
                    - Search for common terms like "Vendor Name," "Supplier Name," or "Issued By" in the text.  
                    - Use text context, sections, and formatting to identify the vendor's name accurately.  
                    - **If a discrepancy is found between the extracted vendor name (e.g., "S $ printpack") and the OCR data (e.g., "S S printpack"), always prioritize the OCR data.**  
                
                
                    2. **Prevent Customer Data Inclusion:**  
                    - **Do not include any Biological E (BE)** related data such as BE email, BE PAN number, or any other BE-specific information in the extracted fields.    
                
                    3. **Validation of Critical Fields:**  
                    - After extraction, verify that mandatory fields, such as **Vendor Name**, are not empty.  
                    - If empty, flag the issue for review as a high-priority error.  
                
                    4. **Error Flagging:**  
                    - If a mandatory field like **Vendor Name** cannot be found, add a note or warning for review instead of proceeding silently.  
            
            """
        import_prompt = f"""
                            Extract the following fields related to **vendor, address, contact, and banking information** from the provided text. The data is extracted from **invoice documents** and pertains to key vendor details, address, contact information, banking details (including routing/intermediate bank information), invoice currency, and payment terms. Ensure:  
                        1. Only the specified fields are extracted.  
                        2. Follow OCR text and Extract the OCR text data.
                        3. Every time check the country in ocr and follow that ocr data.
                        3. If a field is not present in the text, leave it empty.  
                        4. Do not infer or include any irrelevant or non-explicitly mentioned data.  
                        5. Provide the output strictly in the JSON format specified below.
                        6. While extracting BANK details please note that some times there will be two account numbers and IBAN numbers based on 'EUR' or 'USD'. Segregate these bank accounts and IBAN numbers based on 'EUR' or 'USD' category. If there is only one account number map it to correct category either 'EUR' or 'USD' that has encountered.
                        7. DO NOT SEPARATE any extracted values with commas(,) implicitly. 
                        8. - **When extracting GSTIN and PAN, ensure that variations like "GSTIN NO", "GST NO", "GSTIN Number", "PAN No", "PAN Number", or similar phrases are also matched. Do not restrict the search to only exact terms like "GSTIN" or "PAN".**
                        9. Only the **vendor's details** are extracted. Do **NOT** gather any **Biological E (BE)** related data, such as BE email, BE PAN number, BE Bank details or any other BE-specific information.
                        10.When extracting **bank details**, note that sometimes there may be headings or multiple sections containing names like "Moldev Bank" which are **headings** and should not be considered as the actual bank name. 
                        - Ensure that headings like **"Moldev Bank"** (or any similar term) are **not considered as the bank name**. The correct bank name should be explicitly stated as part of the bank's details, such as **"Bank of America"**. If the actual bank name is not mentioned, leave it blank.
                        11.Only the **vendor's details** are extracted. Do **NOT** gather any **Biological E (BE)** related data, such as BE email, BE PAN number, BE Bank details, or any other BE-specific information. **IGNORE** any BE PAN numbers like AAECB4426E, including those in **instructions** sections or other parts of the document that are not explicitly part of the vendor's details. If a **BE PAN number** is found outside the vendor details section, **REMOVE** that BE PAN number from the extraction entirely.
                        
                        12.- If email does not related to BE extract that email.
                        13. - **When extracting PO No, ensure that variations like "Customer PO" or similar phrases are also matched.**
                        14. - Exclude PO numbers if they are related to Biological E Limited; only extract PO numbers that are explicitly associated with the vendor, not with Biological E.
                        15. - SWIFT Code and BIC Code should be extracted separately without confusion; ensure the SWIFT Code is not mistakenly captured as BIC, and the BIC Code is not captured as SWIFT. 
                        16. GRN Date Extraction:
                            Capture the GRN date only if it is clearly associated with GRN-related details. This includes flexible matches such as:
                                - "GRN Date", "grn date", "GRN Date:", or "grn date:"
                                -  Phrases like "Signature & Date", "Signed & Dated", "Authorized Signature & Date" (case-insensitive, with or without colons)

                            Strictly exclude any dates labeled as "Date of Receipt", including all case and punctuation variations such as:
                                - "Date of Receipt", "date of receipt", "DATE OF RECEIPT"
                                - "Date of Receipt:", "date of receipt:", "DATE OF RECEIPT:"
                            - Capture the GRN date related to the GRN details; this may include flexible cases such as text containing "Signature & Date", "Signed & Dated", or explicitly labeled "GRN Date" (with or without a colon, and case-insensitive, e.g., "GRN Date", "grn date:", "GRN date:"). Strictly exclude any dates labeled as "Date of Receipt", including all possible variations such as "Date of Receipt", "Date of Receipt:", "date of receipt", "date of receipt:", "DATE OF RECEIPT", or "DATE OF RECEIPT:" — under no circumstances should these be considered for GRN date extraction.
                        17. GRN No Extraction:
                            - Capture the GRN NO related to the GRN details; this may include cases where it appears as "GRN No:" or explicitly as "GRN NO".
                        18. PO No Extraction:
                            - Extract the PO number if the heading contains any variation such as 'PO No', 'PO Number', or 'Purchase Order Number' (case-insensitive and allowing minor differences or additional words); ensure flexible matching to capture all valid PO references,but exclude any PO numbers related to Biological E.
                        19. HSN Code Extraction:
                            - Sometimes HSN Code is taking from Commodity value but dont take from Commodity.
                        

                        
                        ### Source of Data:  
                        The input text originates from **invoice documents**, which typically include information about the vendor, their address, contact details, bank account, payment terms, and routing/intermediate bank details.  
                        
                        ### Fields to Extract:  
                        
                        #### Vendor Details (captured from the vendor section in the invoice):  
                        - Vendor Name  
                        - PAN Number  
                        - ECC Number  
                        
                        #### Address Details (captured from the vendor’s address section in the invoice):  
                        - Street/House No.  
                        - Street  
                        - Building  
                        - City  
                        - District  
                        - State  
                        - Country  
                        - PO Box  
                        
                        #### Contact Details (captured from the vendor’s contact section in the invoice):  
                        - Email IDs  
                        - Mobile/Landline numbers  
                        - FAX Number  
                        - Region  
                        - Country Code  
                        
                        #### Bank Details (captured from the bank/payment section in the invoice):  
                        - Bank Account Name/Beneficiary Name  
                        - Bank Account Number   
                        - IBAN Number
                        - Bank Name  
                        - SWIFT Code  
                        - BIC Code  
                        - SORT Code  
                        - Bank Key  
                        - Bank Address  
                        - Country  
                        
                        #### Routing/Intermediate Bank Details:  
                        - Routing/Intermediate Bank Account Number  
                        - Routing/Intermediate Bank Name  
                        - Routing/Intermediate SWIFT Code  
                        - Routing/Intermediate BIC Code  
                        - Routing/Intermediate SORT Code  
                        
                        #### Invoice Currency (captured from the currency section in the invoice):  
                        - Invoice Currency  
                        
                        #### Payment Terms (captured from the payment section or terms section in the invoice):  
                        - Payment Terms  
                        
                        
                        
                        ### JSON Output Format:  
                        
                        {{  
                            "Vendor Name": "",  
                            "PAN Number": "",  
                            "ECC Number": ""  , 
                            "Street/HouseNo.": "",  
                            "Street": "",  
                            "Building": "",  
                            "City": "",  
                            "District": "",  
                            "State": "",  
                            "Country": "",
                            "PO No": "",  
                            "PO Box": ""  ,
                            "Email Id's": "",  
                            "Mobile/Landline numbers": "",  
                            "FAX No.": "",  
                            "Region": "",  
                            "Country Code": ""   
                            "Bank Account Name/Beneficiary Name": "",  
                            "Bank Account Number":"",
                            "Bank Name": "", 
                            "IFSC Code": "", 
                            "IBAN Number": "",  
                            "GSTIN Number": "",
                            "SWIFT Code": "",  
                            "BIC Code": "",  
                            "SORT Code": "",  
                            "Bank Key": "",  
                            "Bank Address": "",  
                            "Country": "",  
                            "Routing/Intermediate Bank Account number": "",  
                            "Routing/Intermediate Bank name": "",  
                            "Routing/Intermediate SWIFT Code": "",  
                            "Routing/Intermediate BIC Code": "",  
                            "Routing/Intermediate SORT Code": "" ,
                            "Invoice Currency": "",  
                            "Payment Terms": "",
                            "Invoice Number":"",  
                            "Invoice Date":"",
                            "HSN Code":"",
                            "IRN No":"",
                            "GRN No":"",
                            "GRN Date":"",
                            "Description of Goods or Services":"",
                            "Basic Amount":"",
                            "CGST":"",
                            "SGST":"",
                            "IGST":"",
                            "Total Invoice Value":""  
                        }}
                        
                        
                        ### Notes:  
                        - The input text is **extracted from invoice documents** and should include information explicitly related to the specified fields and also cross check with OCR text take data from the OCR text.
                        - **Do not include any Biological E (BE)** related data such as BE email, BE PAN number, BE-Bank details or any other BE-specific information in the extracted fields.    
                        - Leave any field blank if the information is not explicitly present in the text.  
                        - Ensure extracted data aligns correctly with its respective field, without any assumptions.  
                        - If "Moldev Bank" or similar terms appear, treat them as headings and **ignore them** when identifying the actual **bank name**.
                        - If the actual bank name is not present, leave the **Bank Name** field blank.
                        - **IGNORE any BE PAN numbers** found in instruction sections or elsewhere in the document. They should **NOT** be included in the extracted data.  
                    

                        """
        if pdf_type == "import":
          initial_prompt = import_prompt
          logger.info("It is related to Import pdfs.")
        else:
          initial_prompt = domestic_prompt
          logger.info("It is related to Domestic pdfs.")
        initial_prompt += f"\nDocument text:\n{full_text}\n"
        enhanced_prompt=initial_prompt
        try:
            logger.info(f"Sending prompt to OpenAI: {enhanced_prompt[:500]}...")  # Log first 500 chars
            
            logger.debug(f"Full enhanced prompt (first 500 chars): {enhanced_prompt[:500]}")
            completion = client.chat.completions.create(
                model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[{"role": "user", "content": enhanced_prompt}]
            )
            raw_response = completion.choices[0].message.content
            logger.info(f"Received raw response from OpenAI: {raw_response}")  # Log full response
            

            # Post-process the response
            logger.info("Attempting to clean and parse the AI response.")
    
            # Post-process the response to ensure all fields are included
            #cleaned_response = raw_response.strip()
            #cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            #cleaned_response = cleaned_response.replace('*', '').replace('/', '').strip()
            # structured_data1 = remove_text_before_marker(raw_response, "```json")
            # logger.info(f"After removing text before marker: {structured_data1}")
            # structured_data1 = remove_string(raw_response, "```")
            # logger.info(f"After removing code block markers: {structured_data1}")
            json_data = clean_and_parse_json_response(raw_response)
            
            if not json_data:
                raise ValueError("Failed to parse AI response as valid JSON")
                
            logger.info("JSON loaded successfully.")
            logger.info("After clearing:",json_data) 
            structured_data1 = remove_text_before_marker(raw_response, "```json")
            logger.info(f"After removing text before marker: {structured_data1}")

            # Correct usage: remove backticks from the already-trimmed string
            structured_data1 = remove_string(structured_data1, "```")
            logger.info(f"After removing code block markers: {structured_data1}")
            



            # Use this instead of the earlier method
            structured_data1 = extract_json_from_response(raw_response)
            logger.info(f"After doing function:{structured_data1}")
           
            logger.info("Attempting to load response as JSON.")
            try:
                #json_data = json.loads(cleaned_response)
                json_data = json.loads(structured_data1)
                
                logger.info("JSON loaded successfully after entering into Try block.")
                logger.debug(f"Loaded JSON: {json_data}")
                # Assign REQUIRED_FIELDS based on pdf_type
                if pdf_type == "domestic":
                    logger.debug("PDF type is domestic.")
                    REQUIRED_FIELDS = REQUIRED_FIELDS_DOMESTIC
                else:
                    logger.debug("PDF type is import.")
                    REQUIRED_FIELDS = REQUIRED_FIELDS_IMPORT

                for date_field in ['invoice_date', 'grn_date']:
                    if date_field in json_data:
                        parsed_date = parse_date(json_data[date_field])
                        if parsed_date:
                            json_data[date_field] = parsed_date.strftime('%Y-%m-%d')
                        else:
                            json_data[date_field] = None

                # Add missing fields with default value "Empty"
                for field in REQUIRED_FIELDS:
                    if field not in json_data:
                        json_data[field] = "Empty"

                # Convert the dictionary to a JSON string before saving or processing
                json_result = json.dumps(json_data, indent=2)
                if use_cache:
                    cache.set(cache_key, json_result, timeout=3600)  # Cache for 1 hour
                return json_result
            except json.JSONDecodeError:
                logger.error("Failed to parse AI response as JSON2.")
            cleaned_response = raw_response.strip()
            cleaned_response = raw_response.strip()

        except Exception as e:
            os.unlink(temp_file.name)
            logger.error(f"OpenAI processing failed: {str(e)}")
            return json.dumps({"error": f"OpenAI processing failed: {str(e)}"})
    else:
        images = convert_from_path(temp_file.name, dpi=dpi)
        image_base64_list = [encode_image(img) for img in images]

        enhanced_prompt = generate_prompt(pdf_type, None, True, custom_prompt)

        try:
            logger.info(f"Sending image prompt to OpenAI: {enhanced_prompt[:500]}...")
            completion = client.chat.completions.create(
                model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": enhanced_prompt}] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                        for img in image_base64_list
                    ]
                }]
            )
            raw_response = completion.choices[0].message.content
            logger.info(f"Received raw response from OpenAI (image): {raw_response[:500]}...")
        except Exception as e:
            os.unlink(temp_file.name)
            logger.error(f"OpenAI processing failed: {str(e)}")
            return json.dumps({"error": f"OpenAI processing failed: {str(e)}"})

    os.unlink(temp_file.name)

    cleaned_response = raw_response.strip()
    cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
    cleaned_response = cleaned_response.replace('*', '').replace('/', '').strip()

    try:
        json_data = json.loads(cleaned_response)
        json_result = json.dumps(json_data, indent=2)
        if use_cache:
            cache.set(cache_key, json_result, timeout=3600)  # Cache for 1 hour
        return json_result
    except json.JSONDecodeError:
        try:
            pairs = cleaned_response.split('\n')
            result = {}
            for pair in pairs:
                if ':' in pair:
                    key, value = map(str.strip, pair.split(':', 1))
                    result[key] = value
            json_result = json.dumps(result, indent=2)
            if use_cache:
                cache.set(cache_key, json_result, timeout=3600)  # Cache for 1 hour
            return json_result
        except Exception as parse_exc:
            logger.error(f"Failed to parse AI response3: {parse_exc}")
            json_result = json.dumps({
                "raw_response": cleaned_response,
                "message": "Response could not be parsed as JSON"
            }, indent=2)
            if use_cache:
                cache.set(cache_key, json_result, timeout=3600)  # Cache for 1 hour
            return json_result

@api_view(['POST'])
@role_required(allowed_roles=['admin', 'manager'], task_name="Extract Single pdf")
def extract_invoice_data(request):
    file = request.FILES.get('pdf')

    if not file:
        return Response({'error': 'No PDF uploaded'}, status=400)

    try:
        # Process the PDF to extract data automatically
        extracted_data = process_pdf_and_extract_data(file)
        
        try:
            # Try to load the extracted data as JSON
            structured_data = json.loads(extracted_data)

            # Determine pdf_type based on folder or default to 'domestic'
            pdf_type = identify_pdf_type_by_folder(file.name)
            if pdf_type == "unknown":
                pdf_type = "domestic"

            # Save extracted data to database
            save_result = save_extracted_data(pdf_type, extracted_data)
            if "error" in save_result:
                logger.error(f"Error saving extracted data: {save_result['error']}")
                return Response({'error': f"Data extraction succeeded but saving failed: {save_result['error']}"}, status=500)

            return Response({
                "message": "Data extracted and saved successfully",
                "data": structured_data,
                "save_result": save_result
            })
        except json.JSONDecodeError:
            return Response({'error': 'Failed to decode extracted data'}, status=400)

    except Exception as e:
        # Log the error and return an error message to the client
        logger.error(f"Error in extracting data: {e}")
        return Response({'error': str(e)}, status=500)

@role_required(allowed_roles=['admin', 'manager'], task_name="extract directory")
@login_required
def process_directory_view(request):
    data = {}
    # Load environment variables from .env file
    load_dotenv()

    # Replace hardcoded path with dynamic environment variable
    predefined_directory_path = os.getenv("PDF_DIRECTORY")

    # Ensure the environment variable is set
    if not predefined_directory_path:
        raise EnvironmentError("The PDF_DIRECTORY environment variable is not set in the .env file.")

    if request.method == "POST":
        pdf_files = list(Path(predefined_directory_path).glob('*.pdf'))

        with ThreadPoolExecutor(max_workers=50) as executor:
            process_func = partial(process_single_pdf, custom_prompt="Extract all details according to document type")
            results = list(executor.map(process_func, pdf_files))

        successful = sum(1 for r in results if isinstance(r, dict) and 'error' not in r)
        failed = len(results) - successful
        domestic_results = [r for r in results if isinstance(r, dict) and r.get('type') == 'domestic']
        import_results = [r for r in results if isinstance(r, dict) and r.get('type') == 'import']

        data = {
            "total_files": len(pdf_files),
            "successful_files": successful,
            "failed_files": failed,
            "domestic_count": len(domestic_results),
            "import_count": len(import_results),
            "domestic_results": domestic_results,
            "import_results": import_results,
        }

    return render(request, "testapp/extract_path.html", {"data": data})

from .utils import process_single_pdf
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

class DataTablesPagination(PageNumberPagination):
    page_size_query_param = 'length'
    page_query_param = 'page'  # We will override paginate_queryset to handle 'start'

    def paginate_queryset(self, queryset, request, view=None):
        import logging
        logger = logging.getLogger(__name__)
        try:
            start = int(request.query_params.get('start', 0))
        except (TypeError, ValueError):
            start = 0
        try:
            length = int(request.query_params.get('length', self.page_size))
        except (TypeError, ValueError):
            length = self.page_size

        logger.debug(f"DataTablesPagination: start={start}, length={length}")

        if length == 0:
            return None  # No pagination

        page_number = (start // length) + 1
        logger.debug(f"DataTablesPagination: Calculated page_number={page_number}")

        # Check if request.query_params is mutable, else create a copy
        if hasattr(request.query_params, '_mutable'):
            request.query_params._mutable = True
            request.query_params['page'] = str(page_number)
            request.query_params['page_size'] = str(length)
            request.query_params._mutable = False
        else:
            # For immutable QueryDict, create a copy and assign
            query_params = request.query_params.copy()
            query_params['page'] = str(page_number)
            query_params['page_size'] = str(length)
            request._request.GET = query_params

        logger.debug(f"DataTablesPagination: Modified request.query_params: {request.query_params}")

        return super().paginate_queryset(queryset, request, view)

    def get_paginated_response(self, data):
        draw = int(self.request.query_params.get('draw', 0))
        return Response({
            'draw': draw,
            'recordsTotal': self.page.paginator.count,
            'recordsFiltered': self.page.paginator.count,
            'data': data
        })

class DomesticInvoiceViewSet(viewsets.ModelViewSet):
    queryset = DomesticInvoice.objects.all().order_by('invoice_number')
    serializer_class = DomesticInvoiceSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['vendor_name']
    search_fields = ['invoice_number', 'vendor_name']
    ordering_fields = ['invoice_number']
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = DataTablesPagination

    def get_permissions(self):
        if self.request.method == 'GET':
            return [permissions.IsAuthenticated()]
        elif self.request.method in ['PUT', 'PATCH']:
            return [permissions.IsAuthenticated()]
        elif self.request.method == 'DELETE':
            return [permissions.IsAuthenticated()]
        return [permissions.IsAuthenticated()]

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        if not queryset.exists():
            sample_fields = self.serializer_class().get_fields()
            empty_response = {field: "Empty" for field in sample_fields}
            return Response({
                "message": "No matching records found.",
                "fields": empty_response
            })

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

class ImportInvoiceViewSet(viewsets.ModelViewSet):
    queryset = ImportInvoice.objects.all().order_by('invoice_number')
    serializer_class = ImportInvoiceSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['vendor_name', 'origin_country']
    search_fields = ['invoice_number', 'vendor_name']
    ordering_fields = ['invoice_number']
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = DataTablesPagination

    def get_permissions(self):
        if self.request.method == 'GET':
            return [permissions.IsAuthenticated()]
        elif self.request.method in ['PUT', 'PATCH']:
            return [permissions.IsAuthenticated()]
        elif self.request.method == 'DELETE':
            return [permissions.IsAuthenticated()]
        return [permissions.IsAuthenticated()]

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        if not queryset.exists():
            sample_fields = self.serializer_class().get_fields()
            empty_response = {field: "Empty" for field in sample_fields}
            return Response({
                "message": "No matching records found.",
                "fields": empty_response
            })

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


@login_required
def invoice_list_view(request):
    can_edit = request.user.has_perm('testapp.change_invoice')
    can_delete = request.user.has_perm('testapp.delete_invoice')

    domestic_invoices = list(DomesticInvoice.objects.all().order_by('invoice_number'))
    import_invoices = list(ImportInvoice.objects.all().order_by('invoice_number'))

    # Add invoice_type attribute to each invoice
    for inv in domestic_invoices:
        inv.invoice_type = 'domestic'
    for inv in import_invoices:
        inv.invoice_type = 'import'

    # Combine both lists
    combined_invoices = domestic_invoices + import_invoices

    # Calculate counts for summary
    total_invoices_count = len(combined_invoices)
    domestic_invoices_count = len(domestic_invoices)
    import_invoices_count = len(import_invoices)
    matched_invoices_count = sum(1 for inv in combined_invoices if getattr(inv, 'status', '').lower() == 'matched')
    unmatched_invoices_count = sum(1 for inv in combined_invoices if getattr(inv, 'status', '').lower() == 'miss matched')

    # Determine if comparison is done (any invoice has status 'matched' or 'miss matched')
    comparison_done = any(getattr(inv, 'status', '').lower() in ['matched', 'miss matched'] for inv in combined_invoices)

    # Determine user role for dashboard context
    user_role = getattr(request.user, 'role', 'user')

    return render(request, 'testapp/invoice_list.html', {
        'can_edit': can_edit,
        'can_delete': can_delete,
        'invoices': combined_invoices,
        'user_role': user_role,  # Pass user role to template for role-specific UI
        'total_invoices_count': total_invoices_count,
        'domestic_invoices_count': domestic_invoices_count,
        'import_invoices_count': import_invoices_count,
        'matched_invoices_count': matched_invoices_count,
        'unmatched_invoices_count': unmatched_invoices_count,
        'comparison_done': comparison_done,
    })


@csrf_exempt
@login_required
def domestic_invoice_delete(request, pk):
    if request.method != 'DELETE':
        return HttpResponseNotAllowed(['DELETE'])
    try:
        invoice = DomesticInvoice.objects.get(pk=pk)
        invoice.delete()
        return JsonResponse({'success': True})
    except DomesticInvoice.DoesNotExist:
        return JsonResponse({'error': 'Invoice not found'}, status=404)




@csrf_exempt
@login_required
def import_invoice_delete(request, pk):
    if request.method != 'DELETE':
        return HttpResponseNotAllowed(['DELETE'])
    try:
        invoice = ImportInvoice.objects.get(pk=pk)
        invoice.delete()
        return JsonResponse({'success': True})
    except ImportInvoice.DoesNotExist:

        return JsonResponse({'error': 'Invoice not found'}, status=404)


from django.core.management import call_command

from testapp.cron import DailyDataExtractionCronJob
from .extraction import run_data_extraction

@role_required(allowed_roles=['admin', 'manager'], task_name="Manual Extraction Trigger")
@login_required
def manual_extraction_trigger(request):
    """
    View to manually trigger the data extraction process with cron job logging.
    """
    from django_cron.models import CronJobLog
    from django.utils.timezone import now

    if request.method == 'POST':
        cron_job = DailyDataExtractionCronJob()
        log = CronJobLog.objects.create(
            code=cron_job.code,
            start_time=now(),
            end_time=now(),
            is_success=False,
            message='',
        )
        try:
            cron_job.do()
            log.is_success = True
            log.message = 'Cron job executed successfully.'
        except Exception as e:
            log.is_success = False
            log.message = f'Error: {str(e)}'
        finally:
            log.end_time = now()
            log.save()
        return JsonResponse({'message': 'Data extraction triggered successfully.'})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)

from django.shortcuts import redirect

@role_required(allowed_roles=['admin', 'manager'])
@login_required
def manual_extraction_button_view(request):
    """
    View to run the extraction process and then stay on the cron job page with a success message.
    """
    try:
        cron_job = DailyDataExtractionCronJob()
        cron_job.force_run()
        # After extraction, render the cron job page with success message
        return render(request, 'testapp/cron_job.html', {'message': 'Extraction Completed Successfully. Just Check Dashboard'})
    except Exception as e:
        return JsonResponse({'error': f'Failed to run extraction: {str(e)}'})

@role_required(allowed_roles=['admin', 'manager'], task_name="Manual Comparison Trigger")
@login_required
def manual_comparison_trigger(request):
    """
    View to manually trigger the comparison process with cron job logging.
    """
    from django_cron.models import CronJobLog
    from django.utils.timezone import now

    if request.method == 'POST':
        log = CronJobLog.objects.create(
            code='testapp.daily_comparison',
            start_time=now(),
            end_time=now(),
            is_success=False,
            message='',
        )
        try:
            result = run_comparison()
            log.is_success = True
            log.message = 'Comparison cron job executed successfully.'
            log.end_time = now()
            log.save()
            if 'error' in result:
                return render(request, 'testapp/upload_excel.html', {'error': result['error']})
            else:
                # Debug log to verify saving
                import logging
                logger = logging.getLogger(__name__)
                logger.info("Comparison cron job executed successfully, returning result.")
                return render(request, 'testapp/compare_result.html', result)
        except Exception as e:
            log.is_success = False
            log.message = f'Error: {str(e)}'
            log.end_time = now()
            log.save()
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to run comparison: {str(e)}")
            return render(request, 'testapp/upload_excel.html', {'error': f'Failed to run comparison: {str(e)}'})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)


@role_required(allowed_roles=['admin', 'manager'])
@login_required
def manual_comparison_button_view(request):
    """
    View to run the comparison process and stay on the cron job page with a success message.
    """
    from django_cron.models import CronJobLog
    from django.utils.timezone import now

    log = None
    try:
        log = CronJobLog.objects.create(
            code='testapp.daily_comparison',
            start_time=now(),
            end_time=now(),
            is_success=False,
            message='',
        )
        result = run_comparison()
        if 'error' in result:
            if log:
                log.is_success = False
                log.message = result['error']
                log.end_time = now()
                log.save()
            return render(request, 'testapp/cron_job.html', {'error': result['error']})
        else:
            if log:
                log.is_success = True
                log.message = 'Comparison cron job executed successfully.'
                log.end_time = now()
                log.save()
            return render(request, 'testapp/cron_job.html', {'message': 'Comparison Successful. Just check Comparison Dashboard.'})
    except Exception as e:
        if log:
            log.is_success = False
            log.message = f'Failed to run comparison: {str(e)}'
            log.end_time = now()
            log.save()
        return render(request, 'testapp/cron_job.html', {'error': f'Failed to run comparison: {str(e)}'})


from django.db.models import Value, CharField, Count
from django.db.models.functions import Cast
from rest_framework.pagination import PageNumberPagination

@api_view(['GET'])
@login_required
def combined_invoice_list(request):
    """
    API endpoint to return combined Domestic and Import invoices with invoice_type field.
    Supports filtering by invoice_type query param: 'domestic' or 'import'.
    Supports filtering by status query param: 'matched' or 'miss matched'.
    Supports pagination, ordering, and search.
    """
    from django.core.paginator import Paginator, EmptyPage
    import operator
    from django.db.models import Q

    invoice_type = request.query_params.get('invoice_type', None)
    status_filter = request.query_params.get('status', None)
    search_value = request.query_params.get('search[value]', None)
    start = int(request.query_params.get('start', 0))
    length = int(request.query_params.get('length', 10))
    order_column_index = request.query_params.get('order[0][column]', None)
    order_dir = request.query_params.get('order[0][dir]', 'asc')

    # Map DataTables column index to model field names for ordering
    column_map = {
        '1': 'invoice_number',
        '2': 'invoice_type',
        '3': 'status',
        '4': 'description_of_goods_or_services',
        '5': 'vendor_name',
        # Add more mappings as needed
    }

    order_field = column_map.get(order_column_index, 'invoice_number')
    reverse_order = (order_dir == 'desc')

    # Fetch domestic and import invoices as querysets
    domestic_qs = DomesticInvoice.objects.all()
    import_qs = ImportInvoice.objects.all()

    if status_filter:
        domestic_qs = domestic_qs.filter(status__iexact=status_filter)
        import_qs = import_qs.filter(status__iexact=status_filter)

    if search_value:
        domestic_qs = domestic_qs.filter(
            Q(invoice_number__icontains=search_value) |
            Q(vendor_name__icontains=search_value)
        )
        import_qs = import_qs.filter(
            Q(invoice_number__icontains=search_value) |
            Q(vendor_name__icontains=search_value)
        )

    domestic_list = list(domestic_qs)
    import_list = list(import_qs)

    # Add invoice_type attribute
    for inv in domestic_list:
        inv.invoice_type = 'domestic'
    for inv in import_list:
        inv.invoice_type = 'import'

    # Combine lists
    combined_list = domestic_list + import_list

    # Filter by invoice_type if specified
    if invoice_type in ['domestic', 'import']:
        combined_list = [inv for inv in combined_list if inv.invoice_type == invoice_type]

    # Sort combined list by order_field
    def get_order_attr(inv):
        attr = getattr(inv, order_field, '')
        # For invoice_type, sort by string
        if order_field == 'invoice_type':
            return attr.lower() if attr else ''
        return attr

    combined_list.sort(key=get_order_attr, reverse=reverse_order)

    # Pagination
    records_total = len(combined_list)
    records_filtered = records_total

    page_number = (start // length) + 1
    paginator = Paginator(combined_list, length)
    try:
        page = paginator.page(page_number)
    except EmptyPage:
        page = []

    # Serialize page data
    # Use appropriate serializer based on invoice_type
    serialized_data = []
    for inv in page:
        if inv.invoice_type == 'domestic':
            serializer = DomesticInvoiceSerializer(inv)
        else:
            serializer = ImportInvoiceSerializer(inv)
        data = serializer.data
        data['invoice_type'] = inv.invoice_type
        serialized_data.append(data)

    # Ensure all expected keys exist in each data item to avoid DataTables warnings
    expected_keys = [
        "id", "invoice_number", "invoice_type", "status", "description_of_goods_or_services",
        "vendor_name", "pan_number", "gstin_number", "invoice_date", "po_no", "hsn_code",
        "irn_no", "grn_no", "grn_date", "basic_amount", "cgst", "sgst", "igst",
        "total_invoice_value", "invoice_amount", "udyog_aadhar_registration_number",
        "udyog_aadhar_certificate_date", "ecc_number", "aadhar_individual", "origin_country",
        "port_of_loading", "port_of_discharge", "customs_declaration_number", "bill_of_lading_number",
        "iban_number", "swift_code", "bic_code", "sort_code", "bank_key", "street_house_no",
        "street", "building", "city", "district", "state", "country", "po_box", "email_ids",
        "mobile_landline_numbers", "fax_no", "region", "country_code", "bank_account_name",
        "bank_account_number", "bank_name", "ifsc_code", "bank_address", "invoice_currency",
        "payment_terms", "incoterms","path",
    ]

    for item in serialized_data:
        for key in expected_keys:
            if key not in item:
                item[key] = ""
        # Replace empty, missing, or "Empty" status with "Not Found"
        status_val = item.get('status')
        if not status_val or status_val.strip().lower() == 'empty':
            item['status'] = "Not Found"

    # Prepare response
    draw = int(request.query_params.get('draw', 0))
    response = {
        'draw': draw,
        'recordsTotal': records_total,
        'recordsFiltered': records_filtered,
        'data': serialized_data,
    }

    return Response(response)

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils.decorators import method_decorator

@csrf_exempt
@login_required
@role_required(allowed_roles=['admin'], task_name="Toggle User Active Status")
def toggle_user_active(request, user_id):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)
    try:
        user = CustomUser.objects.get(pk=user_id)
        user.is_active = not user.is_active
        user.save()
        return JsonResponse({'success': True, 'is_active': user.is_active})
    except CustomUser.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'User not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@csrf_exempt
@login_required
def domestic_invoice_delete(request, pk):
    if request.method != 'DELETE':
        return HttpResponseNotAllowed(['DELETE'])
    try:
        invoice = DomesticInvoice.objects.get(pk=pk)
        invoice.delete()
        return JsonResponse({'success': True})
    except DomesticInvoice.DoesNotExist:
        return JsonResponse({'error': 'Invoice not found'}, status=404)




@csrf_exempt
@login_required
def import_invoice_delete(request, pk):
    if request.method != 'DELETE':
        return HttpResponseNotAllowed(['DELETE'])
    try:
        invoice = ImportInvoice.objects.get(pk=pk)
        invoice.delete()
        return JsonResponse({'success': True})
    except ImportInvoice.DoesNotExist:

        return JsonResponse({'error': 'Invoice not found'}, status=404)


@role_required(allowed_roles=['admin', 'manager'])
@login_required
def logs_view(request):
    """
    View to display application logs.
    """
    log_file_path = os.path.join(settings.BASE_DIR, 'logs', 'app.log')
    logs = []
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
             logs = f.readlines()
    else:
        logs = ["Log file not found."]
    return render(request, 'testapp/logs.html', {'logs': logs})


@role_required(allowed_roles=['admin', 'manager'])
@login_required
def revalidation_trigger_view(request):
    """
    View to trigger re-validation process.
    """
    from .views import run_comparison  # Import the existing comparison function
    if request.method == 'POST':
        result = run_comparison()
        if 'error' in result:
            return render(request, 'testapp/revalidation_result.html', {'error': result['error']})
        else:
            return render(request, 'testapp/revalidation_result.html', result)
    else:
        return render(request, 'testapp/revalidation_trigger.html')

@login_required
def comparison_dashboard_view(request):
    # Show existing comparison data without running the comparison operation
    try:
        domestic_invoices = list(DomesticInvoice.objects.all())
        import_invoices = list(ImportInvoice.objects.all())

        domestic_results = []
        import_results = []

        for inv in domestic_invoices:
            if inv.status and inv.status.lower() in ['matched', 'miss matched']:
                mismatched_fields = inv.Description.split(', ') if inv.Description else []
                # Debug logging for invoice_data and master_data
                logger.info(f"Domestic Invoice {inv.invoice_number} - mismatched_invoice_data: {getattr(inv, 'mismatched_invoice_data', '')}")
                logger.info(f"Domestic Invoice {inv.invoice_number} - mismatched_master_data: {getattr(inv, 'mismatched_master_data', '')}")
                domestic_results.append({
                    'invoice_number': inv.invoice_number,
                    'vendor_name': inv.vendor_name,
                    'mismatched_fields': ', '.join(mismatched_fields),
                    'mismatched_count': len(mismatched_fields),
                    'invoice_data': inv.mismatched_invoice_data if hasattr(inv, 'mismatched_invoice_data') else '',
                    'master_data': inv.mismatched_master_data if hasattr(inv, 'mismatched_master_data') else '',
                })

        for inv in import_invoices:
            if inv.status and inv.status.lower() in ['matched', 'miss matched']:
                mismatched_fields = inv.Description.split(', ') if inv.Description else []
                # Debug logging for invoice_data and master_data
                logger.info(f"Import Invoice {inv.invoice_number} - mismatched_invoice_data: {getattr(inv, 'mismatched_invoice_data', '')}")
                logger.info(f"Import Invoice {inv.invoice_number} - mismatched_master_data: {getattr(inv, 'mismatched_master_data', '')}")
                import_results.append({
                    'invoice_number': inv.invoice_number,
                    'vendor_name': inv.vendor_name,
                    'mismatched_fields': ', '.join(mismatched_fields),
                    'mismatched_count': len(mismatched_fields),
                    'invoice_data': inv.mismatched_invoice_data if hasattr(inv, 'mismatched_invoice_data') else '',
                    'master_data': inv.mismatched_master_data if hasattr(inv, 'mismatched_master_data') else '',
                })

        domestic_mismatched_count_total = sum(len(inv.Description.split(', ')) if inv.Description else 0 for inv in domestic_invoices if inv.status and inv.status.lower() in ['matched', 'miss matched'])
        import_mismatched_count_total = sum(len(inv.Description.split(', ')) if inv.Description else 0 for inv in import_invoices if inv.status and inv.status.lower() in ['matched', 'miss matched'])

        result = {
            'domestic_results': domestic_results,
            'import_results': import_results,
            'domestic_mismatched_count_total': domestic_mismatched_count_total,
            'import_mismatched_count_total': import_mismatched_count_total,
        }

        return render(request, 'testapp/compare_result.html', result)
    except Exception as e:
        return render(request, 'testapp/compare_result.html', {'error': f'Failed to load comparison data: {str(e)}'})

        
@login_required
@role_required(allowed_roles=['admin', 'manager'])
def export_reports_view(request):
    """
    View to export validation reports as CSV with a UI.
    """
    from .views import run_comparison  # Use existing comparison function
    result = run_comparison()
    if 'error' in result:
        return render(request, 'testapp/export_error.html', {'error': result['error']})

    # Save the CSV content to a temporary file
    import tempfile
    import csv

    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix='.csv')
    writer = csv.writer(temp_file)
    writer.writerow(['Invoice Number', 'Vendor Name', 'Mismatched Fields', 'Invoice Data', 'Master Data'])
    for res in result.get('domestic_results', []) + result.get('import_results', []):
        writer.writerow([
            res.get('invoice_number', ''),
            res.get('vendor_name', ''),
            res.get('mismatched_fields', ''),
            res.get('invoice_data', ''),
            res.get('master_data', '')
        ])
    temp_file.close()

    # Pass the file path to the template for download
    return render(request, 'testapp/export_ui.html', {'csv_file_path': temp_file.name})
from django.db.models import Value
from django.db.models.functions import Concat

from django_cron.models import CronJobLog
from django.core.paginator import Paginator
from django.utils.timezone import localtime

from django.db.models import Q

@role_required(allowed_roles=['admin', 'manager'])
@login_required
def cron_job_logs_view(request):
    """
    View to display cron job logs with filtering by cron job type (Extraction, Comparison, or All).
    """
    import logging
    logger = logging.getLogger(__name__)
    cron_type = request.GET.get('type', 'all')  # 'extraction', 'comparison', or 'all'

    # Log distinct codes in CronJobLog for debugging
    distinct_codes = CronJobLog.objects.values('code').annotate(count=Count('id')).order_by('count')
    logger.info(f"Distinct CronJobLog codes with counts: {list(distinct_codes)}")

    # Define possible cron_code values for extraction and comparison jobs (scheduled and manual)
    extraction_cron_codes = [
        'testapp.daily_data_extraction',
        # Add other known extraction cron codes here if any
    ]
    comparison_cron_codes = [
        'testapp.daily_comparison',
        'testapp.manual_comparison_cron_job',
        # Add other known comparison cron codes here if any
    ]

    if cron_type == 'extraction':
        query = Q()
        for code in extraction_cron_codes:
            query |= Q(code__icontains=code)
        logs = CronJobLog.objects.filter(query).order_by('-start_time')
    elif cron_type == 'comparison':
        query = Q()
        for code in comparison_cron_codes:
            query |= Q(code__icontains=code)
        logs = CronJobLog.objects.filter(query).order_by('-start_time')
    else:
        logs = CronJobLog.objects.all().order_by('-start_time')

    # Debug: Log the count of logs fetched
    logger.info(f"Fetched {logs.count()} logs for cron_type '{cron_type}'")

    paginator = Paginator(logs, 25)  # Show 25 logs per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Format logs for display
    formatted_logs = []
    for log in page_obj:
        formatted_logs.append({
            'id': log.id,
            'cron_code': log.code,
            'start_time': localtime(log.start_time).strftime('%Y-%m-%d %H:%M:%S') if log.start_time else '',
            'end_time': localtime(log.end_time).strftime('%Y-%m-%d %H:%M:%S') if log.end_time else '',
            'is_success': log.is_success,
            'message': log.message or '',
        })

    context = {
        'logs': formatted_logs,
        'page_obj': page_obj,
        'cron_type': cron_type,
    }
    return render(request, 'testapp/cron_job_logs.html', context)

@login_required
def domestic_comparison_view(request):
    # Only fetch, do not run comparison
    domestic_invoices = list(DomesticInvoice.objects.all())
    domestic_results = []
    for inv in domestic_invoices:
        if inv.status and inv.status.lower() in ['matched', 'miss matched']:
            # Split and filter out empty/whitespace fields
            mismatched_fields = [f.strip() for f in (inv.Description.split(',') if inv.Description else []) if f.strip()]
            if mismatched_fields:  # Only add if there is at least one real mismatch
                domestic_results.append({
                    'invoice_number': inv.invoice_number,
                    'vendor_name': inv.vendor_name,
                    'mismatched_fields': ', '.join(mismatched_fields),
                    'mismatched_count': len(mismatched_fields),
                    'invoice_data': getattr(inv, 'mismatched_invoice_data', ''),
                    'master_data': getattr(inv, 'mismatched_master_data', ''),
                })
    # Apply search filter if needed
    search = request.GET.get('search', '').strip().lower()
    if search:
        domestic_results = [
            r for r in domestic_results
            if search in r['invoice_number'].lower() or search in r['vendor_name'].lower()
        ]
    # Calculate total mismatched fields count across all invoices
    mismatched_count = sum(r['mismatched_count'] for r in domestic_results)
    return render(request, 'testapp/domestic_comparison.html', {
        'domestic_results': domestic_results,
        'domestic_mismatched_count_total': mismatched_count,
    })
@login_required
def import_comparison_view(request):
    # Only fetch, do not run comparison
    import_invoices = list(ImportInvoice.objects.all())
    import_results = []
    for inv in import_invoices:
        if inv.status and inv.status.lower() in ['matched', 'miss matched']:
            mismatched_fields = [f.strip() for f in (inv.Description.split(',') if inv.Description else []) if f.strip()]
            if mismatched_fields:  # Only add if there is at least one real mismatch
                import_results.append({
                    'invoice_number': inv.invoice_number,
                    'vendor_name': inv.vendor_name,
                    'mismatched_fields': ', '.join(mismatched_fields),
                    'mismatched_count': len(mismatched_fields),
                    'invoice_data': getattr(inv, 'mismatched_invoice_data', ''),
                    'master_data': getattr(inv, 'mismatched_master_data', ''),
                })
    # Apply search filter if needed
    search = request.GET.get('search', '').strip().lower()
    if search:
        import_results = [
            r for r in import_results
            if search in r['invoice_number'].lower() or search in r['vendor_name'].lower()
        ]
    # Calculate total mismatched fields count across all invoices
    mismatched_count = sum(r['mismatched_count'] for r in import_results)
    return render(request, 'testapp/import_comparison.html', {
        'import_results': import_results,
        'import_mismatched_count_total': mismatched_count,
    })

from django.shortcuts import render
from django.core.cache import cache
from django_cron.models import CronJobLog

def get_cron_status():
    extraction_running = cache.get('extraction_job_running', False)

    if extraction_running:
        status = 'Processing'
        message = 'Processing'
    else:
        log = CronJobLog.objects.filter(code='testapp.daily_data_extraction').order_by('-start_time').first()
        if log:
            status = 'Success' if log.is_success else 'Failed'
            message = log.message
        else:
            status = ''
            message = ''
    return status, message


def cron_status_view(request):
    status, message = get_cron_status()
    return render(request, 'cron_job.html', {
        'status': status,
        'message': message,
    })
