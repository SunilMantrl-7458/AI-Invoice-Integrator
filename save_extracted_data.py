import json
import logging
from .models import DomesticInvoice, ImportInvoice
 
logger = logging.getLogger(__name__)
 
import re,os
 
def normalize_key(key):
    # Remove all non-alphanumeric characters except slash and lowercase

    return re.sub(r'[^a-zA-Z0-9/]', '', key).lower()

# Example mapping for import invoices
IMPORT_FIELD_MAP = {
    "SWIFT Code": "swift_code",
    "IBAN Number": "iban_number",
    "Mobile/Landline numbers": "mobile_landline_numbers",  # If needed for import
    # add more mappings as needed
}

# Example mapping for domestic invoices
DOMESTIC_FIELD_MAP = {
    "Mobile/Landline numbers": "mobile_landline_numbers",
    # add more mappings as needed
}
 
from dateutil.parser import parse as date_parse
 
# def save_extracted_data(pdf_type, extracted_json):
def save_extracted_data(pdf_type, extracted_json, pdf_path=None):
    """
    Save extracted JSON data to the appropriate model based on pdf_type.
    Allow saving even if required fields are empty, but log warnings.
    Optionally save the PDF file path if provided.
    """
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return {"error": "Invalid JSON data"}

    if pdf_type == "domestic":
        model_class = DomesticInvoice
        required_fields = ['invoice_number', 'vendor_name']
    elif pdf_type == "import":
        model_class = ImportInvoice
        required_fields = ['invoice_number', 'vendor_name']
    else:
        return {"error": f"Unknown pdf_type: {pdf_type}"}

    # Normalize data keys for matching
    normalized_data = {normalize_key(k): v for k, v in data.items()}

    # Map JSON keys to model fields (assuming keys match model field names or close)
    # For missing fields, use None or empty string as appropriate
    model_fields = {field.name for field in model_class._meta.get_fields() if field.concrete and not field.auto_created}

    instance_data = {}
    for key in model_fields:
        value = None
        # Select the correct mapping based on pdf_type
        if pdf_type == "import":
            field_map = IMPORT_FIELD_MAP
        else:
            field_map = DOMESTIC_FIELD_MAP

        # Find if any mapping points to this model field
        mapped_json_key = next((json_key for json_key, model_field in field_map.items() if model_field == key), None)
        if mapped_json_key and mapped_json_key in data:
            value = data[mapped_json_key]
        else:
            norm_key = normalize_key(key)
            if norm_key in normalized_data:
                value = normalized_data[norm_key]

        # Parse and format date fields
        if 'date' in key and isinstance(value, str) and value.strip() not in ['', 'Empty']:
            try:
                val_str = value.strip()
                if val_str.isdigit() and len(val_str) == 8:
                    # Parse as DDMMYYYY
                    parsed_date = date_parse(f"{val_str[:2]}/{val_str[2:4]}/{val_str[4:]}", dayfirst=True)
                else:
                    parsed_date = date_parse(value, dayfirst=True)
                # Format date as DD/MM/YY
                value = parsed_date.strftime('%d/%m/%y')
            except Exception as e:
                logger.warning(f"Failed to parse date field {key} with value '{value}': {e}")
        # Convert None or empty string to "Empty" string for consistent saving
        if value is None or (isinstance(value, str) and value.strip() == ""):
            value = "Empty"
        instance_data[key] = value

    # Save PDF path if provided and model has 'path' field
    if pdf_path is not None and "path" in model_fields:
        path = os.path.splitext(os.path.basename(pdf_path))[0]
        instance_data["path"] = path

    # Log warnings for missing required fields but do not block saving
    missing_required = [field for field in required_fields if instance_data.get(field) in [None, ""]]
    if missing_required:
        error_msg = f"Missing required fields for {pdf_type} invoice: {', '.join(missing_required)}"
        logger.error(error_msg)
        return {"error": error_msg}

    # Duplicate check: skip saving if invoice_number already exists
    # invoice_number = instance_data.get('invoice_number')
    # if invoice_number:
    #     if model_class.objects.filter(invoice_number=invoice_number).exists():
    #         logger.info(f"Duplicate invoice {invoice_number} found for {pdf_type}. Skipping save.")
    #         return {"skipped": True, "reason": f"Duplicate invoice_number {invoice_number}"}

    try:
        instance = model_class.objects.create(**instance_data)
        logger.info(f"Saved {pdf_type} invoice with id {instance.id}")
        return {"success": True, "id": instance.id}
    except Exception as e:
        logger.error(f"Error saving {pdf_type} invoice: {e}")
        return {"error": str(e)}