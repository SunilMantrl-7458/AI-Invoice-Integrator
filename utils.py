from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from .views import save_extracted_data, azure_ocr_extract_text, identify_pdf_type_by_folder, process_pdf_and_extract_data
from .models import DomesticInvoice, ImportInvoice
logger = logging.getLogger(__name__)
import os

# def process_single_pdf(pdf_file, custom_prompt):
#     try:
#         text = azure_ocr_extract_text(str(pdf_file))
#         pdf_type = identify_pdf_type_by_folder(str(pdf_file))
#         result_data = process_pdf_and_extract_data(pdf_file, custom_prompt)
#         # Save extracted data to database
#         save_result = None
#         try:
#             # Always pass the file name so the path field is set correctly
#             save_result = save_extracted_data(pdf_type, result_data, pdf_file.name)
#             if save_result and "error" in save_result:
#                 logger.error(f"Failed to save extracted data for {pdf_file.name}: {save_result['error']}")
#         except Exception as e:
#             logger.error(f"Error saving extracted data for {pdf_file.name}: {e}")
#         return {"file": pdf_file.name, "result": result_data, "type": pdf_type, "save_result": save_result}
#     except Exception as e:
#         return Exception(f"Error processing {pdf_file.name}: {str(e)}")



def process_single_pdf(pdf_file, custom_prompt):
    try:
        pdf_type = identify_pdf_type_by_folder(str(pdf_file))
        if pdf_type == "domestic":
            model_class = DomesticInvoice
        elif pdf_type == "import":
            model_class = ImportInvoice
        invoice_number = os.path.splitext(os.path.basename(pdf_file))[0]
        print(invoice_number)
        if model_class.objects.filter(path=invoice_number).exists():
            logger.info(f"File {pdf_file.name} already exists in the database. Skipping processing.")
            return {"file": pdf_file.name, "result": None, "type": pdf_type, "save_result": None}
        else:    
            logger.info(f"Processing file: {pdf_file.name}")
            text = azure_ocr_extract_text(str(pdf_file))
            pdf_type = identify_pdf_type_by_folder(str(pdf_file))
            result_data = process_pdf_and_extract_data(pdf_file, custom_prompt)
            # Save extracted data to database
            save_result = None
            try:
                # Always pass the file name so the path field is set correctly
                save_result = save_extracted_data(pdf_type, result_data, pdf_file.name)
                if save_result and "error" in save_result:
                    logger.error(f"Failed to save extracted data for {pdf_file.name}: {save_result['error']}")
            except Exception as e:
                logger.error(f"Error saving extracted data for {pdf_file.name}: {e}")
            return {"file": pdf_file.name, "result": result_data, "type": pdf_type, "save_result": save_result}
        
    except Exception as e:
        logger.error(f"Error saving extracted data for {pdf_file.name}: {e}")
        return Exception(f"Error processing {pdf_file.name}: {str(e)}")

