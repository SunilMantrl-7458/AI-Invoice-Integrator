from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from .views import process_single_pdf
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def run_data_extraction():
    """
    Function to run data extraction on the predefined directory.
    This can be called by the management command or cron job.
    """
    # Replace hardcoded path with dynamic environment variable
    predefined_directory_path = os.getenv("PDF_DIRECTORY")

    # Ensure the environment variable is set
    if not predefined_directory_path:
        raise EnvironmentError("The PDF_DIRECTORY environment variable is not set in the .env file.")

    pdf_files = list(Path(predefined_directory_path).rglob('*.pdf'))

    # Increased max_workers from 5 to 50 to improve speed, adjust if rate limiting occurs
    with ThreadPoolExecutor(max_workers=50) as executor:
        process_func = partial(process_single_pdf, custom_prompt="Extract all details according to document type")
        results = list(executor.map(process_func, pdf_files))

    return results
