# AI-Invoice-Integrator

## Overview
This project is a comprehensive solution for managing and processing invoices, tailored for businesses that handle both domestic and import transactions. It streamlines the workflow by automating data extraction, validation, and comparison processes. The system is designed to ensure accuracy, efficiency, and transparency in invoice management. Key highlights include:

- **Automated Data Extraction**: Leverages OCR and other utilities to extract data from uploaded PDF invoices, reducing manual effort and errors.
- **Invoice Comparison**: Identifies mismatched fields between invoices and provides detailed reports for reconciliation.
- **Role-Based Access Control**: Ensures secure access to features based on user roles (admin, manager, user).
- **Cron Job Integration**: Supports both manual and scheduled execution of data extraction and comparison tasks, ensuring timely processing.
- **User-Friendly Interface**: Provides an intuitive UI for uploading invoices, viewing logs, and managing data.
- **Scalability**: Built on Django, the system is scalable and can handle large volumes of data efficiently.

This project is ideal for organizations looking to digitize and optimize their invoice management processes while maintaining high standards of data integrity and security.

## Features
- **Invoice Management**: Manage domestic and import invoices with CRUD operations.
- **Data Extraction**: Extract data from uploaded PDFs using OCR and other utilities.
- **Comparison**: Compare invoices for mismatched fields and generate reports.
- **Cron Jobs**: Schedule and execute data extraction and comparison tasks.
- **User Management**: Role-based access control for admin, manager, and user roles.
- **Logs Viewer**: View application logs for debugging and monitoring.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd Database_integrations
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Apply migrations:
   ```bash
   python manage.py migrate
   ```

5. Create a superuser:
   ```bash
   python manage.py createsuperuser
   ```

6. Run the development server:
   ```bash
   python manage.py runserver
   ```

## Usage

- Access the application at `http://127.0.0.1:8000/`.
- Log in with the superuser credentials to access the admin dashboard.
- Use the provided UI to upload invoices, trigger data extraction, and view comparison results.

## Project Structure

```
Database_integrations/
├── Database_integrations/       # Core Django project files
├── logs/                        # Application logs
├── static/                      # Static files (images, JS, CSS)
├── templates/                   # HTML templates
├── testapp/                     # Main application
│   ├── models.py                # Database models
│   ├── views.py                 # Application views
│   ├── serializers.py           # API serializers
│   ├── tasks.py                 # Background tasks
│   ├── cron.py                  # Cron job logic
│   └── ...                      # Other utility files
└── manage.py                    # Django management script
```

## Key Files
- `views.py`: Contains the main logic for handling requests and rendering templates.
- `models.py`: Defines the database schema for invoices and users.
- `cron.py`: Implements cron job functionality for data extraction and comparison.
- `templates/`: Contains HTML templates for the UI.

## Requirements
- Python 3.8+
- Django 3.2+
- Additional dependencies listed in `requirements.txt`.

## Contributing
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push the branch.
4. Submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For any inquiries or support, please contact [mantrisunil43@gmail.com].
