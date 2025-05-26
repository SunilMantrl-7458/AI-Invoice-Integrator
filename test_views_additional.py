import unittest
from unittest.mock import patch, MagicMock
from rest_framework.test import APIRequestFactory, force_authenticate
from testapp.views import extract_invoice_data
import json

class DummyUser:
    def __init__(self, username='testuser', is_authenticated=True):
        self.username = username
        self.is_authenticated = is_authenticated
        self.role = 'admin'
        self.is_active = True

    def has_perm(self, perm):
        return True

class TestAdditionalViews(unittest.TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.user = DummyUser()

    @patch('testapp.views.process_pdf_and_extract_data')
    @patch('testapp.views.save_extracted_data')
    def test_extract_invoice_data_success(self, mock_save, mock_process):
        mock_process.return_value = json.dumps({"Invoice Number": "INV123", "Vendor Name": "Test Vendor"})
        mock_save.return_value = {}

        request = self.factory.post('/extract_invoice_data/', {'pdf': MagicMock()}, format='multipart')
        force_authenticate(request, user=self.user)
        response = extract_invoice_data(request)
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response.data)
        self.assertEqual(response.data['data']['Invoice Number'], 'INV123')

    def test_extract_invoice_data_no_file(self):
        request = self.factory.post('/extract_invoice_data/', {}, format='multipart')
        force_authenticate(request, user=self.user)
        response = extract_invoice_data(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.data)

if __name__ == '__main__':
    unittest.main()
