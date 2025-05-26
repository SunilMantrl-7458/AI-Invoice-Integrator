
from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.utils import timezone

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('manager', 'Manager'),
        ('user', 'User'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='admin')
    groups = models.ManyToManyField(
        Group,
        related_name='customuser_set',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_query_name='customuser',
        verbose_name='groups',
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name='customuser_set',
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='customuser',
        verbose_name='user permissions',
    )

    def save(self, *args, **kwargs):
        # Ensure superusers have role 'admin'
        if self.is_superuser:
            self.role = 'admin'
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.username} ({self.role})"

class DomesticInvoice(models.Model):
    invoice_number = models.CharField(max_length=255)
    invoice_date= models.CharField(max_length=255, blank=True, null=True)
    invoice_amount = models.CharField(max_length=255, blank=True, null=True)
    vendor_name = models.CharField(max_length=255)
    po_no=models.CharField(max_length=250, blank=True, null=True)
    hsn_code=models.CharField(max_length=250, blank=True, null=True)
    irn_no=models.CharField(max_length=250, blank=True, null=True)
    grn_no=models.CharField(max_length=250, blank=True, null=True)
    grn_date=models.CharField(max_length=255, blank=True, null=True)
    basic_amount=models.CharField(max_length=250, blank=True, null=True)
    cgst=models.CharField(max_length=250, blank=True, null=True)
    igst=models.CharField(max_length=250, blank=True, null=True)
    sgst=models.CharField(max_length=250, blank=True, null=True)
    pan_number = models.CharField(max_length=50, blank=True, null=True)
    gstin_number = models.CharField(max_length=50, blank=True, null=True)
    udyog_aadhar_registration_number = models.CharField(max_length=100, blank=True, null=True)
    udyog_aadhar_certificate_date = models.CharField(max_length=100, blank=True, null=True)
    ecc_number = models.CharField(max_length=50, blank=True, null=True)
    aadhar_individual = models.CharField(max_length=50, blank=True, null=True)
    street_house_no = models.CharField(max_length=255, blank=True, null=True)
    street = models.CharField(max_length=255, blank=True, null=True)
    building = models.CharField(max_length=255, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    district = models.CharField(max_length=100, blank=True, null=True)
    state = models.CharField(max_length=100, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    po_box = models.CharField(max_length=50, blank=True, null=True)
    email_ids = models.CharField(max_length=250, blank=True, null=True)
    mobile_landline_numbers = models.CharField(max_length=50, blank=True, null=True)
    fax_no = models.CharField(max_length=50, blank=True, null=True)
    region = models.CharField(max_length=100, blank=True, null=True)
    country_code = models.CharField(max_length=10, blank=True, null=True)
    bank_account_name = models.CharField(max_length=255, blank=True, null=True)
    bank_account_number = models.CharField(max_length=100, blank=True, null=True)
    bank_name = models.CharField(max_length=255, blank=True, null=True)
    ifsc_code = models.CharField(max_length=50, blank=True, null=True)
    bank_address = models.CharField(max_length=255, blank=True, null=True)
    payment_terms = models.CharField(max_length=255, blank=True, null=True)
    total_invoice_value = models.CharField(max_length=255, blank=True, null=True)
    description_of_goods_or_services = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=50, blank=True, null=True)
    Description=models.TextField(blank=True, null=True)
    mismatched_master_data=models.CharField(max_length=500, blank=True, null=True)
    mismatched_invoice_data=models.CharField(max_length=500, blank=True, null=True)
    path=models.CharField(max_length=500, blank=True, null=True)
    def __str__(self):
        return self.invoice_number

class ImportInvoice(models.Model):
    invoice_number = models.CharField(max_length=255)
    invoice_date= models.CharField(max_length=255, blank=True, null=True)
    invoice_amount = models.CharField(max_length=255, blank=True, null=True)
    vendor_name = models.CharField(max_length=255)
    po_no=models.CharField(max_length=250, blank=True, null=True)
    hsn_code=models.CharField(max_length=250, blank=True, null=True)
    irn_no=models.CharField(max_length=250, blank=True, null=True)
    grn_no=models.CharField(max_length=250, blank=True, null=True)
    grn_date=models.CharField(max_length=255, blank=True, null=True)
    basic_amount=models.CharField(max_length=250, blank=True, null=True)
    cgst=models.CharField(max_length=250, blank=True, null=True)
    igst=models.CharField(max_length=250, blank=True, null=True)
    sgst=models.CharField(max_length=250, blank=True, null=True)
    pan_number = models.CharField(max_length=50, blank=True, null=True)
    ecc_number = models.CharField(max_length=50, blank=True, null=True)
    origin_country = models.CharField(max_length=100, blank=True, null=True)
    port_of_loading = models.CharField(max_length=100, blank=True, null=True)
    port_of_discharge = models.CharField(max_length=100, blank=True, null=True)
    customs_declaration_number = models.CharField(max_length=100, blank=True, null=True)
    bill_of_lading_number = models.CharField(max_length=100, blank=True, null=True)
    street_house_no = models.CharField(max_length=255, blank=True, null=True)
    street = models.CharField(max_length=255, blank=True, null=True)
    building = models.CharField(max_length=255, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    district = models.CharField(max_length=100, blank=True, null=True)
    state = models.CharField(max_length=100, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    po_box = models.CharField(max_length=50, blank=True, null=True)
    email_ids = models.CharField(max_length=250, blank=True, null=True)
    mobile_landline_numbers = models.CharField(max_length=50, blank=True, null=True)
    fax_no = models.CharField(max_length=50, blank=True, null=True)
    region = models.CharField(max_length=100, blank=True, null=True)
    country_code = models.CharField(max_length=10, blank=True, null=True)
    bank_account_name = models.CharField(max_length=255, blank=True, null=True)
    bank_account_number = models.CharField(max_length=100, blank=True, null=True)
    iban_number = models.CharField(max_length=100, blank=True, null=True)
    bank_name = models.CharField(max_length=255, blank=True, null=True)
    swift_code = models.CharField(max_length=50, blank=True, null=True)
    bic_code = models.CharField(max_length=50, blank=True, null=True)
    sort_code = models.CharField(max_length=50, blank=True, null=True)
    bank_key = models.CharField(max_length=50, blank=True, null=True)
    bank_address = models.TextField(blank=True, null=True)
    invoice_currency = models.CharField(max_length=20, blank=True, null=True)
    payment_terms = models.TextField(blank=True, null=True)
    # incoterms = models.CharField(max_length=100, blank=True, null=True)
    total_invoice_value = models.CharField(max_length=255, blank=True, null=True)
    description_of_goods_or_services = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=50, blank=True, null=True)
    Description=models.TextField(blank=True, null=True)
    mismatched_master_data=models.CharField(max_length=500, blank=True, null=True)
    mismatched_invoice_data=models.CharField(max_length=500, blank=True, null=True)

    path=models.CharField(max_length=500, blank=True, null=True)

    # Routing/Intermediate bank fields
    routing_intermediate_bank_account_number = models.CharField(max_length=100, blank=True, null=True)
    routing_intermediate_bank_name = models.CharField(max_length=255, blank=True, null=True)
    routing_intermediate_swift_code = models.CharField(max_length=50, blank=True, null=True)
    routing_intermediate_bic_code = models.CharField(max_length=50, blank=True, null=True)
    routing_intermediate_sort_code = models.CharField(max_length=50, blank=True, null=True)



    def __str__(self):
        return self.invoice_number

class UserProfile(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='profile')

    def __str__(self):
        return f"Profile of {self.user.username}"

    class Meta:
        permissions = [
            ("view_only_invoice", "Can view invoices only"),
            ("edit_invoice", "Can edit invoices"),
            ("can_delete_invoice", "Can delete invoices"),
            ("view_only", "Can only view invoices (no edit/delete)"),
            ("edit_only", "Can only edit invoices (no delete)"),
            ("delete_only", "Can only delete invoices (no edit)"),
        ]

# class CustomCronLog(models.Model):
#     job_code = models.CharField(max_length=100)
#     status = models.CharField(max_length=20)
#     message = models.TextField()
#     start_time = models.DateTimeField()
#     end_time = models.DateTimeField()
