from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import CustomUser, DomesticInvoice, ImportInvoice

class CustomUserCreationForm(UserCreationForm):
    role = forms.ChoiceField(choices=CustomUser.ROLE_CHOICES)

    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'role', 'password1', 'password2')

class CustomUserChangeForm(UserChangeForm):
    role = forms.ChoiceField(choices=CustomUser.ROLE_CHOICES)

    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'role')

class DirectoryForm(forms.Form):
    directory_path = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter directory path containing PDF invoices'
        }),
        required=True
    )

class DomesticInvoiceForm(forms.ModelForm):
    class Meta:
        model = DomesticInvoice
        fields = '__all__'

class ImportInvoiceForm(forms.ModelForm):
    class Meta:
        model = ImportInvoice
        fields = '__all__'
