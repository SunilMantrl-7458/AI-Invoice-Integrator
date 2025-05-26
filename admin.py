from django.contrib import admin
from django_cron.models import CronJobLog
from django.contrib.auth.admin import UserAdmin
from .models import DomesticInvoice, ImportInvoice, CustomUser
from .forms import CustomUserCreationForm, CustomUserChangeForm
from django_cron.models import CronJobLog, CronJobLock

class DomesticInvoiceAdmin(admin.ModelAdmin):
    list_display = ['invoice_number', 'vendor_name', 'city', 'state', 'country', 'payment_terms']
    search_fields = ['invoice_number', 'vendor_name', 'city', 'state', 'country']
    list_filter = ['state', 'country']

    def has_add_permission(self, request):
        return request.user.has_perm('testapp.edit_invoice') or request.user.has_perm('testapp.edit_only')

    def has_change_permission(self, request, obj=None):
        return request.user.has_perm('testapp.edit_invoice') or request.user.has_perm('testapp.edit_only')

    def has_delete_permission(self, request, obj=None):
        return request.user.has_perm('testapp.can_delete_invoice') or request.user.has_perm('testapp.delete_only')

    def has_view_permission(self, request, obj=None):
        return (
            request.user.has_perm('testapp.view_only_invoice') or
            request.user.has_perm('testapp.view_only') or
            request.user.has_perm('testapp.edit_invoice') or
            request.user.has_perm('testapp.edit_only') or
            request.user.has_perm('testapp.can_delete_invoice') or
            request.user.has_perm('testapp.delete_only')
        )


class ImportInvoiceAdmin(admin.ModelAdmin):
    list_display = ['invoice_number', 'vendor_name', 'origin_country', 'port_of_loading', 'port_of_discharge', 'invoice_currency']
    search_fields = ['invoice_number', 'vendor_name', 'origin_country', 'port_of_loading', 'port_of_discharge', 'invoice_currency']
    list_filter = ['origin_country', 'invoice_currency']

    def has_add_permission(self, request):
        return request.user.has_perm('testapp.edit_invoice') or request.user.has_perm('testapp.edit_only')

    def has_change_permission(self, request, obj=None):
        return request.user.has_perm('testapp.edit_invoice') or request.user.has_perm('testapp.edit_only')

    def has_delete_permission(self, request, obj=None):
        return request.user.has_perm('testapp.can_delete_invoice') or request.user.has_perm('testapp.delete_only')

    def has_view_permission(self, request, obj=None):
        return (
            request.user.has_perm('testapp.view_only_invoice') or
            request.user.has_perm('testapp.view_only') or
            request.user.has_perm('testapp.edit_invoice') or
            request.user.has_perm('testapp.edit_only') or
            request.user.has_perm('testapp.can_delete_invoice') or
            request.user.has_perm('testapp.delete_only')
        )

class CustomUserAdmin(UserAdmin):
    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    model = CustomUser
    list_display = ['username', 'email', 'role', 'is_staff', 'is_active']
    list_filter = ['role', 'is_staff', 'is_active']
    fieldsets = (
        (None, {'fields': ('username', 'email', 'password', 'role')}),
        ('Permissions', {'fields': ('is_staff', 'is_active', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'role', 'password1', 'password2', 'is_staff', 'is_active')}

        ),
    )
    search_fields = ('username', 'email')
    ordering = ('username',)

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(DomesticInvoice, DomesticInvoiceAdmin)
admin.site.register(ImportInvoice, ImportInvoiceAdmin)
# Removed registration of CronJobLog and CronJobLock to avoid AlreadyRegistered error
# as django_cron already registers these models in its own admin.py
# admin.site.register(CronJobLog)
# admin.site.register(CronJobLock)
