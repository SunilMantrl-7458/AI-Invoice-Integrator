from django.urls import path
from . import views_user_management
from . import views
from . import views_admin
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('users/', views_user_management.user_list, name='user_list'),
    path('users/create/', views_user_management.user_create, name='user_create'),
    path('users/edit/<int:user_id>/', views_user_management.user_edit, name='user_edit'),
    path('users/delete/<int:user_id>/', views_user_management.user_delete, name='user_delete'),

    # Manual extraction trigger URL
    # path('manual-extraction-trigger/', login_required(views.manual_extraction_trigger), name='manual_extraction_trigger'),

    # Manual extraction button page
    path('manual-extraction/', login_required(views.manual_extraction_button_view), name='manual_extraction_button'),

    # Admin panel log viewer
    path('admin/logs/', login_required(views_admin.log_viewer), name='admin_log_viewer'),

    # Admin panel re-validation trigger
    path('admin/trigger-revalidation/', login_required(views_admin.trigger_revalidation), name='admin_trigger_revalidation'),

    # Admin panel export invoices
    path('admin/export-invoices/', login_required(views_admin.export_invoices_csv), name='admin_export_invoices'),

    # Admin dashboard
    path('admin/dashboard/', login_required(views_admin.admin_dashboard_view), name='admin_dashboard'),
]
