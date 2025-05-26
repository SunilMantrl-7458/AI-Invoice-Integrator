from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import get_user_model
from django.contrib import messages
from .forms import CustomUserCreationForm, CustomUserChangeForm

User = get_user_model()

def admin_required(view_func):
    decorated_view_func = login_required(user_passes_test(lambda u: u.is_authenticated and u.role == 'admin')(view_func))
    return decorated_view_func

@admin_required
def user_list(request):
    role_filter = request.GET.get('role', '')
    status_filter = request.GET.get('status', '')

    users = User.objects.all()

    if role_filter:
        users = users.filter(role=role_filter)

    if status_filter:
        if status_filter.lower() == 'active':
            users = users.filter(is_active=True)
        elif status_filter.lower() == 'inactive':
            users = users.filter(is_active=False)

    context = {
        'users': users,
        'role_filter': role_filter,
        'status_filter': status_filter,
    }
    return render(request, 'testapp/user_list.html', context)

from django.http import JsonResponse

@admin_required
def user_create(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"User {user.username} with role {user.role} created successfully.")
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'username': user.username,
                    'role': user.role,
                })
            else:
                return redirect('user_list')
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'errors': form.errors,
                })
    else:
        form = CustomUserCreationForm()
    return render(request, 'testapp/user_form.html', {'form': form})

@admin_required
def user_edit(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        form = CustomUserChangeForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return redirect('user_list')
    else:
        form = CustomUserChangeForm(instance=user)
    return render(request, 'testapp/user_form.html', {'form': form, 'edit_mode': True})

@admin_required
def user_delete(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        user.delete()
        return redirect('user_list')
    return render(request, 'testapp/user_confirm_delete.html', {'user': user})
