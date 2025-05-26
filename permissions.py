from rest_framework import permissions

class ViewOnlyPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_superuser or request.user.has_perm('testapp.view_only')

    def has_object_permission(self, request, view, obj):
        return request.user.is_superuser or request.user.has_perm('testapp.view_only')


class EditOnlyPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.method in ['POST', 'PUT', 'PATCH']:
            return (
                request.user.is_superuser or
                request.user.has_perm('testapp.edit_only') or
                request.user.has_perm('testapp.edit_invoice')
            )
        return False

    def has_object_permission(self, request, view, obj):
        if request.method in ['POST', 'PUT', 'PATCH']:
            return (
                request.user.is_superuser or
                request.user.has_perm('testapp.edit_only') or
                request.user.has_perm('testapp.edit_invoice')
            )
        return False


class DeleteOnlyPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.method == 'DELETE':
            return (
                request.user.is_superuser or
                request.user.has_perm('testapp.delete_only') or
                request.user.has_perm('testapp.can_delete_invoice')
            )
        return False

    def has_object_permission(self, request, view, obj):
        if request.method == 'DELETE':
            return (
                request.user.is_superuser or
                request.user.has_perm('testapp.delete_only') or
                request.user.has_perm('testapp.can_delete_invoice')
            )
        return False
