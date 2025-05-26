from django.contrib.auth import get_user_model

def users_list(request):
    User = get_user_model()
    users = User.objects.all()
    return {'users': users}
