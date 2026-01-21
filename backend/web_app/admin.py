from django.contrib import admin
from .models import Interaction

@admin.register(Interaction)
class InteractionAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'question', 'created_at')
    search_fields = ('question', 'answer', 'user_id')
    list_filter = ('created_at',)
