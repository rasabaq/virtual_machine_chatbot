from django.contrib import admin
from .models import User, Conversation, Message, Interaction


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('email', 'name', 'is_active', 'created_at')
    search_fields = ('email', 'name')
    list_filter = ('is_active', 'created_at')


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'created_at', 'updated_at')
    search_fields = ('title', 'user__email')
    list_filter = ('created_at', 'updated_at')


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'role', 'content_preview', 'created_at')
    search_fields = ('content',)
    list_filter = ('role', 'created_at')

    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content'


@admin.register(Interaction)
class InteractionAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'question', 'created_at')
    search_fields = ('question', 'answer', 'user_id')
    list_filter = ('created_at',)
