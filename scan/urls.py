from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [

    path('upload', views.pic),
    path('peter', views.sendpic),

]