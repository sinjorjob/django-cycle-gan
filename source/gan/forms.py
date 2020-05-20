from django import forms
from .models import Photo


class PhotoForm(forms.ModelForm):


    class Meta:
        model = Photo
        fields = ('image',)
    
    image = forms.ImageField(widget=forms.ClearableFileInput(attrs={'class':'custom-file-input'}))