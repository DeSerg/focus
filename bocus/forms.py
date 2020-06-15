from django import forms


class UploadPhotosForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
class RecognizeForm(forms.Form):
    file = forms.FileField()
