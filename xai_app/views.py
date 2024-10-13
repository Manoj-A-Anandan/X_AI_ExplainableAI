from rest_framework.decorators import api_view
from rest_framework.response import Response
from .ml_model import TitanicModel
from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to the XAI Project!")

# Load the model and train it
model = TitanicModel()
model.train_model(r'D:\Projects\Xai\xai_backend\titanic - titanic.csv')

@api_view(['POST'])
def explain(request):
    instance = request.data.get('instance')

    # Validate the instance input
    if instance is None or (isinstance(instance, list) and not instance):
        return Response({'error': 'Invalid input: instance is required.'}, status=400)

    # Call the model to get explanations
    explanations = model.explain_instance(instance)

    return Response({
        'lime_explanation': explanations['lime_explanation'],
        'shap_values': explanations['shap_values'],
        'expected_value': explanations['expected_value']
    })
