# ragapi/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .llm_chain import analyze_questions

@csrf_exempt
def hackrx_webhook(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        data = json.loads(request.body)
        
        # FIX: The new documentation uses "questions" and "documents" keys.
        questions = data.get("questions")
        documents = data.get("documents")
        
        if not questions:
            return JsonResponse({'error': "Missing 'questions' in request body"}, status=400)
            
        # You may not need to use the documents URL immediately, 
        # but your view should at least be aware of it.
        if not documents:
            return JsonResponse({'error': "Missing 'documents' in request body"}, status=400)

        # Call the analysis function with the list of questions.
        # Note: You'll need to adapt analyze_questions in llm_chain.py
        # to handle the new document URL.
        llm_response = analyze_questions(questions)
        
        # The platform expects a final JSON with an "answers" key.
        if 'answers' in llm_response:
            return JsonResponse({'answers': llm_response['answers']}, safe=False)
        else:
            return JsonResponse({'error': "Failed to generate answers"}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
