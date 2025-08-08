# ragapi/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .llm_chain import analyze_query

@csrf_exempt
def hackrx_webhook(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        data = json.loads(request.body)
        query = data.get("query")
        if not query:
            return JsonResponse({'error': "Missing 'query'"}, status=400)

        result = analyze_query(query)
        return JsonResponse(result, safe=False)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
