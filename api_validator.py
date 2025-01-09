import requests
import subprocess
import json
import re
from huggingface_hub import InferenceClient

# Huggingface api
client = InferenceClient(api_key="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")

def validate_api_call(api_request):
    """
    Validates an API call by executing it based on its language (Python or curl).
    Returns a tuple (is_valid, error_log).
    """
    try:
        if api_request.strip().startswith("curl"):  # If it's a curl command
            # Execute curl using subprocess
            result = subprocess.run(
                api_request.split(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr  # Return the error log
        else:  # Assume it's Python code
            # Use exec() to run Python code in a restricted namespace
            namespace = {}
            exec(api_request, {}, namespace)
            if "response" in namespace:
                response = namespace["response"]
                if isinstance(response, requests.Response) and response.status_code == 200:
                    return True, None
                else:
                    return False, response.text if response else "Unknown error"
            return False, "No valid response object found in the executed code."
    except Exception as e:
        return False, str(e)

def attempt_to_fix_api_call(api_request, error_logs):
    pattern = r"^name '(.+?)' is not defined$"
    match = re.match(pattern, error_logs)
    if match:
        return api_request

    input_text = f"""
        THE ORIGINAL EXECUTABLE STRING: {api_request}
        
        THE ERROR LOGS OBTAINED FROM THE FAILED EXECUTION: {error_logs}

        BASED ON THE ERRORS, CORRECT THE EXECUTABLE STRING.

        THE OUTPUT SHOULD BE STRICTLY LIKE THE FOLLOWING. NO OTHER TEXTS SHOULD BE THERE:

        ```<language_used>
        <the_corrected_executable_string>
        ```   

        Answer:
        """
    
    messages = [
	    {
	    	"role": "user",
	    	"content": input_text
	    }
    ]

    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3", 
    	messages=messages, 
    	max_tokens=500
    )

    return completion.choices[0].message.content
