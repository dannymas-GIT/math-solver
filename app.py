from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import traceback
from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic client
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Adjust SSL-related logging
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/math')
def math():
    return render_template('math.html')

@app.route('/science')
def science():
    return render_template('science.html')

@app.route('/law')
def law():
    return render_template('law.html')

@app.route('/business')
def business():
    return render_template('business.html')

@app.route('/api/solve', methods=['POST'])
def solve_problem():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        problem_type = data.get('problemType')
        question = data.get('question')

        if not problem_type or not question:
            return jsonify({'error': 'Invalid input'}), 400

        logging.info(f"Received request - Problem Type: {problem_type}, Question: {question}")

        claude_result = solve_with_claude(problem_type, question)
        gpt4_result = solve_with_gpt4(problem_type, question)

        return jsonify({
            'claude_result': claude_result,
            'gpt4_result': gpt4_result,
            'problem_type': problem_type,
            'question': question
        })
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        app.logger.error(error_msg)
        app.logger.error(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        model = data.get('model')
        message = data.get('message')
        context = data.get('context', '')
        problem_type = data.get('problemType', '')

        if not model or not message:
            return jsonify({'error': 'Invalid input'}), 400

        logging.info(f"Received chat request - Model: {model}, Message: {message}, Problem Type: {problem_type}")

        if model == 'claude':
            result = chat_with_claude(message, context, problem_type)
        elif model == 'gpt4':
            result = chat_with_gpt4(message, context, problem_type)
        else:
            return jsonify({'error': 'Invalid model specified'}), 400

        return jsonify({'result': result})
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        app.logger.error(error_msg)
        app.logger.error(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

# Update the solve_with_claude and solve_with_gpt4 functions to handle business questions
def solve_with_claude(problem_type, question):
    prompt = f"{HUMAN_PROMPT} Solve the following {problem_type} problem: {question}. Format your response with numbered steps and make the final answer bold. Remove any unnecessary whitespace.{AI_PROMPT}"
    try:
        response = anthropic_client.completions.create(
            model="claude-2",
            prompt=prompt,
            max_tokens_to_sample=1000
        )
        formatted_response = format_response(response.completion)
        return formatted_response
    except Exception as e:
        logging.error(f"Error in solve_with_claude: {str(e)}")
        return f"Error: {str(e)}"

def solve_with_gpt4(problem_type, question):
    prompt = f"Solve the following {problem_type} problem: {question}. Format your response with numbered steps and make the final answer bold. Remove any unnecessary whitespace."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a {problem_type} expert. Provide detailed and accurate solutions to {problem_type} problems."},
                {"role": "user", "content": prompt}
            ]
        )
        formatted_response = format_response(response.choices[0].message.content)
        return formatted_response
    except Exception as e:
        logging.error(f"Error in solve_with_gpt4: {str(e)}")
        return f"Error: {str(e)}"

def format_response(response):
    # Split the response into lines
    lines = response.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('Step '):
            formatted_lines.append(line)
        elif line.lower().startswith('final answer:'):
            formatted_lines.append(f"<strong>{line}</strong>")
        elif line:
            formatted_lines.append(line)
    
    return '<br>'.join(formatted_lines)

def chat_with_claude(message, context, problem_type):
    prompt = f"{HUMAN_PROMPT} {context}\nUser: {message}\nAssistant: As a {problem_type} expert, I'll help you with your question.{AI_PROMPT}"
    try:
        response = anthropic_client.completions.create(
            model="claude-2",
            prompt=prompt,
            max_tokens_to_sample=1000
        )
        return response.completion
    except Exception as e:
        logging.error(f"Error in chat_with_claude: {str(e)}")
        return f"Error: {str(e)}"

def chat_with_gpt4(message, context, problem_type):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a {problem_type} expert. Provide detailed and accurate answers to {problem_type}-related questions."},
                {"role": "assistant", "content": context},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in chat_with_gpt4: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    logging.info(f"Starting application on port: {port}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logging.error(f"Error starting the application: {e}")
        logging.error(traceback.format_exc())

    logging.info("Application exiting.")
