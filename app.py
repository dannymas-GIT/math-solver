from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback
from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anthropic client
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/test')
def test():
    return jsonify({"message": "Server is running!"}), 200

def format_exponents(text):
    # Replace ^2, ^3, etc. with superscript HTML
    text = re.sub(r'\^(\d+)', r'<sup>\1</sup>', text)
    # Replace x^2, x^3, etc. with x²,x³, etc.
    text = re.sub(r'x\^2', 'x²', text)
    text = re.sub(r'x\^3', 'x³', text)
    text = re.sub(r'x\^4', 'x⁴', text)
    text = re.sub(r'x\^5', 'x⁵', text)
    text = re.sub(r'x\^6', 'x⁶', text)
    text = re.sub(r'x\^7', 'x⁷', text)
    text = re.sub(r'x\^8', 'x⁸', text)
    text = re.sub(r'x\^9', 'x⁹', text)
    return text

def format_output(text):
    # Remove excess whitespace
    text = re.sub(r'\n\s*\n', '\n', text.strip())
    
    # Bold the final answer (including the actual answer)
    text = re.sub(r'(Final answer|The answer is|Result):?\s*(.+)', r'<strong>Final Answer: \2</strong>', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Convert numbered steps into an HTML list
    lines = text.split('\n')
    in_list = False
    formatted_lines = []
    step_counter = 1
    for line in lines:
        if re.match(r'^\d+[\)\.]\s', line):
            if not in_list:
                formatted_lines.append('<ol>')
                in_list = True
            formatted_lines.append(f'<li>{line[line.index(" ")+1:]}</li>')
            step_counter += 1
        else:
            if in_list:
                formatted_lines.append('</ol>')
                in_list = False
                step_counter = 1
            formatted_lines.append(line)
    if in_list:
        formatted_lines.append('</ol>')
    
    return '<br>'.join(formatted_lines)

def solve_with_claude(problem_type, expression):
    prompt = f"{HUMAN_PROMPT} Solve the following {problem_type} problem and explain the steps. Number each step starting from 1. Clearly state the final answer at the end: {expression}{AI_PROMPT}"
    
    try:
        response = anthropic_client.completions.create(
            model="claude-2",
            prompt=prompt,
            max_tokens_to_sample=1000
        )
        return format_output(format_exponents(response.completion))
    except Exception as e:
        logging.error(f"Error in solve_with_claude: {str(e)}")
        return f"Error: {str(e)}"

def solve_with_gpt4(problem_type, expression):
    prompt = f"Solve the following {problem_type} problem and explain the steps. Number each step. Clearly state the final answer at the end: {expression}"
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a mathematical assistant. Solve problems step by step, numbering each step, and clearly state the final answer at the end."},
                {"role": "user", "content": prompt}
            ]
        )
        return format_output(format_exponents(response.choices[0].message.content))
    except Exception as e:
        logging.error(f"Error in solve_with_gpt4: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/api/solve', methods=['POST'])
def solve_math():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        problem_type = data.get('problemType')
        expression = data.get('expression')

        if not problem_type or not expression:
            return jsonify({'error': 'Invalid input'}), 400

        logging.info(f"Received request - Problem Type: {problem_type}, Expression: {expression}")

        claude_result = solve_with_claude(problem_type, expression)
        gpt4_result = solve_with_gpt4(problem_type, expression)

        return jsonify({
            'claude_result': claude_result,
            'gpt4_result': gpt4_result,
            'problem_type': problem_type,
            'expression': expression
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

        if not model or not message:
            return jsonify({'error': 'Invalid input'}), 400

        logging.info(f"Received chat request - Model: {model}, Message: {message}")

        if model == 'claude':
            prompt = f"{HUMAN_PROMPT} Context: {context}\n\nHuman: {message}{AI_PROMPT}"
            response = anthropic_client.completions.create(
                model="claude-2",
                prompt=prompt,
                max_tokens_to_sample=1000
            )
            result = format_output(format_exponents(response.completion))
        elif model == 'gpt4':
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a mathematical assistant. Provide further explanations and clarifications based on the context and user's question."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {message}"}
                ]
            )
            result = format_output(format_exponents(response.choices[0].message.content))
        else:
            return jsonify({'error': 'Invalid model specified'}), 400

        return jsonify({'result': result})
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        app.logger.error(error_msg)
        app.logger.error(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    logging.info(f"Starting application on port: {port}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logging.error(f"Error starting the application: {e}")
        logging.error(traceback.format_exc())

    logging.info("Application exiting.")
