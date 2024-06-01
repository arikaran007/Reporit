
import os
import tempfile
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain_openai import OpenAI
from config import WHITE, GREEN, RESET_COLOR, model_name
from utils import format_user_question
from file_processing import clone_github_repo, load_and_index_files
from questions import ask_question, QuestionContext
from flask import Flask, render_template, request, redirect, url_for

load_dotenv()
OPENAI_API_KEY = ""

app = Flask(__name__)

index = None
documents = None
file_type_counts = None
filenames = None
repo_name = None
github_url = None
conversation_history = ""
question_context = None

@app.route('/', methods=['GET', 'POST'])
def index_route():
    global index, documents, file_type_counts, filenames, repo_name, github_url, question_context
    if request.method == 'POST':
        github_url = request.form['github_url']
        repo_name = github_url.split("/")[-1]
        print("Cloning the repository...")
        with tempfile.TemporaryDirectory() as local_path:
            if clone_github_repo(github_url, local_path):
                index, documents, file_type_counts, filenames = load_and_index_files(local_path)
                if index is None:
                    print("No documents were found to index. Exiting.")
                    return "No documents were found to index."
                print("Repository cloned. Indexing files...")
            else:
                print("Failed to clone the repository.")
                return "Failed to clone the repository."

        llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)
        template = """
        Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}
        Instr:
        1. Answer based on context/docs.
        2. Focus on repo/code.
        3. Consider:
        a. Purpose/features - describe.
        b. Functions/code - provide details/samples.
        c. Setup/usage - give instructions.
        4. Unsure? Say "I am not sure".
        5. Never Answer to the question that is not relevant to the repository.
        6. Don't generate irrelevant content.
        Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["repo_name", "github_url", "conversation_history", "question", "numbered_documents",
                             "file_type_counts", "filenames"]
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        question_context = QuestionContext(index, documents, llm_chain, model_name, repo_name, github_url,
                                           conversation_history, file_type_counts, filenames)
        return redirect(url_for('ask_question_route'))
    return render_template('index.html')

@app.route('/ask_question_route', methods=['GET', 'POST'])
def ask_question_route():
    if request.method == 'POST':
        user_question = request.form['user_question']
        if user_question.lower() == "exit":
            return redirect(url_for('index_route'))
        print('Thinking...')
        user_question = format_user_question(user_question)
        answer = ask_question(user_question, question_context)
        print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
        question_context.conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
        return render_template('chat.html', answer=answer)
    return render_template('chat.html')



if __name__ == '__main__':
    app.run(debug=True)