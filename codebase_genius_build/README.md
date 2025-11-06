
# Codebase Genius – AI-Powered Code Documentation System

**Codebase Genius** is an autonomous, multi-agent system that automatically generates high-quality documentation for software repositories. Built using **Jac (JacLang)** and Python, this system analyses a repository’s code, maps relationships, and produces human-readable Markdown documentation with diagrams.

---

## 🧠 Features

* **Automated Repository Mapping:** Clone and traverse GitHub repositories, generating a structured file-tree and summarizing README files.
* **Code Analysis:** Builds a **Code Context Graph (CCG)** capturing functions, classes, and module relationships.
* **Documentation Generation:** Produces a clean, well-organized Markdown report including project overview, installation, usage, API references, and diagrams.
* **Multi-Agent Architecture:**

  * **Code Genius (Supervisor):** Orchestrates workflow and delegates tasks.
  * **Repo Mapper:** Clones repositories and builds a file-tree map.
  * **Code Analyzer:** Constructs the CCG and parses code relationships.
  * **DocGenie:** Generates final documentation.
* **Extensible:** Designed to support multiple languages, starting with Python and Jac.

---

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Vicmuratha/Generative-AI.git
cd Generative-AI
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Set your LLM API key**
   Create a `.env` file in the root directory and add your API key (OpenAI or Gemini):

```
OPENAI_API_KEY=<your_openai_key>
GEMINI_API_KEY=<your_gemini_key>  # optional if using Gemini
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ⚙️ Usage

### 1. Run the backend Jac server

Navigate to the backend folder (choose your implementation version, e.g., `v1`) and run:

```bash
jac serve main.jac
```

This starts the Jac server exposing walkers for repository analysis and documentation generation.

### 2. Use the API

* Send a **GitHub repository URL** to the server via HTTP POST.
* Receive the generated Markdown documentation with diagrams.

### 3. Sample Output

Generated documentation will be saved under:

```
./outputs/<repo_name>/docs.md
```

It includes:

* Project overview
* Installation & usage instructions
* API reference with function/class relationships
* Diagrams illustrating code structure

---


## 📝 Contributing

1. Clone the repository and create a new branch for your feature:

```bash
git checkout -b feature/your-feature
```

2. Implement your changes and commit regularly:

```bash
git add .
git commit -m "Describe your changes"
```

3. Push your branch and open a Pull Request:

```bash
git push origin feature/your-feature
```

---

## ⚠️ NOTES

* Handle invalid URLs, private repositories, or unsupported languages gracefully.
* Test each agent individually before integrating for smoother workflow.
* For Python code parsing, external libraries like **Tree-sitter** may be used via Jac’s `py_module`.

---

## 📚 References

* [byLLM Task Manager Example](https://github.com/jaseci-labs/Agentic-AI/tree/main/task_manager/byllm)
* [Jac Beginner’s Guide](https://jaseci-labs.github.io/jac/docs/basics/)
* [Jac Language Reference](https://jaseci-labs.github.io/jac/docs/reference/)

---



