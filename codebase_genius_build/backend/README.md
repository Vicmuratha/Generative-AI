# Backend README
1. Create venv and activate
   python3 -m venv venv
   source venv/bin/activate
2. Install requirements
   pip install gitpython graphviz python-dotenv google-generativeai
3. Ensure Graphviz is installed on your OS for diagrams
4. Run Jac server (in BE/v1)
   cd backend/BE/v1
   jac serve main.jac
5. Run Streamlit frontend
   cd frontend/FE
   streamlit run app.py
