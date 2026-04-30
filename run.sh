backend 
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8010 --reload

front 
python -m streamlit run frontend/app.py
