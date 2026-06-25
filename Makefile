server:
	uv run fastapi dev app/main.py

client:
	uv run streamlit run app/client/streamlit_app.py
