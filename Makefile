server:
	fastapi dev app/main.py

client:
	streamlit run app/client/streamlit_app.py