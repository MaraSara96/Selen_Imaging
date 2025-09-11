  FROM python:3.10.6-buster
  RUN pip install --upgrade pip
  COPY requirements.txt /requirements.txt
  RUN pip install -r requirements.txt
  COPY params.py /params.py
  COPY app.py /app.py
  COPY model_load.py /model_load.py
  COPY 20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras /20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras
  COPY models/ /models/
  # COPY cellpose_model_file/cpsam cellpose_model_file/cpsam
  COPY api.py /api.py
  # COPY api_testing.py /api_testing.py
  CMD uvicorn api:segmentation --host 0.0.0.0 --port $PORT
