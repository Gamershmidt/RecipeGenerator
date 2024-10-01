from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Define default arguments for the DAG
default_args = {
    'owner': 'your_name',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG (schedule_interval is set to every 5 minutes)
dag = DAG(
    'Symptom2disease',
    default_args=default_args,
    description='A pipeline that preprocesses data, trains a model, and deploys it.',
    schedule_interval='*/5 * * * *',  # This schedules the DAG to run every 5 minutes
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Task 1: Data Preprocessing - runs the data preprocessing script
preprocess_data = BashOperator(
    task_id='preprocess_data',
    bash_command='python /home/sofia/Документы/Symptom2Disease/code/datasets/data_processing.py',
    dag=dag,
)

# Task 2: Model Training - runs the model training script
train_model = BashOperator(
    task_id='train_model',
    bash_command="python /home/sofia/Документы/Symptom2Disease/code/models/training.py",
    dag=dag,
)

# Task 3: Build and Deploy Docker Container - builds the Docker image using docker-compose
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command='docker-compose -f /home/sofia/Документы/Symptom2Disease/docker-compose.yml up --build',
    dag=dag,
)

# Set task dependencies: preprocessing -> training -> deployment
preprocess_data >> train_model >> deploy_model
