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
    'Adbvc',
    default_args=default_args,
    description='A pipeline that preprocesses data, trains a model, and deploys it.',
    schedule_interval='*/5 * * * *',  # This schedules the DAG to run every 5 minutes
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Use the DAG context manager
with dag:

    # Task 1: Pull the latest versioned data from DVC storage
    dvc_pull = BashOperator(
        task_id='dvc_pull',
        bash_command='cd /path/to/your/dvc && dvc pull'
    )

    # Task 2: Run the data pipeline (reproduce the DVC pipeline stages)
    dvc_repro = BashOperator(
        task_id='dvc_repro',
        bash_command='cd /path/to/your/dvc && dvc repro'
    )

    # Task 3: Push the results to the DVC remote storage
    dvc_push = BashOperator(
        task_id='dvc_push',
        bash_command='cd /path/to/your/dvc && dvc push'
    )

    # Task 4: Model Training - runs the model training script
    train_model = BashOperator(
        task_id='train_model',
        bash_command="python /path/to/your/project/code/models/model_training.py",
    )

    # Task 5: Build and Deploy Docker Container - builds the Docker image using docker-compose
    deploy_model = BashOperator(
        task_id='deploy_model',
        bash_command='docker-compose -f /path/to/your/project/docker-compose.yml up --build',
    )

    start_mlflow_ui = BashOperator(
        task_id='start_mlflow_ui',
        bash_command='mlflow ui --host 0.0.0.0 --port 5000 &',  # Run in the background
        dag=dag,
    )


    # Set task dependencies: dvc pipeline -> training -> deployment
    dvc_pull >> dvc_repro >> dvc_push >> start_mlflow_ui >> train_model >> deploy_model
