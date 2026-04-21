from dotenv import load_dotenv
import os
load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
from google.cloud import bigquery

client = bigquery.Client(project=os.getenv('GCP_PROJECT_ID'))

# Create dataset
dataset_id = f"{client.project}.chicago_transit"
dataset = bigquery.Dataset(dataset_id)
dataset.location = 'US'
dataset = client.create_dataset(dataset, exists_ok=True)
print(f'Dataset created: {dataset.dataset_id}')