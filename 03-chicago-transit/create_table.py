from dotenv import load_dotenv
import os
import json
load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
from google.cloud import bigquery

client = bigquery.Client(project=os.getenv('GCP_PROJECT_ID'))

# Load schema from our existing schema file
with open('gcp/bigquery_schema.json') as f:
    schema_json = json.load(f)

# Convert to BigQuery SchemaField objects
schema = []
for field in schema_json:
    schema.append(bigquery.SchemaField(
        name=field['name'],
        field_type=field['type'],
        mode=field['mode'],
        description=field.get('description', '')
    ))

# Create table with date partitioning on service_date
table_id = f"{client.project}.chicago_transit.transit_events"
table = bigquery.Table(table_id, schema=schema)
table.time_partitioning = bigquery.TimePartitioning(
    type_=bigquery.TimePartitioningType.DAY,
    field='service_date'
)
table.clustering_fields = ['route']

table = client.create_table(table, exists_ok=True)
print(f'Table created: {table.project}.{table.dataset_id}.{table.table_id}')
print(f'Partitioned on: service_date')
print(f'Clustered on: route')

# Also create forecast_results table
forecast_schema = [
    bigquery.SchemaField('route', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('ds', 'DATE', mode='REQUIRED'),
    bigquery.SchemaField('yhat', 'FLOAT', mode='NULLABLE'),
    bigquery.SchemaField('yhat_lower', 'FLOAT', mode='NULLABLE'),
    bigquery.SchemaField('yhat_upper', 'FLOAT', mode='NULLABLE'),
    bigquery.SchemaField('time_bucket', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('generated_at', 'TIMESTAMP', mode='NULLABLE'),
]
forecast_table_id = f"{client.project}.chicago_transit.forecast_results"
forecast_table = bigquery.Table(forecast_table_id, schema=forecast_schema)
forecast_table = client.create_table(forecast_table, exists_ok=True)
print(f'Table created: {forecast_table.project}.{forecast_table.dataset_id}.{forecast_table.table_id}')