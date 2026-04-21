from dotenv import load_dotenv
import os
load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
from google.cloud import bigquery

client = bigquery.Client(project=os.getenv('GCP_PROJECT_ID'))

query = """
SELECT route, service_date, rides, temperature_2m
FROM `key-range-494015-t7.chicago_transit.transit_events`
LIMIT 5
"""

for row in client.query(query).result():
    print(dict(row))