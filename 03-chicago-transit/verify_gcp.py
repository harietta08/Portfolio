from dotenv import load_dotenv
import os
load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

from google.cloud import bigquery, pubsub_v1

project = os.getenv('GCP_PROJECT_ID')
client = bigquery.Client(project=project)

# Check dataset
dataset = client.get_dataset('chicago_transit')
print(f'Dataset: {dataset.dataset_id} in {dataset.location}')

# Check tables
tables = list(client.list_tables('chicago_transit'))
print(f'Tables: {[t.table_id for t in tables]}')

# Check Pub/Sub topic
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project, os.getenv('PUBSUB_TOPIC_ID'))
topic = publisher.get_topic(request={"topic": topic_path})
print(f'Topic: {topic.name}')

# Check subscription
subscriber = pubsub_v1.SubscriberClient()
sub_path = subscriber.subscription_path(project, os.getenv('PUBSUB_SUBSCRIPTION_ID'))
sub = subscriber.get_subscription(request={"subscription": sub_path})
print(f'Subscription: {sub.name}')
subscriber.close()

print()
print('GCP setup complete. All resources verified.')
