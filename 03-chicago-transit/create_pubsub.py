from dotenv import load_dotenv
import os
load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
from google.cloud import pubsub_v1

project_id = os.getenv('GCP_PROJECT_ID')

# Create topic
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, 'cta-transit-events')
try:
    topic = publisher.create_topic(request={"name": topic_path})
    print(f'Topic created: {topic.name}')
except Exception as e:
    if 'ALREADY_EXISTS' in str(e):
        print(f'Topic already exists: {topic_path}')
    else:
        raise

# Create subscription
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, 'cta-transit-events-sub')
try:
    subscription = subscriber.create_subscription(
        request={"name": subscription_path, "topic": topic_path}
    )
    print(f'Subscription created: {subscription.name}')
except Exception as e:
    if 'ALREADY_EXISTS' in str(e):
        print(f'Subscription already exists: {subscription_path}')
    else:
        raise

subscriber.close()
print('Pub/Sub setup complete.')