import json
import boto3
import os

TABLE_NAME = os.getenv('tableName')

def write_score(score_data):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TABLE_NAME)
    response = table.put_item(
        Item = score_data
    )

def lambda_handler(event, context):
    for record in event['Records']:
        if 'type' in record['messageAttributes'] and record['messageAttributes']['type']['stringValue'] == 'Score' :
            body = json.loads(record["body"])
            write_score(body)