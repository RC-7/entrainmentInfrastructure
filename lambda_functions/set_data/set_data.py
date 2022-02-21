import json
import boto3
import os

TABLE_NAME = os.getenv('tableName')


def write_score(score_data):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TABLE_NAME)
    response = table.put_item(
        Item=score_data
    )


def lambda_handler(event, context):
    data_types = ['Score', 'LevelID', 'PData', 'EntrainmentSettings']
    required_column_values = {
        'Score': ('participantID', 'score', 'timestamp', 'levelID'),
        'LevelID': ('participantID', 'startTime', 'timestamp', 'levelID', 'endTime'),
        'PData': ('participantID', 'experience', 'sex', 'age', 'timestamp'),
        'EntrainmentSettings': ('participantID', 'customEntrainment', 'session', 'timestamp')
    }

    for record in event['Records']:
        data_type = record['messageAttributes']['type']['stringValue']
        if 'type' in record['messageAttributes'] and data_type in data_types:
            body = json.loads(record["body"])
            if all(key in body for key in required_column_values[data_type]):
                write_score(body)
            else:
                raise Exception('Invalid column values, check DLQ' + event)
