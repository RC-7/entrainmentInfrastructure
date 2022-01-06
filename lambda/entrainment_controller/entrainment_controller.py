import boto3
import json
import os
from boto3.dynamodb.conditions import Key
from boto3.dynamodb.conditions import Attr

TABLE_NAME = os.getenv('tableName')
PARTICIPANT_ID = os.getenv('participantID')
EXPERIMENT_START_TIME = os.getenv('experimentStartTime')

DEFAULT_VALUES = {
    'frequency': 30
}

dynamodb = boto3.resource('dynamodb')
EXPERIMEN_TABLE = dynamodb.Table(TABLE_NAME)

def buildFilterExpression ():
    fe = ""
    fe += "Key('timestamp').gt('" + EXPERIMENT_START_TIME + "') & "
    fe += "Key('participantID').eq('" + PARTICIPANT_ID +"') & "
    fe += "Attr('customEntrinment').begins_with('" + "{" +"')"
    return fe

def respond(body):
    return {
        'statusCode': '200',
        'body': json.dumps(body),
        'headers': {
            'Content-Type': 'application/json',
            "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods" : "OPTIONS,POST",
            "Access-Control-Allow-Origin" : "*",
            "X-Requested-With" : "*"
        },
    }

# TODO Return only the most recent settings
def get_entrainment_settings():
    fe = buildFilterExpression()
    projection_attributes = 'customEntrinment'

    response = EXPERIMEN_TABLE.scan(FilterExpression = eval(fe), ProjectionExpression = projection_attributes)

    custom_settings_count = response['Count']
    custom_settings = response['Items']

    while 'LastEvaluatedKey' in response:
        response = EXPERIMEN_TABLE.query(FilterExpression = eval(fe), ProjectionExpression = projection_attributes)
        custom_settings.append(response['Items'])
        custom_settings_count += response['Count']

    if custom_settings_count > 0:
        return custom_settings
    else:
        return DEFAULT_VALUES


def lambda_handler(event, context):
    custom_entrainment_settings = get_entrainment_settings()
    return respond(custom_entrainment_settings)