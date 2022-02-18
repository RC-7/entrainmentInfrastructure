from pkgutil import get_data
import boto3
import json
import os
import datetime
from boto3.dynamodb.conditions import Key
from boto3.dynamodb.conditions import Attr

TABLE_NAME = os.getenv('tableName')

dynamodb = boto3.resource('dynamodb')
EXPERIMENT_TABLE = dynamodb.Table(TABLE_NAME)


def build_entrainment_filter(_):
    time_five_min_ago = str(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5))
    fe = ""
    fe += "Key('timestamp').gt('" + time_five_min_ago + "') & "
    fe += "Attr('customEntrainment').begins_with('" + "{" + "')"
    return fe


def build_generic_filter(requestValues):
    filter_values = requestValues['projectionValues']
    table_keys = ['timestamp', 'participantID']
    key_actions = ['gt', 'between', 'le', 'lt', 'ge', 'begins_with']

    fe = ""

    for key in filter_values:
        if filter_values[key]['condition'] not in key_actions:
            return False

        if key in table_keys:
            fe += "Key('" + key + "')."
        else:
            fe += "Attr('" + key + "')."
        if filter_values[key]['condition'] == 'between':
            filter_value = filter_values[key]['value']
            fe += key + "('" + filter_value[0] + "','" + filter_value[1] + "')"
        else:
            fe += key + "('" + filter_values[key]['value'] + "')"
        fe += ' & '
    return fe[:-2]


def respond(body, status):
    return {
        'statusCode': status,
        'body': json.dumps(body),
        'headers': {
            'Content-Type': 'application/json',
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
            "Access-Control-Allow-Origin": "*",
            "X-Requested-With": "*"
        },
    }


def get_data(filter_expression, projection_attributes):
    response = EXPERIMENT_TABLE.scan(FilterExpression=eval(filter_expression),
                                     ProjectionExpression=projection_attributes)

    data_count = response['Count']
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = EXPERIMENT_TABLE.query(FilterExpression=eval(filter_expression),
                                          ProjectionExpression=projection_attributes)
        data.append(response['Items'])
        data_count += response['Count']

    if data_count > 0:
        return [data, '200']
    else:
        return ['No results found for query', '404']


def lambda_handler(event, _):
    print(event)
    header_lookup = {
        "entrainmentSettings": build_entrainment_filter,
        "generic": build_generic_filter
    }

    header_value = json.loads(event['headers'])
    print(header_value)
    request_values = json.loads(event['body'])
    projection_attributes = request_values['projectionAttributes']
    filter_expression = header_lookup[header_value](request_values)
    if not filter_expression:
        return respond('Invalid filter expression', '404')
    [data, status] = get_data(filter_expression, projection_attributes)
    return respond(data, status)
