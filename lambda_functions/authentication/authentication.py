import boto3
import json
import os
import hashlib
from base64 import b64encode
import datetime
import uuid
import smtplib
import random
from botocore.exceptions import ClientError

from boto3.dynamodb.conditions import Attr

TABLE_NAME = os.getenv('tableName')

dynamodb = boto3.resource('dynamodb')
EXPERIMENT_TABLE = dynamodb.Table(TABLE_NAME)


def any_and_not_all(iterable):
    return any(iterable) and not all(iterable)


def respond(code, body):
    return {
        'statusCode': code,
        'body': json.dumps(body),
        'headers': {
            'Content-Type': 'application/json',
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
            "Access-Control-Allow-Origin": "*",
            "X-Requested-With": "*"
        },
    }


def query_table(fe, projection_attributes):
    response = EXPERIMENT_TABLE.scan(FilterExpression=eval(fe), ProjectionExpression=projection_attributes)
    count = response['Count']
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = EXPERIMENT_TABLE.scan(FilterExpression=eval(fe), ProjectionExpression=projection_attributes)
        data.append(response['Items'])
        count += response['Count']
    return [count, data]


# Need to expand on this if an experiment has > 2 sessions
def record_return_participant(participant_info_from_table):
    participant_info = {
        'participantID': participant_info_from_table[0]['participantID'],
        'timestamp': str(datetime.datetime.now(datetime.timezone.utc)),
        'session': '2'
    }
    create_participant_table_entry(participant_info)


def get_participant_data(hash_value):
    projection_attributes = 'participantID, assignedGroup'
    fe = build_filter_expression(hash_value)

    [participant_count, participant_info] = query_table(fe, projection_attributes)

    if participant_count == 1:
        record_return_participant(participant_info)
        return [True, *participant_info]
    elif participant_count == 0:
        return [False, 0]
    elif participant_count > 1:
        return [False, -1]


def create_participant_table_entry(participant_info):
    response = EXPERIMENT_TABLE.put_item(
        Item=participant_info
    )


def build_filter_expression(hash_value):
    fe = "Attr('hashIdentifier').eq('" + hash_value + "') & "
    fe += "Attr('assignedGroup').exists()"
    return fe


def create_hash(secret_key, name):
    hash_calculator = hashlib.md5()
    hash_calculator.update(secret_key)
    hash_calculator.update(name)
    hash_value = hash_calculator.hexdigest()
    return hash_value


def check_id(secret_key, name):
    secret_key = bytes(secret_key, 'utf-8')
    name = bytes(name, 'utf-8')
    hash_value = create_hash(secret_key, name)
    return get_participant_data(hash_value)


def email_secret_key(key, email_participant):
    email_from = os.getenv('email_address')
    reply_to = email_from

    subject = '[Entrainment Experiment] Your secret key'
    body = 'Hi <br><br> Thank you again for participating in the study. Your secret key is: <b>{key}</b>. <br><br> \
        Please keep this safe as you will need it for the next session and it is not shared with anyone else'.format(
        key=key)

    username = email_from
    password = os.getenv('password')

    gmail_smtp_server = 'smtp.gmail.com'
    google_smtp_port = 587

    message = "From: {email_from}\nTo: {email_to}\nContent-type: text/html\nReply-To: {email_reply}\nSubject: \
     {subject}\n\n{body}".format(email_from=email_from, email_to=email_participant, email_reply=reply_to,
                                 subject=subject, body=body)

    try:
        smtp_session = smtplib.SMTP(gmail_smtp_server, google_smtp_port)
        smtp_session.ehlo()
        smtp_session.starttls()
        smtp_session.login(username, password)
        smtp_session.sendmail(email_from, email_participant, message)
        smtp_session.close()
        return True
    except Exception as ex:
        print(ex)
        return False


def check_count_variables():
    count_control_group = os.getenv('controlCount')
    count_test_group = os.getenv('interventionCount')
    if count_control_group == '':
        fe = "Attr('assignedGroup').eq('C')"
        projection_attribute = 'assignedGroup'
        try:
            [control_group_count, _] = query_table(fe, projection_attribute)
            os.environ["controlCount"] = str(control_group_count)
        except ClientError as err:
            os.environ["controlCount"] = '0'
    if count_test_group == '':
        fe = "Attr('assignedGroup').eq('T')"
        projection_attribute = 'assignedGroup'
        try:
            [test_group_count, _] = query_table(fe, projection_attribute)
            os.environ["interventionCount"] = str(test_group_count)
        except ClientError as err:
            os.environ["interventionCount"] = '0'


def assign_testing_group():
    check_count_variables()
    count_control_group = os.getenv('controlCount')
    count_test_group = os.getenv('interventionCount')
    group_assigned = ''
    if count_test_group == count_control_group:
        group_assigned = random.choice(['C', 'T'])
    elif count_test_group > count_control_group:
        group_assigned = 'C'
    else:
        group_assigned = 'T'

    if group_assigned == 'C':
        os.environ["controlCount"] = str(int(count_control_group) + 1)
    else:
        os.environ["interventionCount"] = str(int(count_test_group) + 1)
    return group_assigned


def create_participant(name, email):
    key_encoded = os.urandom(5)
    secret_key = b64encode(key_encoded)
    encoded_name = bytes(name, 'utf-8')
    participant_hash = create_hash(secret_key, encoded_name)
    group = assign_testing_group()
    participant_ID = str(uuid.uuid4())

    participant_info = {
        'participantID': participant_ID,
        'timestamp': str(datetime.datetime.now(datetime.timezone.utc)),
        'hashIdentifier': participant_hash,
        'assignedGroup': group
    }
    create_participant_table_entry(participant_info)

    # Return status
    # secret_key_string = str(secret_key.decode('utf-8'))
    # status = email_secret_key(secret_key_string, email)

    return [True, participant_ID, group]


def create_negative_response_body(message, response_body):
    response_body['participant_ID'] = -1
    response_body['message'] = message


def lambda_handler(event, context):
    request_values = json.loads(event['body'])
    response_message = {
        'participant_ID': '',
        'message': '',
        'group': '',
        'session': ''
    }
    response_code = ''

    if all(key in request_values for key in ('secret_key', 'name')):
        name = request_values['name']
        secret_key = request_values['secret_key']
        participant_data = check_id(secret_key, name)
        if participant_data[0]:
            response_message['participant_ID'] = participant_data[1]['participantID']
            response_message['group'] = participant_data[1]['assignedGroup']
            response_message['session'] = 2
            response_message['message'] = 'Participant ID found'
            response_code = '200'
        else:
            response_code = '204'
            if participant_data[1] == 0:
                create_negative_response_body('No participant ID found for name and secret key provided',
                                              response_message)
            if participant_data[1] == -1:
                create_negative_response_body('Multiple participant ID\'s found', response_message)

    if any_and_not_all(key in request_values for key in ('secret_key', 'name')) and not \
            'email' in request_values.keys():
        create_negative_response_body('Not enough info given to querry participant ID', response_message)
        response_code = '404'
    if all(key in request_values for key in ('email', 'name')) and not \
            'secret_key' in request_values.keys():
        name = request_values['name']
        email = request_values['email']
        [status, participant_id, group] = create_participant(name, email)
        if status:
            response_message['participant_ID'] = participant_id
            response_message['group'] = group
            response_message['session'] = 1
            response_message['message'] = 'Participant ID created, check your email for your secret code'
            response_code = '200'
        else:
            create_negative_response_body('Error creating user', response_message)
            response_code = '404'

    if any_and_not_all(key in request_values for key in ('email', 'name')) and not \
            'secret_key' in request_values.keys():
        response_message['participant_ID'] = -1
        response_message['message'] = 'Not enough info given to create user'
        response_code = '404'

    return respond(response_code, response_message)
