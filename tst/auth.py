import requests
import json

def valid_auth_request(lambda_url, api_key):
    header_values = {
        'x-api-key' : api_key,
    }
    body_values = {
        'name': 'yes',
        'email': 'test@mail.com'
    }
    response = requests.post(lambda_url, headers=header_values, data=json.dumps(body_values))

    print('---------------------')
    print('Valid request response: ')
    print(response.text)
    print('---------------------')


def querry_valid_participant(lambda_url, api_key):
    header_values = {
        'x-api-key' : api_key,
    }
    body_values = {
        'name': '',
        'email': 'test@mail.com',
        'secret_key': ''
    }
    response = requests.post(lambda_url, headers=header_values, data=json.dumps(body_values))

    print('---------------------')
    print('Valid request response: ')
    print(response.text)
    print('---------------------')

def querry_invalid_secret_key(lambda_url, api_key):
    header_values = {
        'x-api-key' : '',
        # 'Content-Type': 'application/json'
    }
    body_values = {
        'name': 'test_name',
        'email': 'testEmail@domain.com',
        'secret_key': ''
    }
    response = requests.post(lambda_url, headers=header_values, data=json.dumps(body_values))

    print('---------------------')
    print('Valid request response: ')
    print(response.status_code)
    print(response.reason)
    print('---------------------')

def send_invalid_auth_request(lambda_url):
    response = requests.post(lambda_url)
    print('---------------------')
    print('Invalid request response: ')
    print(response.text)
    print('---------------------')


def main():
    f = open('runnners/aws_resources.json')
    aws_resources = json.load(f)
    lambda_url = aws_resources['authentication_lambda_url']['value']
    api_key = aws_resources['authentication_api_key']['value']
    querry_valid_participant(lambda_url, api_key)

if __name__ == "__main__":
    main()