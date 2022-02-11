import requests
import json

def valid_auth_request(lambda_url):
    header_values = {
        'x-api-key' : '',
        # 'Content-Type': 'application/json'
    }
    body_values = {
        'name': 'test_name',
        'email': 'testEmail@domain.com'
    }
    response = requests.post(lambda_url, headers=header_values, data=json.dumps(body_values))

    print('---------------------')
    print('Valid request response: ')
    print(response.text)
    print('---------------------')

def send_invalid_auth_request(lambda_url):
    response = requests.post(lambda_url)
    print('---------------------')
    print('Invalid request response: ')
    print(response.text)
    print('---------------------')


def main():
    f = open('AWS_IaC/aws_resources.json')
    name = 'yes'
    aws_resources = json.load(f)
    valid_auth_request(aws_resources['authentication_lambda_url'])

if __name__ == "__main__":
    main()