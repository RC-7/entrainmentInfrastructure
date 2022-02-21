data "archive_file" "lambda_auth" {
  type = "zip"

  source_dir  = "${path.module}/../lambda_functions/authentication"
  output_path = "${path.module}/../lambda_functions/authentication/authentication.zip"
}

resource "aws_s3_bucket_object" "auth" {
  bucket = aws_s3_bucket.lambda_bucket.id

  key    = "auth.zip"
  source = data.archive_file.lambda_auth.output_path

  etag = filemd5(data.archive_file.lambda_auth.output_path)
}


resource "aws_iam_role" "lambda_exec_auth" {
  name = "auth_lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Sid    = ""
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_policy_auth" {
  name = "lambda_policy_auth"
  role = aws_iam_role.lambda_exec_auth.id

  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [{
      "Effect" : "Allow",
      "Action" : [
        "dynamodb:PutItem",
        "dynamodb:BatchGetItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan",
      ],
      "Resource" : "${aws_dynamodb_table.experimentTable.arn}"
      },
    ]
  })
}

resource "aws_lambda_function" "auth" {
  function_name = "Authentication"

  s3_bucket = aws_s3_bucket.lambda_bucket.id
  s3_key    = aws_s3_bucket_object.auth.key

  runtime = "python3.8"
  handler = "authentication.lambda_handler"

  source_code_hash = data.archive_file.lambda_auth.output_base64sha256

  role = aws_iam_role.lambda_exec_auth.arn

  environment {
    variables = {
      tableName     = "${var.table_name}"
      email_address = "${var.email_address}"
      password      = ""
      controlCount = ""
      interventionCount=""

    }
  }
}


resource "aws_api_gateway_rest_api" "authentication_api" {
  name        = "authentication_lambda"
  description = "Lambda for autherizing participants"
}

resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.authentication_api.id
  parent_id   = aws_api_gateway_rest_api.authentication_api.root_resource_id
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_method" "proxy" {
  rest_api_id      = aws_api_gateway_rest_api.authentication_api.id
  resource_id      = aws_api_gateway_resource.proxy.id
  http_method      = "ANY"
  authorization    = "NONE"
  api_key_required = "true"
}

resource "aws_api_gateway_integration" "lambda" {
  rest_api_id = aws_api_gateway_rest_api.authentication_api.id
  resource_id = aws_api_gateway_method.proxy.resource_id
  http_method = aws_api_gateway_method.proxy.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.auth.invoke_arn
}

resource "aws_api_gateway_method" "proxy_root" {
  rest_api_id      = aws_api_gateway_rest_api.authentication_api.id
  resource_id      = aws_api_gateway_rest_api.authentication_api.root_resource_id
  http_method      = "ANY"
  authorization    = "NONE"
  api_key_required = "true"
}

resource "aws_api_gateway_integration" "lambda_root" {
  rest_api_id = aws_api_gateway_rest_api.authentication_api.id
  resource_id = aws_api_gateway_method.proxy_root.resource_id
  http_method = aws_api_gateway_method.proxy_root.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.auth.invoke_arn
}

resource "aws_api_gateway_deployment" "authentication" {
  depends_on = [
    aws_api_gateway_integration.lambda,
    aws_api_gateway_integration.lambda_root,
  ]

  rest_api_id = aws_api_gateway_rest_api.authentication_api.id
  stage_name  = "experiment"
}

resource "aws_api_gateway_stage" "experiment_auth" {
  deployment_id = aws_api_gateway_deployment.authentication.id
  rest_api_id   = aws_api_gateway_rest_api.authentication_api.id
  stage_name    = "experiment_auth"
}

resource "aws_api_gateway_usage_plan" "auth_usage_plan" {
  name = "auth_usage_plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.authentication_api.id
    stage  = aws_api_gateway_deployment.authentication.stage_name
  }
  throttle_settings {
    burst_limit = var.throttle_burst_limit
    rate_limit  = var.throttle_rate_limit
  }
  quota_settings {
    limit  = var.quota_limit
    period = var.quota_preiod
  }
}

resource "aws_api_gateway_api_key" "auth_api_key" {
  name        = "auth api key"
  description = "API key to access the authentication lambda"
}

resource "aws_api_gateway_usage_plan_key" "auth_usage_plan_key" {
  key_id        = aws_api_gateway_api_key.auth_api_key.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.auth_usage_plan.id
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.auth.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_rest_api.authentication_api.execution_arn}/*/*"
}

# TODO Unify with entrainment resource
resource "aws_iam_role_policy_attachment" "lambda_policy_auth" {
  role       = aws_iam_role.lambda_exec_auth.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}