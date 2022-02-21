resource "aws_sqs_queue" "set_data_queue" {
  name                      = var.set_data_queue
  message_retention_seconds = var.sqs_retention
  redrive_policy = jsonencode({
    "deadLetterTargetArn" : aws_sqs_queue.deadletter_queue.arn,
    "maxReceiveCount" : var.sqs_max_recieve
  })
}

resource "aws_sqs_queue" "deadletter_queue" {
  name                      = "${var.set_data_queue}-DLQ"
  message_retention_seconds = var.dlq_retention
}


data "archive_file" "lambda_set_data" {
  type = "zip"

  source_dir  = "${path.module}/../lambda_functions/set_data"
  output_path = "${path.module}/../lambda_functions/set_data/set_data.zip"
}

resource "aws_s3_bucket_object" "set_data" {
  bucket = aws_s3_bucket.lambda_bucket.id

  key    = "set_data.zip"
  source = data.archive_file.lambda_set_data.output_path

  etag = filemd5(data.archive_file.lambda_set_data.output_path)
}


resource "aws_iam_role" "lambda_exec_set_data" {
  name = "set_data_lambda"

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

resource "aws_iam_role_policy" "lambda_policy_set_data" {
  name = "lambda_policy_set_data"
  role = aws_iam_role.lambda_exec_set_data.id

  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [{
      "Effect" : "Allow",
      "Action" : [
        "dynamodb:PutItem",
      ],
      "Resource" : "${aws_dynamodb_table.experimentTable.arn}"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "sqs:ChangeMessageVisibility",
          "sqs:ChangeMessageVisibilityBatch",
          "sqs:DeleteMessage",
          "sqs:DeleteMessageBatch",
          "sqs:GetQueueAttributes",
          "sqs:GetQueueUrl",
          "sqs:ReceiveMessage"
        ],
        "Resource" : aws_sqs_queue.set_data_queue.arn
      },
    ]
  })
}

resource "aws_lambda_function" "set_data" {
  function_name = "SetData"

  s3_bucket = aws_s3_bucket.lambda_bucket.id
  s3_key    = aws_s3_bucket_object.set_data.key

  runtime = "python3.8"
  handler = "set_data.lambda_handler"

  source_code_hash = data.archive_file.lambda_set_data.output_base64sha256

  role = aws_iam_role.lambda_exec_set_data.arn

  environment {
    variables = {
      tableName = "${var.table_name}"
    }
  }
}

# resource "aws_lambda_event_source_mapping" "event_sources_mapping_set_data" {
#   event_source_arn = aws_sqs_queue.set_data_queue.arn
#   function_name    = aws_lambda_function.set_data.arn
# }

# TODO Unify with entrainment resource
resource "aws_iam_role_policy_attachment" "lambda_policy_save_data" {
  role       = aws_iam_role.lambda_exec_set_data.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}