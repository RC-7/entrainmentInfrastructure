resource "aws_sqs_queue" "set_score_queue" {
  name                      = var.set_score_queue
  message_retention_seconds = var.sqs_retention
  redrive_policy = jsonencode({
    "deadLetterTargetArn" : aws_sqs_queue.deadletter_queue.arn,
    "maxReceiveCount" : var.sqs_max_recieve
  })
}

resource "aws_sqs_queue" "deadletter_queue" {
  name                      = "${var.set_score_queue}-DLQ"
  message_retention_seconds = var.dlq_retention
}


data "archive_file" "lambda_set_score" {
  type = "zip"

  source_dir  = "${path.module}/../lambda_functions/set_score"
  output_path = "${path.module}/../lambda_functions/set_score/set_score.zip"
}

resource "aws_s3_bucket_object" "set_score" {
  bucket = aws_s3_bucket.lambda_bucket.id

  key    = "set_score.zip"
  source = data.archive_file.lambda_set_score.output_path

  etag = filemd5(data.archive_file.lambda_set_score.output_path)
}


resource "aws_iam_role" "lambda_exec_set_score" {
  name = "set_score_lambda"

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

resource "aws_iam_role_policy" "lambda_policy_set_score" {
  name = "lambda_policy_set_score"
  role = aws_iam_role.lambda_exec_set_score.id

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
        "Resource" : aws_sqs_queue.set_score_queue.arn
      },
    ]
  })
}

resource "aws_lambda_function" "set_score" {
  function_name = "SetScore"

  s3_bucket = aws_s3_bucket.lambda_bucket.id
  s3_key    = aws_s3_bucket_object.set_score.key

  runtime = "python3.8"
  handler = "set_score.lambda_handler"

  source_code_hash = data.archive_file.lambda_set_score.output_base64sha256

  role = aws_iam_role.lambda_exec_set_score.arn

  environment {
    variables = {
      tableName = "${var.table_name}"
    }
  }
}

# resource "aws_lambda_event_source_mapping" "event_sources_mapping_set_score" {
#   event_source_arn = aws_sqs_queue.set_score_queue.arn
#   function_name    = aws_lambda_function.set_score.arn
# }

# TODO Unify with entrainment resource
resource "aws_iam_role_policy_attachment" "lambda_policy_save_score" {
  role       = aws_iam_role.lambda_exec_set_score.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}