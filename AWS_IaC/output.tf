# Output value definitions

output "lambda_bucket_name" {
  description = "Name of the S3 bucket used to store function code."
  value       = aws_s3_bucket.lambda_bucket.id
}

output "experimentTableArn" {
  description = "The arn for the table holding experiment data."
  value       = aws_dynamodb_table.experimentTable.arn
}

output "set_score_sqs_url" {
  description = "The url for the Set Score SQS queue."
  value       = aws_sqs_queue.set_score_queue.url
}

output "authentication_lambda_url" {
  value = aws_api_gateway_deployment.authentication.invoke_url
}

output "authentication_api_key" {
  value     = aws_api_gateway_api_key.auth_api_key.value
  sensitive = true
}

output "get_data_url" {
  value = aws_api_gateway_deployment.get_data.invoke_url
}

output "get_data_api_key" {
  value     = aws_api_gateway_api_key.get_data_api_key.value
  sensitive = true
}