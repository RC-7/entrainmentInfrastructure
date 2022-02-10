# Output value definitions

output "lambda_bucket_name" {
  description = "Name of the S3 bucket used to store function code."
  value       = aws_s3_bucket.lambda_bucket.id
}

# output "entrainment_controller_function_name" {
#   description = "Name of the Lambda function."
#   value       = aws_lambda_function.entrainment_controller.function_name
# }

# output "entrainment_controller_base_url" {
#   description = "Base URL for API Gateway stage."
#   value       = aws_apigatewayv2_stage.entrainment_controller_api.invoke_url
# }

# output "entrainment_controller_execution_arn" {
#   description = "Execution Arn for API Gateway stage."
#   value       = aws_apigatewayv2_stage.entrainment_controller_api.execution_arn
# }

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

output "authentication_lambda_api_key" {
  value = aws_api_gateway_api_key.auth_api_key.value
  sensitive = true
}