#########################################
############ Dynamo DB table ############
#########################################

resource "aws_dynamodb_table" "experimentTable" {
  name           = var.table_name
  billing_mode   = var.table_billing_mode
  hash_key       = var.table_hash_key
  range_key      = var.table_range_key
  read_capacity  = var.table_read_capacity
  write_capacity = var.table_write_capacity

  # Key attributes

  attribute {
    name = "participantID"
    type = "S"
  }
  attribute {
    name = "timestamp"
    type = "S"
  }

# TODO Add as GSI

}