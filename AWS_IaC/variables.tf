# Input variable definitions

variable "aws_region" {
  description = "AWS region for all resources."

  type    = string
  default = "eu-west-1"
}

#############################################
################ Cloudwatch #################
#############################################

variable "cloudwatch_retention" {
  description = "Cloudwatch log retention policy"
  default     = 7
}

#############################################
################ Table setup ################
#############################################


variable "table_name" {
  description = "Dynamodb table name (space is not allowed)"
  default     = "LearningOptimiserData"
}

variable "table_billing_mode" {
  description = "Controls how you are charged for read and write throughput and how you manage capacity."
  default     = "PROVISIONED"
}


variable "table_read_capacity" {
  description = "The number of read units for this index. Must be set if billing_mode is set to PROVISIONED."
  default     = 5
}

variable "table_write_capacity" {
  description = "The number of write units for this index. Must be set if billing_mode is set to PROVISIONED."
  default     = 5
}

variable "table_hash_key" {
  description = " The attribute to use as the hash (partition) key. Must also be defined as an attribute."
  default     = "participantID"
}

variable "table_range_key" {
  description = "The attribute to use as the range (sort) key. Must also be defined as an attribute."
  default     = "timestamp"
}

variable "environment" {
  description = "Name of environment"
  default     = "experiment"
}