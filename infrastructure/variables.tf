variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-central-1"
}

variable "availability_zone" {
  description = "Availability zone for the EC2 instance"
  type        = string
  default     = "eu-central-1a"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "m7g.large"
}

variable "root_volume_size" {
  description = "Size of the root volume in GB"
  type        = number
  default     = 128
}

variable "ssh_public_key" {
  description = "SSH public key for EC2 instance access"
  type        = string
  default     = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICP5rZbJFEOQhRWgTVQbnFxsmDHFevrmoK34LvuSzemu"
}

variable "key_name" {
  description = "Name of the SSH key pair"
  type        = string
  default     = "ssh-key"
}
