terraform {
  required_version = ">= 1.8.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.92.0"
    }
  }
}

provider "aws" {
  profile = "secure-eval"
  region  = var.aws_region
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "main" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = var.availability_zone
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route_table" "main" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "main" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.main.id
}

resource "aws_security_group" "main" {
  vpc_id = aws_vpc.main.id
}

resource "aws_vpc_security_group_ingress_rule" "main" {
  security_group_id = aws_security_group.main.id

  cidr_ipv4   = "0.0.0.0/0"
  from_port   = 22
  to_port     = 22
  ip_protocol = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "main" {
  security_group_id = aws_security_group.main.id

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "-1"
}

resource "aws_key_pair" "main" {
  key_name   = var.key_name
  public_key = var.ssh_public_key
}

resource "aws_instance" "server" {
  ami                         = "ami-077e7b988e15f909f" # Amazon Linux 2023 AMI
  instance_type               = var.instance_type
  availability_zone           = var.availability_zone
  key_name                    = aws_key_pair.main.key_name
  vpc_security_group_ids      = [aws_security_group.main.id]
  subnet_id                   = aws_subnet.main.id
  associate_public_ip_address = true

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  enclave_options {
    enabled = true
  }

  metadata_options {
    http_tokens = "required"
  }

  user_data = <<-EOF
    #!/bin/bash
    # Install dependencies
    sudo dnf -y install aws-nitro-enclaves-cli aws-nitro-enclaves-cli-devel git-all

    # Update user permissions
    sudo usermod -aG ne ec2-user && sudo usermod -aG docker ec2-user

    # Allocate resources to enclave 
    # Leaves 1 vCPU and 4 GB RAM for the parent
    PARENT_MEMORY=4096
    PARENT_CPU=1
    sudo sed -i "s/^memory_mib:.*/memory_mib: $(awk '/MemTotal/ {printf "%d", $2/1024 - '"$PARENT_MEMORY"'}' /proc/meminfo)/" /etc/nitro_enclaves/allocator.yaml
    sudo sed -i "s/^cpu_count:.*/cpu_count: $(($(nproc) - $PARENT_CPU))/" /etc/nitro_enclaves/allocator.yaml

    # Enable and start services
    sudo systemctl enable --now docker
    sudo systemctl enable --now nitro-enclaves-allocator.service
    sudo systemctl enable --now nitro-enclaves-vsock-proxy.service

    # Pull enclave code
    git clone https://github.com/philipbholm/secure-eval.git /home/ec2-user/secure-eval
    sudo chown -R ec2-user:ec2-user /home/ec2-user/secure-eval

    # Aliases
    echo "alias stop='nitro-cli terminate-enclave --all'" >> .bashrc
    echo "alias desc='nitro-cli describe-enclaves'" >> .bashrc

    sudo reboot
  EOF
}
