#!/bin/bash

set -e

PUBLIC_IP=$(terraform output -raw public_ip)

ssh -i ~/.ssh/sandbox.pem ec2-user@$PUBLIC_IP
