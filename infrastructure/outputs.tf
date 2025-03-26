output "public_ip" {
  description = "Public IP of server with enclave"
  value = aws_instance.server.public_ip
}