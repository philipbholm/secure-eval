# Infrastructure for SecureEval

## Requirements
- Terraform >=1.8.0

## Instance options
All 8g series are ARM-based Graviton 4 chips
Uses always-on memory encryption, dedicated cache for every vCPU
and supports Pointer Authentication and Branch Target Identification
by default with AL2023
https://spectrum.ieee.org/aws-graviton4

Use 7g series if 8g is not available. 
Graviton 2 and up encrypts the DRAM interface. Gravtion 4 also
encrypt the interface with the Nitro Cards.

M8g series: General purpose workloads
m8g.large: 2 vCPU, 8 GiB RAM
m8g.48xlarge: 192 vCPU, 768 GiB RAM

C8g series: Compute-intensive workloads
c8g.large: 2 vCPU, 4 GiB RAM
c8g.48xlarge: 192 vCPU, 382 GiB RAM

R8g series: Memory-intensive workloads
r8g.large: 2 vCPU, 16 GiB RAM
r8g.48xlarge: 96 vCPU, 1536 GiB RAM

X8g series: Even more memory-intensive workloads
x8g.large:    2 vCPU,   32 GiB RAM (0.3019 USD/hour)
x8g.48xlarge: 192 vCPU, 3072 GiB RAM (28.0608 USD/hour)


## TODO
- Explain how to update aws profile
- Restrict security group rules
- Add X86 instance options
- Test inferentia chips
- Remove the need for SSH key?
    - Try to make the proxy and enclave start automatically with user_data or similar