import datetime
import hashlib
import socket
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from nsm import NSMSession

PORT = 8000


class Server:
    def __init__(self, port: int):
        self.port = port
        self.server_socket = None
        self._key_path = "enclave.key"
        self._cert_path = "enclave.pem"
        self._private_key = None
        self._public_key = None
        self._certificate = None
        self._generate_key_and_certificate()
        self._ssl_context = self._setup_ssl_context()

    def start(self):
        self.server_socket = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        self.server_socket.bind((socket.VMADDR_CID_ANY, self.port))
        self.server_socket.listen(1)
        print(f"[enclave] Server listening on port {self.port}")
        while True:
            client_socket, client_address = self.server_socket.accept()
            try:
                secure_socket = self._ssl_context.wrap_socket(
                    client_socket, server_side=True
                )
                print(f"[enclave] TLS established with {client_address}")
                data = secure_socket.recv(4096)
                nonce = bytes.fromhex(data.decode("utf-8"))
                print(f"[enclave] Nonce: {nonce}")
                secure_socket.sendall(self._get_attestation_document(nonce))
            except Exception as e:
                print(f"[enclave] Error handling client: {e}")
            finally:
                secure_socket.close()
                print(f"[enclave] TLS closed with {client_address}")

    def _generate_key_and_certificate(self):
        curve_order_hex = "01fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa51868783bf2f966b7fcc0148f709a5d03bb5c9b8899c47aebb6fb71e91386409"
        curve_order = int(curve_order_hex, 16)

        with NSMSession() as nsm:
            while True:
                random_bytes = nsm.get_random_bytes(66)
                private_value = int.from_bytes(random_bytes, byteorder="big")
                if 0 < private_value < curve_order - 1:
                    self._private_key = ec.derive_private_key(private_value, ec.SECP521R1())
                    self._public_key = self._private_key.public_key()
                    break

        subject = issuer = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "enclave")]
        )
        self._certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(self._public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=10))
            .sign(self._private_key, hashes.SHA256())
        )

        with open(self._cert_path, "wb") as cert_file:
            cert_file.write(self._certificate.public_bytes(serialization.Encoding.PEM))
        with open(self._key_path, "wb") as key_file:
            key_file.write(
                self._private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

    def _setup_ssl_context(self):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=self._cert_path, keyfile=self._key_path)
        ctx.load_verify_locations(cafile="/app/certs/client.pem")
        ctx.check_hostname = False  # Not helpful for self-signed certs
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.verify_flags = ssl.VERIFY_X509_STRICT
        # ctx.set_ciphers("TLS_AES_256_GCM_SHA384")
        # ctx.set_alpn_protocols(["h2", "http/1.1"])
        # ctx.set_ecdh_curve("secp521r1")

        ctx.options |= ssl.OP_NO_TICKET
        ctx.options |= ssl.OP_SINGLE_ECDH_USE
        ctx.options |= ssl.OP_SINGLE_DH_USE
        return ctx

    def _get_attestation_document(self, nonce):
        certificate_hash = hashlib.sha256(self._certificate.public_bytes(serialization.Encoding.DER)).digest()
        with NSMSession() as nsm:
            return nsm.get_attestation_document(
                user_data=certificate_hash,
                nonce=nonce,
                public_key=self._public_key.public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ),
            )


if __name__ == "__main__":
    print("[enclave] Starting server")
    server = Server(PORT)
    try:
        server.start()
    except KeyboardInterrupt:
        print("[enclave] Server stopped")
    except Exception as e:
        print(f"[enclave] Unexpected error: {e}")
