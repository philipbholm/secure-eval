import base64
import ctypes
import fcntl
import os

import cbor2

MAX_REQUEST_SIZE = 0x1000
MAX_RESPONSE_SIZE = 0x3000
IOCTL_MAGIC = 0x0A


class Iovec(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_void_p),  # pointer to first element of buffer
        ("len", ctypes.c_size_t),  # length of buffer
    ]


class IoctlMessage(ctypes.Structure):
    _fields_ = [
        ("request", Iovec),
        ("response", Iovec),
    ]


class NSMSession:
    def __init__(self):
        self.fd = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        if self.fd is None:
            try:
                self.fd = os.open("/dev/nsm", os.O_RDWR)
            except OSError as e:
                raise Exception(f"Failed to open NSM device: {e}")
        return self

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
    
    def get_random_bytes(self, length):
        request = cbor2.dumps("GetRandom")
        response = cbor2.loads(self._send(request))
        return response["GetRandom"]["random"][:length]
    
    def get_attestation_document(self, user_data, nonce, public_key):
        request = cbor2.dumps({
            "Attestation": {
                "user_data": user_data,
                "nonce": nonce,
                "public_key": public_key,
            }
        })
        response = cbor2.loads(self._send(request))
        return base64.b64encode(response["Attestation"]["document"])

    def _send(self, request):
        request = bytearray(request)
        response = bytearray(MAX_RESPONSE_SIZE)

        req_array = (ctypes.c_char * len(request)).from_buffer(request)
        res_array = (ctypes.c_char * len(response)).from_buffer(response)

        iovec_req = Iovec(
            base=ctypes.cast(ctypes.pointer(req_array), ctypes.c_void_p).value,
            len=len(request),
        )

        iovec_res = Iovec(
            base=ctypes.cast(ctypes.pointer(res_array), ctypes.c_void_p).value,
            len=len(response),
        )

        ioctl_msg = IoctlMessage(request=iovec_req, response=iovec_res)

        # [ioc] Command input: dir: 3, typ: 10, nr: 0, size: 32
        # [ioc] cDIRSHIFT: 30, cTYPESHIFT: 8, cNRSHIFT: 0, cSIZESHIFT: 16
        ioctl_cmd = (
            (3 << 30) | (IOCTL_MAGIC << 8) | (0 << 0) | (ctypes.sizeof(ioctl_msg) << 16)
        )

        try:
            fcntl.ioctl(self.fd, ioctl_cmd, ioctl_msg)
            result = bytes(response[: ioctl_msg.response.len])
            return result
        except Exception as e:
            print(f"[nsm] _send failed: {e}")
            raise e
