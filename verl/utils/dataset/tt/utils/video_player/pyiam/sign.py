import base64
import codecs
import hmac
import time
from hashlib import sha1

AUTH_PREFIX_V1 = "VARCH1-HMAC-SHA1"
DEFAULT_TTL = 3600  # in seconds
SEP = ":"


def sign_rpc_request(ak, sk, method='', caller='', extra=None, ttl=0):
    if not extra:
        extra = {}
    if ttl <= 0:
        ttl = DEFAULT_TTL
    deadline = str(int(time.time()) + ttl)

    arr = ['method=' + method, 'caller=' + caller, 'deadline=' + deadline]
    arr.extend([k + '=' + extra[k] for k in sorted(extra.keys())])

    raw = '&'.join(arr)
    hashed = hmac.new(codecs.encode(sk), codecs.encode(raw), sha1)
    dig = hashed.digest()
    ciphertext = base64.standard_b64encode(dig)
    return SEP.join([AUTH_PREFIX_V1, ak, deadline, codecs.decode(ciphertext)])


def sign_http_request(ak, sk, method, url, ttl=0, required=None):
    if not required:
        required = {}
    from urllib.parse import urlparse, parse_qsl
    q = urlparse(url)
    m = method + q.path
    caller = ''
    extraInQuery = dict(parse_qsl(q.query))
    extra = {}
    for key, value in extraInQuery.items():
        if required.get(key, False):
            extra[key] = value
    return sign_rpc_request(ak, sk, method=m, caller=caller, extra=extra, ttl=ttl)


if __name__ == '__main__':
    ak = '123'
    sk = '467'
    method = 'MGetPlayInfosV2'
    caller = 'content.arch.dianch'
    extra = {
        'vid': 'v09044b10000bnvdeqc1psvnipk18shg',
        'a': 'b',
        '1': '333'
    }
    print(sign_rpc_request(ak, sk, method, caller, extra=extra))

    url = 'http://127.0.0.1:9998/v1/api/get_video_info?vid=v09044b10000bnvdeqc1psvnipk18shg&a=b&1=333'
    print(sign_http_request(ak, sk, 'GET', url, 0, {}))
