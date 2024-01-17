#!/usr/bin/env python3

from typing import (
    cast,
    TYPE_CHECKING,
)
if TYPE_CHECKING:
    from typing import (
        Mapping,
    )

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_AUTH_URI = 'https://inb.bsc.es/auth/realms/openebench/protocol/openid-connect/token'
DEFAULT_CLIENT_ID = 'THECLIENTID'
DEFAULT_GRANT_TYPE = 'password'

def getAccessToken(oeb_credentials: "Mapping[str, str]") -> "str":
    authURI = oeb_credentials.get('authURI', DEFAULT_AUTH_URI)
    payload = {
        'client_id': oeb_credentials.get('clientId', DEFAULT_CLIENT_ID),
        'grant_type': oeb_credentials.get('grantType', DEFAULT_GRANT_TYPE),
        'username': oeb_credentials['user'],
        'password': oeb_credentials['pass'],
    }
    
    req = urllib.request.Request(authURI, data=urllib.parse.urlencode(payload).encode('UTF-8'), method='POST')
    with urllib.request.urlopen(req) as t:
        token = json.load(t)
        
        access_token = cast("str", token['access_token'])
        #logger.debug("Token {}".format(access_token))
        
        return access_token
