# kkbox_client.py
from kkbox_developer_sdk.auth_flow import KKBOXOAuth
from kkbox_developer_sdk.api import KKBOXAPI
from config import CLIENT_ID, CLIENT_SECRET


def get_kkbox_api():
    # 建立 OAuth 物件
    auth = KKBOXOAuth(CLIENT_ID, CLIENT_SECRET)

    # 用 client credentials flow 取得 access token
    token = auth.fetch_access_token_by_client_credentials()

    # 用 token 建 KKBOXAPI 物件
    kkbox = KKBOXAPI(token)

    return kkbox
