import requests
import time
import jwt

def generate_token(apikey:str,exp_seconds:int):
    try:
        id,secret=apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey",e)

    payload={
        "api_key":id,
        "exp":int(round(time.time()*1000))+exp_seconds*1000,
        "timestamp":int(round(time.time()*1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg":"HS256","sigh_type":"SIGN"},
    )


url="https://open.bigmodel.cn/api/paas/v4/embeddings"
headers={
    "Content-Type": "application/json",
    "Authorization":generate_token("5df05402dfe79a036798a084202ded4b.ZfNFzCP5bApmNimG",1000)
}

data={
    "model":"embedding-2",
    "messages":[{"role":"users","content":"你好"}]
}

response = requests.post(url,headers=headers,json=data)

print("Status:",response.status_code)
print("JSON:",response.json())