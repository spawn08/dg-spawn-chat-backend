import json
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import HTTPException, Depends
from starlette.status import HTTP_401_UNAUTHORIZED
from config.config import Config as cfg
from aiohttp import ClientSession

security = HTTPBasic()

notif_session = None

async def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "onebotsolution" or credentials.password != "OneBotFinancialServices":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"})
    return True

async def send_notification(reg_id: str, username: str):
    
    global notif_session

    if username is None:
        username = 'User'

    message = cfg.NOTIF_MESSAGE.format(username=username)
    payload_data = {'data': {'title': 'BotBuilder: SpawN AI', 'body': message, 'type': 'default'},
                    'registration_ids': [reg_id]}
    print(payload_data)
    headers = {'Content-Type': 'application/json', 'Authorization': cfg.PROFESSOR_SPAWN_API_KEY}
    print(headers)
    if notif_session == None:
        notif_session = ClientSession()

    async with notif_session.post(cfg.NOTIFICATION_URL, data=json.dumps(payload_data), headers=headers) as resp:
        respose = await resp.json()
        print(respose)
    pass