import aiohttp
from aiohttp import ClientSession
from multiprocessing import Pool
import json
pool = Pool(processes=1)

notif_session = None
reg_id = "dKueowBn8K4:APA91bFv-DXrCGGM0MPzhkdpV3EAeGgxbZ--kRLCv-twFPEAZGB0gBb6TEK0Z1kLAqwwWoSWrcdBI2X1nmy0n8tDK03S1xsMlc_eTCXmWgVpJ2mvnraeHmfAArQe8tsc_l4k-Bnw_RAY"
notif_message = "Dear {username}, Your model training has been complete. Your virtual assistant is ready to answer your queries. \n Regards, \n SpawN AI Team"
NOTIFICATION_URL = "https://fcm.googleapis.com/fcm/send"


async def send_notification(reg_id: str, username: str):
    global notif_message
    global notif_session

    if username is None:
        username = "User"

    message = notif_message.format(username=username)
    payload_data = {"data": {"title": "BotBuilder: SpawN AI", "body": message, "type": "default"},
                    "registration_ids": [reg_id]}
    # print(payload_data)
    headers = {"Content-Type": "application/json", "Authorization": "key=AIzaSyBxYCj9Aw6RrI_gsshp1tISVWebR1uScL4"}
    if notif_session == None:
        notif_session = ClientSession()

    async with notif_session.request('post',NOTIFICATION_URL, data=json.dumps(payload_data), headers=headers) as resp:
        respose = await resp.json()
        print(respose)
        notif_session.close()
    pass


# result = pool.apply_async(send_notification, [reg_id ,"spawnai"])
import asyncio

asyncio.run(send_notification(reg_id=reg_id, username="spawnai"))
