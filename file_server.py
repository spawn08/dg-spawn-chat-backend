from sanic import Sanic
from sanic import response
from signal import signal, SIGINT
from sanic import request
import asyncio
import requests
from elasticsearch import Elasticsearch
import json
from functools import wraps
import base64

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


import uvloop

app = Sanic(__name__)

file_data = {}

def check_authorization(request):
    url_pass = b"spawnai:spawn1992"
    encode_auth = base64.b64encode(url_pass)
    auth = b'Basic %s' % encode_auth
    if auth.decode('utf-8') == request.token:
        return True
    else:
        return False


def authorized():
    def decorator(f):
        @wraps(f)
        async def decorated_function(request,*args,**kwargs):
            isauthorized = check_authorization(request)
            if isauthorized:
                response1 = await f(request,*args,**kwargs)
                return response1
            else:
                message = {'message': 'You are not authorized user to access this url'}
                return response.text(str(message))
        return decorated_function
    return decorator

@app.route('/api/getFile', methods=["GET"])
@authorized()
async def classify(request):
    file_name = request.args.get('fileName')



    if(file_name is not None):
        res = es.get('spawnai',doc_type='spawnai_file',id=file_name)
        bot_data_response = res['_source']
        print(bot_data_response)
        return response.json(bot_data_response)
    else:
        print('Error')
        return response.json({'msg':'Error processing request','status':'false'})


asyncio.set_event_loop(uvloop.new_event_loop())
server = app.create_server(host="0.0.0.0", port=8090, return_asyncio_server=True)
loop = asyncio.get_event_loop()
task = asyncio.ensure_future(server)
signal(SIGINT, lambda s, f: loop.stop())
try:
    loop.run_forever()
except:
    loop.stop()

