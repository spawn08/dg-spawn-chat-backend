import requests
import time

while True:
    time.sleep(60*60)
    result = requests.get('http://localhost:8000/clear_cache', headers={'Authorization':'Basic b25lYm90c29sdXRpb246T25lQm90RmluYW5jaWFsU2VydmljZXM='})
    print(result.json())
