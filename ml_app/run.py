import uvicorn
import argparse
from app import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line utility for accepting port number')
    parser.add_argument('--port', type=int, help='Port number for running application')
    parser.add_argument('--host', type=str, help='Hostname for the application')
    args = parser.parse_args()
    if args.port is not None and args.host is not None:
        uvicorn.run(app, port=args.port, host=args.host)
    else: uvicorn.run(app, host='localhost', port=8000)    