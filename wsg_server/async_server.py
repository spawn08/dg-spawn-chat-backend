from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

from flask_server import app
from flask_server import load_models
import argparse

if __name__ == '__main__':
    load_models()
    parser = argparse.ArgumentParser(description='Input the port from user')
    parser.add_argument('--port', type=int, help='Port number for which application to be run')
    args = parser.parse_args()
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(args.port, 'localhost')
    print('port number ->' + str(args.port))
    IOLoop.instance().start()
    print("Server Started")
