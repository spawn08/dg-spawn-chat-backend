import argparse

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

from flask_server import app
from train_model import LoadModel
import tensorflow as tf

if __name__ == '__main__':
    print(tf.version)
    load_models = LoadModel()
    load_models.load_current_model()
    parser = argparse.ArgumentParser(description='Input the port from user')
    parser.add_argument('--port', type=int, help='Port number for which application to be run')
    parser.add_argument('--host',type=str, help= 'Hostname for the application')
    args = parser.parse_args()
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(args.port, args.host)
    print('port number ->' + str(args.port))
    print("Server Started")
    IOLoop.instance().start()
