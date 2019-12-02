# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from utool import util_inject
print, rrr, profile = util_inject.inject2(__name__)


def is_local_port_open(port):
    """
    Args:
        port (int):

    Returns:
        bool:

    References:
        http://stackoverflow.com/questions/7436801/identifying-listening-ports-using-python

    CommandLine:
        python -m utool.util_web is_local_port_open --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_web import *  # NOQA
        >>> port = 32183
        >>> assert is_local_port_open(80) is False, 'port 80 should always be closed'
        >>> assert is_local_port_open(port) is True, 'maybe this port is actually used?'
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('127.0.0.1', port))
    s.close()
    return result != 0


def find_open_port(base=5000):
    port = base
    while not is_local_port_open(port):
        port += 1
    return port


def get_localhost():
    import socket
    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        return '127.0.0.1'


def _testping():
    r"""
    CommandLine:
        python -m utool.util_web --exec-_testping

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_web import *  # NOQA
        >>> result = _testping()
        >>> print(result)
    """
    import requests
    url = 'http://%s:%s' % (get_localhost(), 5832)
    requests.post(url, data={'hello': 'world'})


def start_simple_webserver(domain=None, port=5832):
    r"""
    simple webserver that echos its arguments

    Args:
        domain (None): (default = None)
        port (int): (default = 5832)

    CommandLine:
        python -m utool.util_web --exec-start_simple_webserver:0
        python -m utool.util_web --exec-start_simple_webserver:1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_web import *  # NOQA
        >>> domain = None
        >>> port = 5832
        >>> result = start_simple_webserver(domain, port)
        >>> print(result)
    """
    import tornado.ioloop
    import tornado.web
    import tornado.httpserver
    import tornado.wsgi
    import flask
    app = flask.Flask('__simple__')
    @app.route('/', methods=['GET', 'POST', 'DELETE', 'PUT'])
    def echo_args(*args, **kwargs):
        from flask import request
        print('Simple server was pinged')
        print('args = %r' % (args,))
        print('kwargs = %r' % (kwargs,))
        print('request.args = %r' % (request.args,))
        print('request.form = %r' % (request.form,))
        return ''
    if domain is None:
        domain = get_localhost()
    app.server_domain = domain
    app.server_port = port
    app.server_url = 'http://%s:%s' % (app.server_domain, app.server_port)
    print('app.server_url = %s' % (app.server_url,))
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(app.server_port)
    tornado.ioloop.IOLoop.instance().start()


def render_html(html_str):
    """
    makes a temporary html rendering
    """
    import utool as ut
    from os.path import abspath
    import webbrowser

    try:
        html_str = html_str.decode('utf8')
    except Exception:
        pass

    html_dpath = ut.ensure_app_resource_dir('utool', 'temp_html')
    fpath = abspath(ut.unixjoin(html_dpath, 'temp.html'))
    url = 'file://' + fpath
    ut.writeto(fpath, html_str)
    webbrowser.open(url)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m utool.util_web
        python -m utool.util_web --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
