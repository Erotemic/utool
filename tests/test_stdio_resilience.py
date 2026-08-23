import sys

from loguru import logger
import utool as ut


def test_toc_without_stdout(monkeypatch):
    messages = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']))
    try:
        monkeypatch.setattr(sys, 'stdout', None)
        timer = ut.tic('windowed timer')
        elapsed = ut.toc(timer)
    finally:
        logger.remove(sink_id)
    assert elapsed >= 0
    assert any('windowed timer' in message for message in messages)


def test_utool_stream_helpers_without_stdio(monkeypatch):
    monkeypatch.setattr(sys, 'stdout', None)
    monkeypatch.setattr(sys, '__stdout__', None)
    stream = ut.util_logging._utool_stdout()
    assert stream.write('discarded') == len('discarded')
    stream.flush()
    ut.util_logging._utool_write()('discarded')
    ut.util_logging._utool_flush()()
    ut.util_logging._utool_print()('discarded')


def test_cmd2_verbose_without_stdout(monkeypatch):
    monkeypatch.setattr(sys, 'stdout', None)
    info = ut.cmd2(
        sys.executable + ' -c \"print(123)\"',
        verbose=1,
    )
    assert info['ret'] == 0
    assert '123' in info['out']
