import inspect
import utool


def func1(arg1, arg2):
    pass


def func2(arg1, arg2,  arg4=5, *args, **kwargs):
    pass


def func3(arg1, arg2, arg4=5, **kwargs):
    pass


def func4(*args):
    pass


def func5(**kwargs):
    pass


def func6(*args, **kwargs):
    pass


def decor(func):
    @utool.accepts_scalar_input2([0, 4])
    def wrp(*args, **kwargs):
        return func(*args, **kwargs)
    wrp = utool.preserve_sig(wrp, func, force=True)
    return wrp


class BoringTestClass(object):
    def __init__(self):
        pass

    @decor
    def method1(self, *args, **kwargs):
        print('method1')

    @decor
    def method2(self, a, b, c=True, d='343', *args_, **kwargs_):
        print('method2')
        print('method2 a=%r' % (a,))
        print('method2 b=%r' % (b,))
        print('method2 c=%r' % (c,))
        print('method2 d=%r' % (d,))
        print('method2 args_=%r' % (args_,))
        print('method2 kwargs_=%r' % (kwargs_,))

    @decor
    def method3(self):
        print('method3')

    @decor
    def getter_eggs(self, eggs):
        print('getter_eggs')
        print('eggs = %r' % (eggs,))

    @decor
    def getter_spam(self, spam_input_, *spamargs, **spamkwds):
        print('getter_spam')
        print('spam_input_ = %r' % (spam_input_,))
        print('spamargs = %r' % (spamargs,))
        print('spamkwds = %r' % (spamkwds,))

    @decor
    def setter(self, input_, values):
        print('setter')
        print('input_ = %r' % (input_,))

    @decor
    def setter_special(self, input_, values, **kwargs):
        print('setter_special')
        print('')


def print_argspec(func):
        print('------------')
        print('func_name = %r' % func.func_name)
        # Extract argspec from orig function
        argspec = inspect.getargspec(func)
        # Get the function definition signature
        defsig = inspect.formatargspec(*argspec)
        # Get function call signature (no defaults)
        callsig = inspect.formatargspec(*argspec[0:3])
        print('argspec = %r' % (argspec,))
        print('callsig = %r' % (callsig,))
        print('defsig = %r' % (defsig,))
        print('------------')


def main():
    print('Testing normal func')
    for func in [func1, func2, func3, func4, func5, func6]:
        print_argspec(func)

    utool.rrrr()
    utool.util_decor.rrr()

    self = BoringTestClass()
    print('Testing class methods')
    for func in [self.method1, self.method2, self.getter_spam, self.getter_eggs]:
        print_argspec(func)
        if hasattr(func.im_func, '_utinfo'):
            print(func.im_func._utinfo['src'])
        print(func.im_func.func_code.co_name)
        print('<<')

    self.method2('a', 'b')
    print('---')
    self.method2('a', 'b', c='c')
    print('---')
    self.method2('a', 'b', 'c')
    print('---')
    self.method2('a', 'b', d='d')
    print('---')
    self.method2('a', 'b', x='x')
    print('---')
    self.method2('a', 'b', 'c', 'd', 'e', 'f', 'x')
    print('---')


if __name__ == '__main__':
    main()
