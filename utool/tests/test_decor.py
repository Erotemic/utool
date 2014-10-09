import inspect


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


def main():
    for func in [func1, func2, func3, func4, func5, func6]:
        print('------------')
        # Extract argspec from orig function
        argspec = inspect.getargspec(func)
        # Get the function definition signature
        defsig = inspect.formatargspec(*argspec)
        # Get function call signature (no defaults)
        callsig = inspect.formatargspec(*argspec[0:3])
        print(argspec)
        print(callsig)
        print(defsig)
        print('------------')


if __name__ == '__main__':
    main()
