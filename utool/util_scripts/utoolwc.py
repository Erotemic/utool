import utool

if __name__ == '__main__':
    fpath_list = utool.glob('.', '*.py', recursive=True)
    wc = sum(len(utool.read_from(fpath).split(' ')) for fpath in fpath_list)
    print('word count = %r' % wc)
