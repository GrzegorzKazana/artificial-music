def flow(*functions):
    """
    composes functions left to right
    """
    def inner(arg):
        for f in functions:
            arg = f(arg)
        return arg
    return inner


def pipe(val, *functions):
    """
    pipes value through binary functions
    """
    return flow(*functions)(val)


def debug(messages):
    if 'numpy' in str(type(messages)):
        print(messages.shape)
        print(messages)
    else:
        for msg in messages:
            print(msg)
        print(len(messages))
    return messages


def unzip(l):
    return tuple(zip(*l))
