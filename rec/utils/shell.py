"""
shell
=====
A better way to run shell commands in Python.
If you just need to quickly run a command, you can use the ``shell`` shortcut
function::
    # >>> from shell import shell
    >>> ls = shell('ls')
    >>> for file in ls.output():
    ...     print file
    'another.txt'
If you need to extend the behavior, you can also use the ``Shell`` object::
    # >>> from shell import Shell
    >>> sh = Shell(has_input=True)
    >>> cat = sh.run('cat -u')
    >>> cat.write('Hello, world!')
    >>> cat.output()
    ['Hello, world!']
#
https://github.com/toastdriven/shell/blob/master/shell.py
https://stackoverflow.com/questions/4256107/running-bash-commands-in-python/51950538
"""
import shlex
import subprocess
import sys


class ShellException(Exception):
    """The base exception for all shell-related errors."""
    pass


class MissingCommandException(ShellException):
    """Thrown when no command was setup."""
    pass


class CommandError(ShellException):
    """Thrown when a command fails."""

    def __init__(self, message, code, stderr):
        self.message = message
        self.code = code
        self.stderr = stderr
        super(CommandError, self).__init__(message)


class Shell(object):
    """
    Handles executing commands & recording output.
    Optionally accepts a ``has_input`` parameter, which should be a boolean.
    If set to ``True``, the command will wait to execute until you call the
    ``Shell.write`` method & send input. (Default: ``False``)
    Optionally accepts a ``record_output`` parameter, which should be a boolean.
    If set to ``True``, the stdout from the command will be recorded.
    (Default: ``True``)
    Optionally accepts a ``record_errors`` parameter, which should be a boolean.
    If set to ``True``, the stderr from the command will be recorded.
    (Default: ``True``)
    Optionally accepts a ``strip_empty`` parameter, which should be a boolean.
    If set to ``True``, only non-empty lines from ``Shell.output`` or
    ``Shell.errors`` will be returned. (Default: ``True``)
    Optionally accepts a ``die`` parameter, which should be a boolean.
    If set to ``True``, raises a CommandError if the command exits with a
    non-zero return code. (Default: ``False``)
    Optionally accepts a ``verbose`` parameter, which should be a boolean.
    If set to ``True``, prints stdout to stdout and stderr to stderr as
    execution happens. (Default: ``False``)
    """

    def __init__(self, has_input=False, record_output=True, record_errors=True,
                 strip_empty=True, die=False, verbose=True):
        self.has_input = has_input
        self.record_output = record_output
        self.record_errors = record_errors
        self.strip_empty = strip_empty
        self.die = die
        self.verbose = verbose

        self.last_command = ''
        self.line_breaks = '\n'
        self.pid = None
        self.code = 0
        self._popen = None
        self._stdout = ''
        self._stderr = ''

    def _split_command(self, command):
        """
        Splits a string command into the individual args to pass to ``Popen``.
        If the ``command`` is an array, it is left untouched.
        """
        if isinstance(command, (tuple, list)):
            return command

        return shlex.split(command)

    def _handle_output(self, stdout, stderr):
        """
        Takes stdout/stderr & optionally retains them internally.
        Requires a ``stdout`` parameter, which should either be the output as
        a string or ``None``.
        Requires a ``stderr`` parameter, which should either be the errors as
        a string or ``None``.
        Records nothing if the ``record_*`` options have been set to ``False``.
        """
        if stdout is not None:
            if self.record_output:
                self._stdout += stdout
            if self.verbose:
                sys.stdout.write(stdout)
                sys.stdout.flush()

        if stderr is not None:
            if self.record_errors:
                self._stderr += stderr
            if self.verbose:
                sys.stderr.write(stderr)
                sys.stderr.flush()

    def _communicate(self, the_input=None):
        """
        Handles communicating with the process & optionally sends input.
        Optionally accepts a ``the_input`` parameter, which can be a string
        to send to the process. (Default: ``None``)
        """
        stdout, stderr = self._popen.communicate(input=the_input)
        self._handle_output(stdout, stderr)

        if self._popen.returncode is not None:
            self.code = self._popen.returncode

        if self.die and self.code != 0:
            raise CommandError(
                message='Command exited with code {}'.format(self.code),
                code=self.code,
                stderr=stderr)

    def run(self, command):
        """
        Runs a given command.
        Requires a ``command`` parameter should be either a string command
        (easier) or an array of arguments to send as the command (if you know
        what you're doing).
        Returns the ``Shell`` instance.
        Example::
            # >>> from shell import Shell
            >>> sh = Shell()
            >>> sh.run('ls -alh')
        """
        self.last_command = command
        command_bits = self._split_command(command)
        kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True,
        }

        if self.has_input:
            kwargs['stdin'] = subprocess.PIPE

        self._popen = subprocess.Popen(
            command_bits,
            **kwargs
        )
        self.pid = self._popen.pid

        if not self.has_input:
            self._communicate()

        return self

    def write(self, the_input):
        """
        If you're working with an interactive process, sends that input to
        the process.
        This needs to be used in conjunction with the ``has_input=True``
        parameter.
        Requires a ``the_input`` parameter, which should be a string of the
        input to send to the command.
        Returns the ``Shell`` instance.
        Example::
            # >>> from shell import Shell
            >>> sh = Shell(has_input=True)
            >>> sh.run('cat -u')
            >>> sh.write('Hello world!')
        """
        if not self._popen:
            raise MissingCommandException(
                "No command has been provided. Please call ``run`` first."
            )

        self._communicate(the_input)
        return self

    def kill(self):
        """
        Kills a given process.
        Example::
            # >>> from shell import Shell
            >>> sh = Shell()
            >>> sh.run('some_long_running_thing')
            >>> sh.kill()
        """
        if not self._popen:
            raise MissingCommandException(
                "No command has been provided. Please call ``run`` first."
            )

        self._popen.kill()
        return self

    def output(self, raw=True):
        """
        Returns the output from running a command.
        Optionally accepts a ``raw`` parameter, which should be a boolean. If
        ``raw`` is set to ``False``, you get an array of lines of output. If
        ``raw`` is set to ``True``, the raw string of output is returned.
        (Default: ``False``)
        Example::
            # >>> from shell import Shell
            >>> sh = Shell()
            >>> sh.run('ls ~')
            >>> sh.output()
            [
                'hello.txt',
                'world.txt',
            ]
        """
        if raw:
            return self._stdout

        lines = self._stdout.split(self.line_breaks)

        if self.strip_empty:
            lines = [line for line in lines if line]

        return lines

    def errors(self, raw=True):
        """
        Returns the errors from running a command.
        Optionally accepts a ``raw`` parameter, which should be a boolean. If
        ``raw`` is set to ``False``, you get an array of lines of errors. If
        ``raw`` is set to ``True``, the raw string of errors is returned.
        (Default: ``False``)
        Example::
            # >>> from shell import Shell
            >>> sh = Shell()
            >>> sh.run('ls /there-s-no-way-anyone/has/this/directory/please')
            >>> sh.errors()
            [
                'ls /there-s-no-way-anyone/has/this/directory/please: No such file or directory'
            ]
        """
        if raw:
            return self._stderr

        lines = self._stderr.split(self.line_breaks)

        if self.strip_empty:
            lines = [line for line in lines if line]

        return lines


class CustomShell(Shell):

    def run_shell(self, command: str):
        # stdout = subprocess.DEVNULL  # ignore output
        kwargs = {
            'stdout': sys.stdout,
            'stderr': sys.stderr,
            'text': True,  # stdout as string not bytes
        }
        res = subprocess.run(command, shell=True, **kwargs)  # best way
        # res.stdout, res.stderr
        self.code = res.returncode
        return self

    def run_stream_stdout(self, command):
        """
        如果使用subprocess.PIPE，执行完命令后才能拿到标准输出，替换成sys.stdout就能达到实时输出效果
        可以做到实时
        https://juejin.cn/post/6926442577294000136
        """
        self.last_command = command
        command_bits = self._split_command(command)
        kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            # 'text': True  # stdout as string not bytes
        }
        self._popen = subprocess.Popen(
            command_bits,
            **kwargs
        )
        self.pid = self._popen.pid

        def output():
            for line in iter(self._popen.stdout.readline, b''):
                line = line.decode(sys.stdout.encoding)
                yield line

        self.output = lambda: output()

        return self


def shell(command, has_input=False, record_output=True, record_errors=True,
          strip_empty=True, die=False, verbose=True,
          stream_output=False, shell=False
          ):
    """
    A convenient shortcut for running commands.
    Requires a ``command`` parameter should be either a string command
    (easier) or an array of arguments to send as the command (if you know
    what you're doing).
    Optionally accepts a ``has_input`` parameter, which should be a boolean.
    If set to ``True``, the command will wait to execute until you call the
    ``Shell.write`` method & send input. (Default: ``False``)
    Optionally accepts a ``record_output`` parameter, which should be a boolean.
    If set to ``True``, the stdout from the command will be recorded.
    (Default: ``True``)
    Optionally accepts a ``record_errors`` parameter, which should be a boolean.
    If set to ``True``, the stderr from the command will be recorded.
    (Default: ``True``)
    Optionally accepts a ``strip_empty`` parameter, which should be a boolean.
    If set to ``True``, only non-empty lines from ``Shell.output`` or
    ``Shell.errors`` will be returned. (Default: ``True``)
    Optionally accepts a ``die`` parameter, which should be a boolean.
    If set to ``True``, raises a CommandError if the command exits with a
    non-zero return code. (Default: ``False``)
    Optionally accepts a ``verbose`` parameter, which should be a boolean.
    If set to ``True``, prints stdout to stdout and stderr to stderr as
    execution happens. (Default: ``False``)
    Returns the ``Shell`` instance, which has been run with the given command.
    Example::
        # >>> from shell import shell
        >>> sh = shell('ls -alh *py')
        >>> sh.output()
        ['hello.py', 'world.py']
    """
    sh = CustomShell(
        has_input=has_input,
        record_output=record_output,
        record_errors=record_errors,
        strip_empty=strip_empty,
        die=die,
        verbose=verbose
    )
    if verbose:
        print("* sh command: {}".format(command))
    if stream_output:
        return sh.run_stream_stdout(command)
    elif '&&' in command or '||' in command or shell is True:
        return sh.run_shell(command)
    return sh.run(command)
