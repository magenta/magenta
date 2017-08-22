## Building a new pip package

### setup.py updates
Update the `REQUIRED_PACKAGES` list in the same file to ensure that all of our
dependencies are listed and that they match the versions of the packages
referenced in the Bazel `WORKSPACE` file. Also check that the correct version of
tensorflow is listed.

### Building the package
```
bazel build //magenta/tools/pip:build_pip_package
bazel-bin/magenta/tools/pip/build_pip_package /tmp/magenta_pkg
bazel-bin/magenta/tools/pip/build_pip_package /tmp/magenta_pkg --gpu
```

Before this next step, make sure your preferred virtualenv or conda environment
is activated.

```
pip install -U /tmp/magenta_pkg/magenta-N.N.N-py2-none-any.whl
```

Next, test that it worked:

```
# cd outside of the magenta repo.
$ python
>>> import magenta
>>> magenta.__version__
```

Verify that the version of the installed package matches the new version number.

Do the same test for the `magenta-gpu` package. The only difference with the
gpu version of the package is that it depends on `tensorflow-gpu` instead of
`tensorflow`.

### Upload the new version to pypi
```
twine upload /tmp/magenta_pkg/magenta-N.N.N-py2-none-any.whl
twine upload /tmp/magenta_pkg/magenta-gpu-N.N.N-py2-none-any.whl
```

After this step, anyone should be able to `pip install magenta` and get the
latest version.

## Adding to the pip package

### Libraries

As a convention, libraries that we want to expose externally through the pip
package should be listed as dependencies in otherwise empty `py_library`
targets that share the same name as their directory. Those targets should then
be referenced as dependencies by the target for the directory above them, all
the way up to the root `//magenta` target. (Targets named the same as their
package are implicit; `//magenta` is short for `//magenta:magenta`.)

This allows us to list `//magenta` as the main `py_library` dependency for
building the pip package and distributes maintenance of public API dependencies
to each package.

For example, to expose `magenta.pipelines.statistics`, the
`magenta/pipelines/BUILD` file will have a section that looks like this:

```
# The Magenta public API.
py_library(
    name = "pipelines",
    deps = [
        ":statistics",
    ],
)
```

And the `magenta/BUILD` file will have a section that looks like this:

```
# The Magenta public API.
py_library(
    name = "magenta",
    visibility = ["//magenta:__subpackages__"],
    deps = [
        "//magenta/pipelines",
    ],
)
```

Because we want `import magenta` to make all modules immediately available,
we also need to create a custom `__init__.py` in the magenta directory and
reference it in the `srcs` field for the `//magenta:magenta` target.

When you add a new module to be exported, you'll also need to add it to this
`__init__.py` file. There are instructions in that file for how to
automatically generate the list based on the python library dependencies.

Now the `//magenta/tools/pip:build_pip_package` target just needs to depend on
`//magenta`.

The `//magenta` dependency also provides an easy way to verify that the models
developed within the magenta repo use the same code that is available to
external developers. Rather than depend directly on library targets, models
should depend only on the `//magenta` target.

Libraries should continue to use dependencies like normal: one target for every
python file, and every `import` statement should have a corresponding
dependency. This ensures we avoid circular dependencies and also makes builds
faster for tests.

### Scripts

Our pip package also includes several executable scripts (e.g.,
`convert_dir_to_note_sequences`). These are just python files that pip
create executable wrappers around and installs to the python binary path. After
installation, users will have the script installed in their path. To add a new
script to the distribution, follow these steps:

First, add the script as a data dependency to the
`//magenta/tools/pip:build_pip_package` target. You will likely also need to
modify the visibility of the script's target so the pip builder can see it by
adding this line to the script's target:

```
visibility = ["//magenta/tools/pip:__subpackages__"],
```

Next, modify `setup.py` so the script's python module is listed under
`CONSOLE_SCRIPTS`.

Finally, you will need to modify the script itself so that it can be invoked
either directly or by the pip-generated wrapper script. The pip wrapper will
look for a method called `console_entry_point` as defined in `setup.py`, but
running the script directly (e.g., after a bazel build'ing it) will just invoke
the `if __name__ == '__main__':` condition. Both of those need to trigger
`tf.app.run` to run the actual `main` function because `tf.app.run` takes care
of things like initializing flags. The easiest way to do this is by adding the
following snippet to the end of your script:

```python
def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
```
