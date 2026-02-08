Before you start a server, build all the static files by running the following command:
```bash
poetry run build_static
```

To start the development server, run the following command:
```bash
poetry run dev
```

If you want to specify the path to the SVETlANNa repository, you can use the `--sv-dir` option:
```bash
poetry run dev --sv-dir ../SVETlANNa/svetlanna
```
by default it is `../SVETlANNa/svetlanna`, so you can use `poetry run dev --sv-dir` as a shorthand.
