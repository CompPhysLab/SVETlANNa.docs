[![en](https://img.shields.io/badge/lang-EN-blue.svg)](https://github.com/CompPhysLab/SVETlANNa.docs/blob/main/README.md)
[![ru](https://img.shields.io/badge/lang-RU-green.svg)](https://github.com/CompPhysLab/SVETlANNa.docs/blob/main/README.ru.md)

# SVETlANNa.docs

Этот репозиторий содержит примеры применения для библиотеки [SVETlANNa](https://github.com/CompPhysLab/SVETlANNa/blob/main).

Русифицированные примеры разположены в папке [examples_ru](https://github.com/CompPhysLab/SVETlANNa.docs/tree/main/examples_ru).

**ВНИМАНИЕ!** Примеры из папки `pipelines` содержат обучение моделей, которое занимает продолжительное время.
Для удобства к некоторым примерам идут примеры, которые не содержат процесс обучения, а загружают веса уже обученных моделей, поэтому
* вместо `pipelines/03_mnist_experiments.ipynb` можно сразу запустить `pipelines/03_load_experiments.ipynb`
* вместо `pipelines/07_weizmann_drnn.ipynb` можно сразу запустить `pipelines/07_load_weizmann_drnn.ipynb`
* вместо `pipelines/08_autoencoder.ipynb` можно сразу запустить `pipelines/08_load_autoencoder.ipynb`
* в примерах `pipelines/02_mnist_by_ozcan.ipynb` и `pipelines/02_mnist_mse.ipynb` можно пропустить запуск ячеек из разделов `4.2.2` и `4.2.3`, переходя сразу к разделу `5`, чтобы пропустить длительный процесс обучения.

**ВНИМАНИЕ!** Запуск примера `GPU/GPU_512_mnist_mse.ipynb` требует около двух дней расчетов на видеокарте RTX 4090.
Для проверки работы GPU лучше использовать вычислительно более простой пример, например `GPU/gpu_usage_example.ipynb`.

---
Большинство примеров можно запускать в облаке без предварительной установки, используя бесплатные серверы Google Colab.
Для этого достаточно открыть ссылку из списка ниже и последовательно выполнить ячейки.

При первом запуске блокнота в Google Colab может появиться предупреждение с просьбой перезапустить сессию для установки обновлённых версий пакетов.
Нужно согласиться и разрешить перезапуск сессии.

- GPU:
    - [gpu_usage_example.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/GPU/gpu_usage_example.ipynb)
    - Остальные примеры могут быть запущены только локально

- clerk:
    - [example.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/clerk/example.ipynb)

- custom_elements:
    - [hello_world_element.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/custom_elements/hello_world_element.ipynb)

- free_propagation:
    - [apertures.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/free_propagation/apertures.ipynb)
    - [gaussian_beam_propagation.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/free_propagation/gaussian_beam_propagation.ipynb)
    - `lens.ipynb` может быть запущен только локально
    - [nonlinear_element.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/free_propagation/nonlinear_element.ipynb)
    - [rectangular_slit_propagation.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/free_propagation/rectangular_slit_propagation.ipynb)
    - [slm.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/free_propagation/slm.ipynb)
    - `square_aperture_propagation.ipynb` может быть запущен только локально

- phase_retrieval:
    - [optimize_example.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/phase_retrieval/optimize_example.ipynb)
    - [optimize_lowapi_example.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/phase_retrieval/optimize_lowapi_example.ipynb)

- pipelines:
    - Все примеры могут быть запущены только локально

- specs:
    - [example.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/specs/example.ipynb)

- tensor_axes:
    - [example.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/tensor_axes/example.ipynb)

- visualization:
    - [example.ipynb](https://colab.research.google.com/github/CompPhysLab/SVETlANNa.docs/blob/main/examples_ru/visualization/example.ipynb)

# Установки и требования

См. файлы [README.ru.md](https://github.com/CompPhysLab/SVETlANNa/blob/main/README.ru.md) основного репозитория.

# Вклад в разработку

Улучшение библиотеки и разработка новых модулей приветствуются (см. файлы `contributing.md`).

# Бладарности

Работа над данным проектом была поддержана [Фондом содействия инновациям](https://en.fasie.ru/)

# Авторы

- [@aashcher](https://github.com/aashcher)
- [@alexeykokhanovskiy](https://github.com/alexeykokhanovskiy)
- [@Den4S](https://github.com/Den4S)
- [@djiboshin](https://github.com/djiboshin)
- [@Nevermind013](https://github.com/Nevermind013)

# Лицензия

[Mozilla Public License Version 2.0](https://www.mozilla.org/en-US/MPL/2.0/)
