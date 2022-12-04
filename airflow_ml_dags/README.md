Homework 3
==============================

# Для корректной работы с переменными, созданными из UI
~~~
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
~~~

# Запуск приложения:
~~~
docker compose up --build
~~~