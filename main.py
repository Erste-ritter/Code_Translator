import re
import os
import io
import logging
import datetime
import numpy as np
from groq import Groq
import soundfile as sf
import streamlit as st
from pydub import AudioSegment
import speech_recognition as sr
from streamlit.components.v1 import html
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Вывод логов в консоль
    ]
)
logger = logging.getLogger(__name__)

# Настройка Groq API
#Тут должен быть указан ваш ключ Groq
client = Groq()

# Список языков программирования
languages = ["py", "j", "cpp", "s", "go"]
language_full_name = {'py': 'Python', 'j': 'Java', 'cpp': 'C++', 's': 'Swift', 'go': 'Go'}

# Функции из вашего кода
def extract_languages_from_text(text):
    logger.info(f"Извлечение языков из текста: {text}")
    language_map = {
        'пайтон': 'py', 'python': 'py', 'питон': 'py',
        'джава': 'j', 'java': 'j',
        'си плюс плюс': 'cpp', 'c++': 'cpp',
        'свифт': 's', 'swift': 's',
        'го': 'go', 'go': 'go'
    }
    prompt = f"""
You are a language extraction assistant. Your task is to analyze the following Russian text and extract the source and target programming languages mentioned in it. Return the result in the format:
source_language_shorthand = '<code>'
target_language_shorthand = '<code>'

Supported languages and their shorthands:
- Python: 'py'
- Java: 'j'
- C++: 'cpp'
- Swift: 's'
- Go: 'go'

Text to analyze:
"{text}"
"""
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        timeout=60
    )
    response = completion.choices[0].message.content
    logger.info(f"Ответ Grok API: {response}")
    source_match = re.search(r"source_language_shorthand = '(\w+)'", response)
    target_match = re.search(r"target_language_shorthand = '(\w+)'", response)
    source_language_shorthand = source_match.group(1) if source_match else None
    target_language_shorthand = target_match.group(1) if target_match else None
    logger.info(f"Извлеченные языки: source={source_language_shorthand}, target={target_language_shorthand}")
    return source_language_shorthand, target_language_shorthand


def check_code_for_errors(code, lang):
    prompt = f"""
You are a strict code analysis assistant specialized in {lang} code.  Your ONLY task is to check for CRITICAL errors that cause runtime failures or affects the code launch. 
Ignore minor issues like style, formatting, or best practices. Input data is always correct and never empty.

IMPORTANT: Return ONLY the following format:
Errors found: X
where X is the number of critical errors.

Here is the {lang} code:
{code}

ONLY return "Errors found: X" with an integer number. DO NOT provide any explanations, comments, or extra text.
"""
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Делаем вывод детерминированным
    )

    response = completion.choices[0].message.content.strip()
    print(f"Raw response: {response}")  # Проверяем сырой ответ

    match = re.search(r"Errors found: (\d+)", response)
    error_count = int(match.group(1)) if match else 0
    return error_count


# Функция для конвертации кода
def translate_code(user_code, source_language_shorthand, target_language_shorthand):
    vector_storage = 'indexed_storage'
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    vector_store = FAISS.load_local(vector_storage, embedding_model, allow_dangerous_deserialization=True)

    language_full_name = {
        'py': 'Python',
        'j': 'Java',
        'cpp': 'C++',
        's': 'Swift',
        'go': 'Go'
    }

    def remove_comments(code, language_shorthand):
        if language_shorthand == 'py':
            comment_marker = '#'
        else:
            comment_marker = '//'
        lines = code.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith(comment_marker)]
        return '\n'.join(cleaned_lines)

    if source_language_shorthand not in language_full_name or target_language_shorthand not in language_full_name:
        print("Error: Invalid language shorthand provided.")
        return None

    results = vector_store.similarity_search(
        query=user_code,
        k=4,
        filter={"language": source_language_shorthand}
    )

    if not results:
        print(f"No matching code snippets found in {source_language_shorthand}.")
        return None

    task_ids = [doc.metadata['task_id'] for doc in results]
    examples = []
    for task_id in task_ids:
        source_file = os.path.join('Dataset', f"{str(task_id)}_{source_language_shorthand}.txt")
        target_file = os.path.join('Dataset', f"{str(task_id)}_{target_language_shorthand}.txt")
        if os.path.exists(source_file) and os.path.exists(target_file):
            with open(source_file, 'r') as f:
                source_code = f.read()
            with open(target_file, 'r') as f:
                target_code = f.read()
            source_code_clean = remove_comments(source_code, source_language_shorthand)
            target_code_clean = remove_comments(target_code, target_language_shorthand)
            examples.append(
                f"Example in {language_full_name[source_language_shorthand]}:\n{source_code_clean}\n\n"
                f"Translated to {language_full_name[target_language_shorthand]}:\n{target_code_clean}"
            )
        else:
            print(f"Warning: Files for task {task_id} not found for both languages.")

    if not examples:
        print("No translation examples found.")
        return None

    examples_text = "\n\n".join(examples)
    prompt = f"""
You are a code translation assistant. Your task is to translate the given {language_full_name[source_language_shorthand]} code into {language_full_name[target_language_shorthand]}. Below are examples of similar code snippets and their translations:

{examples_text}

Now, translate the following {language_full_name[source_language_shorthand]} code into {language_full_name[target_language_shorthand]}:

{user_code}
"""
    print(prompt)
    completion = client.chat.completions.create(
        model="DeepSeek-R1-Distill-Llama-70B",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    generated_code = completion.choices[0].message.content
    print(f"Generated {language_full_name[target_language_shorthand]} code:\n{generated_code}")
    return generated_code

def extract_code(response):
    logger.info("Извлечение кода из ответа...")
    match = re.search(r'```(?:\w+)?\s*(.*?)```', response, re.DOTALL)
    code = match.group(1).strip() if match else None
    logger.info(f"Извлеченный код: {code[:50] if code else 'Не удалось извлечь код'}...")
    return code


# Функция для преобразования аудио в текст
def audio_to_text(audio_data):
    logger.info("Начало преобразования аудио в текст...")
    recognizer = sr.Recognizer()
    try:
        # Сохраняем аудио на диск для отладки
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_raw_path = f"debug_raw_audio_{timestamp}.webm"
        with open(debug_raw_path, "wb") as f:
            f.write(audio_data)
        logger.info(f"Исходное аудио сохранено для отладки: {debug_raw_path}")

        # Конвертируем аудио в WAV с помощью pydub
        audio = AudioSegment.from_file(debug_raw_path, format="webm")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        audio_data_wav = wav_io.read()

        # Сохраняем WAV-файл для отладки
        debug_wav_path = f"debug_audio_{timestamp}.wav"
        with open(debug_wav_path, "wb") as f:
            f.write(audio_data_wav)
        logger.info(f"Аудио в формате WAV сохранено для отладки: {debug_wav_path}")

        # Читаем WAV-файл для распознавания
        with io.BytesIO(audio_data_wav) as f:
            data, samplerate = sf.read(f)
            logger.info(f"Аудиоданные: форма={data.shape}, частота дискретизации={samplerate}")
            # Если данные стерео, преобразуем в моно
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
                logger.info("Аудио преобразовано в моно.")
            # Сохраняем данные во временный WAV-файл для speech_recognition
            with io.BytesIO() as wav_io:
                sf.write(wav_io, data, samplerate, format='WAV')
                wav_io.seek(0)
                with sr.AudioFile(wav_io) as source:
                    audio = recognizer.record(source)
                    logger.info("Аудио успешно записано для распознавания.")
                    try:
                        text = recognizer.recognize_google(audio, language='ru-RU')
                        logger.info(f"Распознанный текст: {text}")
                        return text
                    except sr.UnknownValueError:
                        logger.error("Не удалось распознать речь.")
                        st.error("Не удалось распознать речь.")
                        return None
                    except sr.RequestError as e:
                        logger.error(f"Ошибка сервиса распознавания: {e}")
                        st.error(f"Ошибка сервиса распознавания: {e}")
                        return None
    except Exception as e:
        logger.error(f"Ошибка при обработке аудио: {e}")
        st.error(f"Ошибка при обработке аудио: {e}")
        return None

# Функция для перевода кода
def perform_translation():
    logger.info("Запуск перевода кода...")
    if not st.session_state["manual_code"]:
        logger.error("Код для перевода отсутствует.")
        st.error("Пожалуйста, введите код для перевода.")
        return False
    st.write("Проверка исходного кода на ошибки...")
    error_count = check_code_for_errors(st.session_state["manual_code"],
                                        language_full_name[st.session_state["selected_source_lang"]])
    if error_count > 0:
        logger.error("Исходный код содержит ошибки.")
        st.error("Исходный код содержит ошибки. Пожалуйста, исправьте их перед переводом.")
        return False
    st.write("Перевод кода...")
    translated_response = translate_code(st.session_state["manual_code"], st.session_state["selected_source_lang"],
                                         st.session_state["selected_target_lang"])
    if translated_response is None:
        logger.error("Не удалось перевести код.")
        st.error("Не удалось перевести код. Проверьте наличие данных в Dataset и FAISS.")
        return False
    translated_code = extract_code(translated_response)
    if translated_code is None:
        logger.error("Не удалось извлечь переведенный код из ответа.")
        st.error("Не удалось извлечь переведенный код из ответа.")
        return False
    st.write("Проверка переведенного кода на ошибки...")
    error_count = check_code_for_errors(translated_code, language_full_name[st.session_state["selected_target_lang"]])
    if error_count > 0:
        logger.warning("Переведенный код содержит ошибки.")
        st.warning("Переведенный код содержит ошибки. Пожалуйста, проверьте его.")
    st.session_state["translated_code"] = translated_code
    logger.info("Перевод успешно завершен.")
    return True

# Инициализация session_state
if "selected_source_lang" not in st.session_state:
    st.session_state["selected_source_lang"] = languages[0]  # По умолчанию 'py'
if "selected_target_lang" not in st.session_state:
    st.session_state["selected_target_lang"] = languages[1]  # По умолчанию 'j'
if "manual_code" not in st.session_state:
    st.session_state["manual_code"] = ""  # Инициализация для ручного ввода кода
if "translated_code" not in st.session_state:
    st.session_state["translated_code"] = ""  # Инициализация для переведенного кода

# Streamlit

# Заголовок
st.title("Конвертер кода между языками программирования")
st.markdown("""
Это приложение позволяет конвертировать код из одного языка программирования в другой. 
Вы можете использовать голосовой ввод или ввести код вручную.
""")
st.markdown("---")  # Горизонтальная линия для визуального разделения

# Выбор способа ввода
st.markdown("<p style='font-size: 16px;'>Выберите способы ввода:</p>", unsafe_allow_html=True)
input_method = st.radio(
    "Способ ввода",
    ["Голосовой ввод", "Ручной ввод"],
    help="Выберите, как вы хотите ввести код: с помощью голоса или вручную."
)

# Динамические блоки в зависимости от выбора способа ввода
if input_method == "Голосовой ввод":
    # Первая строка: загрузка аудиофайла
    uploaded_audio = st.file_uploader(
        "Загрузите аудиофайл (формат: .webm или .wav)",
        type=["webm", "wav"],
        help="Загрузите аудиофайл, содержащий голосовую команду, например: 'Переведи код с Python на Java'."
        )    
    
    # Вторая строка: кнопки для записи аудио
    st.write("Или запишите аудио:")
    
    audio_recorder_html = """
    <div>
        <button id="startRecording" onclick="startRecording()">Начать запись</button>
        <button id="stopRecording" onclick="stopRecording()" disabled>Остановить запись</button>
        <a id="downloadLink" style="display: none;"></a>
        <p id="status" style="font-family: &quot;Source Sans Pro&quot;, sans-serif;">Нажмите 'Начать запись', чтобы записать голосовую команду. После завершения записи нажмите 'Остановить запись' и загрузите файл.</p>
    </div>

    <style>
    /* Стили для кнопки "Начать запись" */
    #startRecording {
        background-color: #4CAF50; /* Зеленый цвет */
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 18px;
        margin: 0px;
    }

    #startRecording:hover {
        background-color: #45a049; /* Темно-зеленый при наведении */
    }

    #startRecording:disabled {
        background-color: #cccccc; /* Серый для отключенных кнопок */
        cursor: not-allowed;
    }

    /* Стили для кнопки "Остановить запись" */
    #stopRecording {
        background-color: #f44336; /* Красный цвет */
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 18px;
        margin: 5px;
    }

    #stopRecording:hover {
        background-color: #d32f2f; /* Темно-красный при наведении */
    }

    #stopRecording:disabled {
        background-color: #cccccc; /* Серый для отключенных кнопок */
        cursor: not-allowed;
    }

    /* Стили для текста статуса */
    #status {
        font-size: 16px;
        color: #333333;
        margin: 10px 0;
        font-family: "Source Sans Pro", sans-serif;
    }

    /* Стили для статуса записи */
    .recording {
        color: red;
        font-weight: bold;
    }
    </style>

    <script>
    let mediaRecorder;
    let audioChunks = [];

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const downloadLink = document.getElementById('downloadLink');
                downloadLink.href = audioUrl;
                downloadLink.download = 'recorded_audio.webm';
                downloadLink.click();
                document.getElementById('status').innerText = 'Запись завершена. Загрузите файл ниже.';
            };

            mediaRecorder.start();
            document.getElementById('startRecording').disabled = true;
            document.getElementById('stopRecording').disabled = false;
            document.getElementById('status').innerText = 'Идет запись...';
        } catch (err) {
            document.getElementById('status').innerText = 'Ошибка: ' + err.message;
        }
    }

    function stopRecording() {
        mediaRecorder.stop();
        document.getElementById('startRecording').disabled = false;
        document.getElementById('stopRecording').disabled = true;
    }
    </script>
    """
    html(audio_recorder_html)

    st.markdown("<p style='font-size: 20px;'>ИЛИ</p>", unsafe_allow_html=True)

    # Третья строка: блок для загрузки файла исходного кода
    uploaded_file = st.file_uploader(
        "Загрузите файл с исходным кодом (форматы: .txt, .py, .java, .cpp, .js)",
        type=["txt", "py", "java", "cpp", "js"],
        help="Загрузите файл с кодом, который вы хотите преобразовать."
        )
    if uploaded_file is not None:
        code = uploaded_file.read().decode("utf-8")
        st.session_state["manual_code"] = code

else:  # Ручной ввод
    # Создаем две колонки для выбора языков и загрузки файлов
    col1, col2 = st.columns(2)

    with col1:
        # Выбор исходного языка
        source_lang_index = languages.index(st.session_state["selected_source_lang"])
        source_language = st.selectbox(
            "Выберите исходный язык",
            languages,
            index=source_lang_index,
            key="source_lang",
            format_func=lambda x: language_full_name[x]
        )
        # Обновляем selected_source_lang при изменении пользователем
        if source_language != st.session_state["selected_source_lang"]:
            st.session_state["selected_source_lang"] = source_language

        # Блок для загрузки файла исходного кода
        uploaded_file = st.file_uploader("Загрузите файл с исходным кодом", type=["txt", "py", "java", "cpp", "js"])
        if uploaded_file is not None:
            code = uploaded_file.read().decode("utf-8")
            st.session_state["manual_code"] = code

    with col2:
        # Выбор языка преобразования
        target_lang_index = languages.index(st.session_state["selected_target_lang"])
        target_language = st.selectbox(
            "Выберите язык преобразования",
            languages,
            index=target_lang_index,
            key="target_lang",
            format_func=lambda x: language_full_name[x]
        )
        # Обновляем selected_target_lang при изменении пользователем
        if target_language != st.session_state["selected_target_lang"]:
            st.session_state["selected_target_lang"] = target_language
        
        st.markdown("<p style='font-size: 14px;'>Кнопка для копирования скоро появится :)</p>", unsafe_allow_html=True)

        # Кнопка для загрузки преобразованного кода
        if st.session_state.get("translated_code"):
            st.download_button(
                label="📥 Скачать преобразованный код",
                data=st.session_state["translated_code"],
                file_name=f"translated_code_{language_full_name[target_language]}.txt",
                mime="text/plain",
                key="download_icon",
                help="Скачать преобразованный код",
                use_container_width=True
            )

# Две колонки для исходного и преобразованного кода
st.markdown("---")  # Горизонтальная линия для визуального разделения
st.header("Исходный и преобразованный код")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Исходный код")
    if input_method == "Ручной ввод":
        manual_code = st.text_area("Введите исходный код", value=st.session_state.get("manual_code", ""), height=400)
        st.session_state["manual_code"] = manual_code
    else:
        if "manual_code" in st.session_state:
            st.text_area("Исходный код", value=st.session_state["manual_code"], height=400, disabled=True)

with col4:
    st.subheader("Преобразованный код")
    if "translated_code" in st.session_state:
        st.text_area("Преобразованный код", value=st.session_state["translated_code"], height=400, disabled=True)

# Кнопка для конвертации
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("Конвертировать код", key="translate_button"):
    if perform_translation():
        st.success("Код успешно преобразован!")
        # Обновляем текстовое поле с преобразованным кодом
        st.session_state["translated_code"] = st.session_state.get("translated_code", "")
    else:
        st.error("Ошибка при преобразовании кода.")
