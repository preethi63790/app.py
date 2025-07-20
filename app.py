{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/preethi63790/app.py/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Fake News Detector\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import string\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.metrics import accuracy_score, f1_score\n",
        "nltk.download('stopwords')\n",
        "stop_words = set(stopwords.words('english'))\n",
        "@st.cache_data\n",
        "def load_data():\n",
        "    fake_df = pd.read_csv(\"Fake.csv\")\n",
        "    true_df = pd.read_csv(\"True.csv\")\n",
        "\n",
        "    fake_df[\"label\"] = 1\n",
        "    true_df[\"label\"] = 0\n",
        "\n",
        "    df = pd.concat([fake_df, true_df], ignore_index=True)\n",
        "    df = df[[\"text\", \"label\"]].dropna()\n",
        "    return df\n",
        "df = load_data()\n",
        "def clean_text(text):\n",
        "    text = text.lower()\n",
        "    text = ''.join(ch for ch in text if ch not in string.punctuation)\n",
        "    tokens = text.split()\n",
        "    tokens = [word for word in tokens if word not in stop_words]\n",
        "    return ' '.join(tokens)\n",
        "df['clean_text'] = df['text'].apply(clean_text)\n",
        "vectorizer = TfidfVectorizer()\n",
        "X = vectorizer.fit_transform(df['clean_text'])\n",
        "y = df['label']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "model = MultinomialNB()\n",
        "model.fit(X_train, y_train)\n",
        "y_pred = model.predict(X_test)\n",
        "acc = accuracy_score(y_test, y_pred)\n",
        "f1 = f1_score(y_test, y_pred)\n",
        "print(\"Model trained successfully!\")\n",
        "print(f\"Accuracy: {acc:.2f}\")\n",
        "print(f\"F1 Score: {f1:.2f}\")\n",
        "st.title(\"Fake News Detector\")\n",
        "st.markdown(\"Enter a news article below to check if it's **Fake** or **Real**.\")\n",
        "user_input = st.text_area(\"Enter news article text:\", height=200)\n",
        "if st.button(\"Predict\"):\n",
        "    if user_input.strip() == \"\":\n",
        "        st.warning(\"Please enter some news text.\")\n",
        "    else:\n",
        "        cleaned = clean_text(user_input)\n",
        "        input_vec = vectorizer.transform([cleaned])\n",
        "        prediction = model.predict(input_vec)[0]\n",
        "        confidence = model.predict_proba(input_vec).max()\n",
        "\n",
        "        result = \"Fake News\" if prediction == 1 else \"Real News\"\n",
        "        st.success(f\"**Prediction:** {result}\")\n",
        "        st.info(f\"**Confidence:** {confidence * 100:.2f}%\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cCzLVFAECDee",
        "outputId": "404b1299-c547-4b73-f2f0-6b7e8dbeaf05"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n",
            "2025-07-20 09:03:04.966 No runtime found, using MemoryCacheStorageManager\n",
            "2025-07-20 09:03:04.968 No runtime found, using MemoryCacheStorageManager\n",
            "2025-07-20 09:03:04.969 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:04.970 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:04.971 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:04.972 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:05.478 Thread 'Thread-8': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:05.479 Thread 'Thread-8': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:05.480 Thread 'Thread-8': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:06.690 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:06.691 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:06.692 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.790 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.791 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.792 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.793 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.794 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.795 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.795 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.796 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.797 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.798 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.799 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.800 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.801 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.802 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.803 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.804 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-07-20 09:03:31.805 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model trained successfully!\n",
            "Accuracy: 0.95\n",
            "F1 Score: 0.95\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3507bf80"
      },
      "source": [
        "# Install pyngrok\n",
        "!pip install pyngrok -q"
      ],
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "b943543d",
        "outputId": "3ab5fca0-3bfd-4652-b217-f125f46518c7"
      },
      "source": [
        "from pyngrok import ngrok\n",
        "import os\n",
        "from google.colab import userdata\n",
        "NGROK_AUTH_TOKEN = userdata.get(\"NGROK_AUTH_TOKEN\")\n",
        "ngrok.set_auth_token(NGROK_AUTH_TOKEN)\n",
        "ngrok.kill()\n",
        "public_url = ngrok.connect(8501)\n",
        "print(f\"Ngrok tunnel established at: {public_url}\")\n",
        "!streamlit run app.py &>/dev/null &"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ngrok tunnel established at: NgrokTunnel: \"https://9ab458461e19.ngrok-free.app\" -> \"http://localhost:8501\"\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "jupyter nbconvert --to script app.py.ipynb"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 106
        },
        "id": "Rvy3jXTTR9Us",
        "outputId": "6c870bf1-3ef9-4be4-c564-bfeeb4213a99"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "error",
          "ename": "SyntaxError",
          "evalue": "invalid syntax (ipython-input-26-560165342.py, line 1)",
          "traceback": [
            "\u001b[0;36m  File \u001b[0;32m\"/tmp/ipython-input-26-560165342.py\"\u001b[0;36m, line \u001b[0;32m1\u001b[0m\n\u001b[0;31m    jupyter nbconvert --to script app.py.ipynb\u001b[0m\n\u001b[0m            ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m invalid syntax\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPCE9RG+AQUteGH73hP/vZx",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}