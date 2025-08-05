
# 🤖 Zephyr Math Chatbot

Welcome to the **Zephyr Math Chatbot** – a lightweight and intelligent math question-answering assistant powered by the [Zephyr model](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) and fine-tuned using LoRA. This project allows users to interact with the chatbot via a **Streamlit** web interface.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)
![LoRA](https://img.shields.io/badge/LoRA-F08536?style=for-the-badge)
![Tokenizer](https://img.shields.io/badge/Tokenizer-BF40BF?style=for-the-badge)

---

## 📁 Project Structure

```text
zephyr-math-chatbot/
├── zephyr_lora_adapter/       # LoRA adapter with Zephyr model tuning
│   └── tokenizer.json         # Custom tokenizer config
├── app.py                     # Streamlit app code
├── gsm8k_clean.csv            # Cleaned dataset of math problems (GSM8K format)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

---

## 🚀 Features

- Chat interface to ask math questions interactively
- Backed by the Zephyr 7B LLM using HuggingFace Transformers
- Fine-tuned with LoRA adapters for improved performance on math tasks
- Uses a cleaned GSM8K dataset for evaluation
- Lightweight and easy to deploy locally or on the web

---

## 🛠️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/nagasriramnani/zephyr-math-chatbot.git
cd zephyr-math-chatbot
```

### 2. Create environment & install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📚 Dataset

This project uses a subset of the **GSM8K dataset** (grade school math questions) stored as `gsm8k_clean.csv`. You can modify or extend this dataset to improve or specialize the chatbot.

---

## 📸 Preview

> Coming soon — Live demo screenshots or Streamlit deployment link!

---

## 🙋‍♂️ Contact

Feel free to reach out if you want to collaborate, contribute, or have feedback.

- 📧 Email: [nagasriramkochetti@gmail.com](mailto:nagasriramkochetti@gmail.com)
- 🌐 GitHub: [nagasriramnani](https://github.com/nagasriramnani)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

Made with ❤️ by Naga Sri Ram Kochetti
