# Dermind AI Endpoint

`Dermind AI Endpoint` adalah backend API berbasis **FastAPI** yang berfungsi sebagai pusat layanan kecerdasan buatan (AI) untuk platform Dermind. API ini mengintegrasikan tiga modul utama: analisis kesehatan kulit, analisis kesehatan mental, dan fitur chatbot interaktif.

Seluruh endpoint pada API ini dilindungi menggunakan sistem autentikasi **API Key** berbasis header demi menjaga keamanan data pengguna.

---

## 🚀 Fitur Utama

* **Skin Analysis Router (`/skin/infer`):** Endpoint untuk memproses dan menganalisis kondisi kesehatan kulit.
* **Mental Health Router (`/mental/infer`):** Endpoint untuk evaluasi atau skrining awal kesehatan mental.
* **Chatbot Router (`/chat`):** Endpoint asisten virtual (AI Chatbot) untuk interaksi pengguna.
* **Secure API Key Gate:** Proteksi global menggunakan header khusus untuk mencegah akses tidak sah.

---

## 🛠️ Teknologi yang Digunakan

* **Python 3.11**
* **FastAPI:** Framework web modern, cepat (high-performance), untuk membangun API.
* **Uvicorn:** Server ASGI cepat untuk menjalankan FastAPI.
* **Python-dotenv:** Untuk manajemen environment variables secara aman.

---

## 📋 Struktur Projek (Arsitektur Modular)

Berdasarkan struktur routing pada `main.py`, projek ini menggunakan pendekatan modular:

```text
dermind-ai-endpoint/
├── routers/
│   ├── skin.py       # Logika & Endpoint Analisis Kulit
│   ├── mental.py     # Logika & Endpoint Kesehatan Mental
│   └── chatbot.py    # Logika & Endpoint Chatbot AI
├── .env              # File konfigurasi sensitif (API Key)
├── main.py           # Entry point aplikasi & konfigurasi keamanan
└── requirements.txt  # Daftar dependensi library
